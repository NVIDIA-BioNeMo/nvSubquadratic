"""Standalone FLOP measurement for a WELL v2 experiment.

Loads a config the same way ``experiments/run.py`` does, instantiates the
datamodule + network + Lightning wrapper, pulls one real training batch,
and runs a single forward + backward inside
``torch.utils.flop_counter.FlopCounterMode``.  No wandb, no Lightning
trainer, no torch.compile — just the FLOP numbers, printed to stdout and
written to a JSON file.

Use this when the SLURM run cannot reach wandb (so the in-training
``FlopCounterCallback`` never gets to log) but you still need the per-step
FLOP counts for a config.

Usage:
    PYTHONPATH=. python benchmarks/well/measure_flops.py \
        --config examples/well/v2/gray_scott_reaction_diffusion/hyena_gaussian_mask.py \
        [--out flops.json] \
        [--device cuda] \
        [overrides ...]

Notes:
    * ``torch.compile`` is intentionally skipped: ``FlopCounterMode`` is a
      ``TorchDispatchMode`` and is most reliable in eager mode.  The
      eager-mode FLOP count is the ground truth — compile changes how
      ops execute, not how many FLOPs they perform.
    * With ``gradient_checkpointing=True`` the backward re-runs the
      checkpointed forward chunks; those re-runs are counted in
      ``flops/bwd``, which is the correct accounting for true per-step
      training cost.
    * The script picks a single batch from ``datamodule.train_dataloader()``.
      Per-rank FLOPs scale with the batch size in that dataloader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.utils.cli import (
    apply_config_overrides,
    load_config_from_file,
    verify_no_interpolator_overwrites,
)
from nvsubquadratic.lazy_config import instantiate


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the FLOP measurement script."""
    parser = argparse.ArgumentParser(description="Measure per-step FLOPs for a WELL v2 config")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment config .py file")
    parser.add_argument("--out", type=str, default=None, help="Optional path to write the JSON result")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run measurement on (cuda or cpu)")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides, e.g. dataset.batch_size=4 (use a small value to fit on one GPU)",
    )
    args, unknown = parser.parse_known_args()
    for arg in unknown:
        args.overrides.append(arg[2:] if arg.startswith("--") else arg)
    return args


def main() -> None:
    """Build the model, fetch a batch, run fwd+bwd under FlopCounterMode, print + write."""
    args = parse_args()

    pl.seed_everything(0)
    torch.set_float32_matmul_precision("high")

    config = load_config_from_file(args.config)
    verify_no_interpolator_overwrites(config, args.overrides)
    config = apply_config_overrides(config, args.overrides)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"[measure_flops] config={args.config}", flush=True)
    print(f"[measure_flops] device={device}  precision={config.train.precision}", flush=True)
    print(f"[measure_flops] batch_size={config.dataset.batch_size}", flush=True)

    # ------------------------------------------------------------------ #
    #  Build datamodule + network + wrapper (mirrors experiments/run.py) #
    # ------------------------------------------------------------------ #
    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    network = instantiate(config.net)

    # Skip torch.compile on purpose — FlopCounterMode is most reliable in eager mode.

    wrapper_kwargs: dict = {"network": network, "cfg": config}
    if hasattr(datamodule, "metadata"):
        wrapper_kwargs["metadata"] = datamodule.metadata
    if hasattr(datamodule, "normalization"):
        wrapper_kwargs["normalization"] = datamodule.normalization
    model = instantiate(config.lightning_wrapper_class, **wrapper_kwargs)

    model = model.to(device)
    model.train()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_class = type(network).__name__
    try:
        patch_size = int(config.net.in_proj_cfg.patch_size)
    except (AttributeError, TypeError, ValueError):
        patch_size = None
    print(
        f"[measure_flops] model={model_class}  patch_size={patch_size}  "
        f"params: total={num_params:,}  trainable={num_trainable_params:,}",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    #  Pull one batch                                                    #
    # ------------------------------------------------------------------ #
    print("[measure_flops] fetching one training batch …", flush=True)
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))

    def _to_device(obj):
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: _to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_device(v) for v in obj)
        return obj

    batch = _to_device(batch)

    # The wrapper's training_step calls ``self.log(...)``; suppress those
    # since we are not inside the actual trainer loop.
    model.log = lambda *a, **k: None  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    #  Measure                                                           #
    # ------------------------------------------------------------------ #
    # Match the precision the actual run would use.
    use_bf16 = "bf16" in config.train.precision
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    print("[measure_flops] running fwd+bwd under FlopCounterMode …", flush=True)
    with FlopCounterMode(display=False) as fc:
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_bf16):
            output = model.training_step(batch, 0)
        fwd_flops = fc.get_total_flops()

        loss = output["loss"] if isinstance(output, dict) else output
        loss.backward()
        total_flops = fc.get_total_flops()

    bwd_flops = total_flops - fwd_flops

    # ------------------------------------------------------------------ #
    #  Report                                                            #
    # ------------------------------------------------------------------ #
    bs = config.dataset.batch_size
    print()
    print(f"[measure_flops] per-rank batch_size = {bs}")
    print(f"[measure_flops] fwd  = {fwd_flops:>20,d}  ({fwd_flops / 1e12:.3f} TFLOPs)")
    print(f"[measure_flops] bwd  = {bwd_flops:>20,d}  ({bwd_flops / 1e12:.3f} TFLOPs)")
    print(f"[measure_flops] step = {total_flops:>20,d}  ({total_flops / 1e12:.3f} TFLOPs)")

    payload = {
        "config": args.config,
        "overrides": args.overrides,
        "model_class": model_class,
        "patch_size": patch_size,
        "batch_size": bs,
        "precision": config.train.precision,
        "device": str(device),
        "num_params": int(num_params),
        "num_trainable_params": int(num_trainable_params),
        "fwd_flops": int(fwd_flops),
        "bwd_flops": int(bwd_flops),
        "step_flops": int(total_flops),
        "fwd_tflops": fwd_flops / 1e12,
        "bwd_tflops": bwd_flops / 1e12,
        "step_tflops": total_flops / 1e12,
    }

    out_path = Path(args.out) if args.out else Path(args.config).with_suffix(".flops.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[measure_flops] wrote {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
