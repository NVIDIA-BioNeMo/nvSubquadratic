"""Validate a ViT-5 checkpoint downloaded from W&B and print metrics.

Downloads the "best" checkpoint for a given W&B run, loads it into the
matching model architecture, and runs a single validation + test pass
on ImageNet-1k.

Usage:
    # Attention baseline (default)
    source .env && PYTHONPATH=. python benchmarks/vit5_imagenet/validate_checkpoint.py

    # FiLM model
    source .env && PYTHONPATH=. python benchmarks/vit5_imagenet/validate_checkpoint.py \
        --model film --run-id peeaqdkq

    # GAP model
    source .env && PYTHONPATH=. python benchmarks/vit5_imagenet/validate_checkpoint.py \
        --model gap --run-id tcji9tfx

    # Any run with a custom config module
    source .env && PYTHONPATH=. python benchmarks/vit5_imagenet/validate_checkpoint.py \
        --config-module examples.vit5_imagenet.v3.vit5_small_pretrain_attention_ema \
        --run-id 44or24g1
"""

import argparse
import importlib
import re
import sys


sys.path.insert(0, ".")

import pytorch_lightning as pl
import torch

from experiments.utils.checkpointing import (
    StripCompiledPrefix,
    download_checkpoint,
    load_checkpoint_state_dict,
)
from nvsubquadratic.lazy_config import instantiate


WANDB_ENTITY = "implicit-long-convs"
WANDB_PROJECT = "nvsubquadratic"

# Pre-configured model presets: (config module path, default run ID, needs fftconv compat)
MODEL_PRESETS = {
    "attention": (
        "examples.vit5_imagenet.v2.vit5_small_pretrain_attention_ema",
        "44or24g1",
        False,
    ),
    "film": (
        "examples.vit5_imagenet.v3.vit5_small_pretrain_hyena_cls_row_gated_film_ema",
        "peeaqdkq",
        True,
    ),
    "gap": (
        "examples.vit5_imagenet.v2.vit5_small_pretrain_hyena_gap_apex_gated_ema",
        "tcji9tfx",
        True,
    ),
}

# Older SIRENKernelND used nn.Sequential("kernel_network") where even indices
# are Linear layers.  Current code uses nn.ModuleList("hidden_linears").
_SIREN_SEQUENTIAL_RE = re.compile(r"(\.kernel\.kernel_network)\.(\d+)\.(weight|bias)$")


def _remap_siren_sequential_to_modulelist(state_dict):
    """Rename ``kernel_network.{2*i}`` keys to ``hidden_linears.{i}``."""
    remapped = {}
    for key, val in state_dict.items():
        m = _SIREN_SEQUENTIAL_RE.search(key)
        if m:
            linear_idx = int(m.group(2)) // 2
            new_key = key[: m.start()] + f".kernel.hidden_linears.{linear_idx}.{m.group(3)}"
            remapped[new_key] = val
        else:
            remapped[key] = val
    return remapped


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate a ViT-5 checkpoint from W&B")
    parser.add_argument(
        "--model",
        choices=list(MODEL_PRESETS.keys()),
        default="attention",
        help="Model preset (default: attention)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="W&B run ID. Overrides the preset default.",
    )
    parser.add_argument(
        "--config-module",
        default=None,
        help="Python module path for get_config(). Overrides --model preset.",
    )
    parser.add_argument(
        "--alias",
        choices=["best", "latest"],
        default="best",
        help="Checkpoint alias to download (default: best)",
    )
    parser.add_argument(
        "--remap-siren-keys",
        action="store_true",
        help="Remap old kernel_network keys to hidden_linears (for pre-refactor checkpoints)",
    )
    return parser.parse_args()


def main():
    """Download a checkpoint from W&B and validate it on ImageNet."""
    args = parse_args()

    # Resolve config module and run ID
    if args.config_module:
        config_module_path = args.config_module
        run_id = args.run_id
        fftconv_compat = False
        if run_id is None:
            parser = argparse.ArgumentParser()
            parser.error("--run-id is required when using --config-module")
    else:
        config_module_path, default_run_id, fftconv_compat = MODEL_PRESETS[args.model]
        run_id = args.run_id or default_run_id

    run_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}"

    print("=" * 60)
    print("ViT-5 Checkpoint Validation")
    print("=" * 60)
    print(f"  Config:     {config_module_path}")
    print(f"  W&B run:    {run_path}")
    print(f"  Alias:      {args.alias}")
    print()

    # Import config
    module = importlib.import_module(config_module_path)
    config = module.get_config()
    config.train.do = False
    config.debug = True
    config.compile = False

    if fftconv_compat:
        import nvsubquadratic.ops.fftconv as _fftconv

        _fftconv.COMPILE_COMPATIBLE = True

    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    # Instantiate model and data
    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    network = instantiate(config.net)
    model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)

    # Download and load checkpoint
    print(f"Downloading checkpoint (alias={args.alias})...")
    ckpt_path = download_checkpoint(run_path=run_path, alias=args.alias)
    print(f"  Downloaded to: {ckpt_path}")

    state_dict = load_checkpoint_state_dict(ckpt_path)

    if args.remap_siren_keys or args.model == "gap":
        state_dict = _remap_siren_sequential_to_modulelist(state_dict)

    strip = StripCompiledPrefix()
    state_dict = strip(state_dict=state_dict, model=model)

    # Key compatibility check
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    print(f"  Checkpoint keys: {len(ckpt_keys)}")
    print(f"  Model keys:      {len(model_keys)}")
    if missing:
        print(f"  Missing ({len(missing)}): {sorted(missing)[:5]}...")
    if unexpected:
        print(f"  Unexpected ({len(unexpected)}): {sorted(unexpected)[:5]}...")
    if not missing and not unexpected:
        print("  All keys match exactly.")
    print()

    model.load_state_dict(state_dict, strict=True)

    # Validate
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    print("Running validation...")
    val_results = trainer.validate(model, datamodule=datamodule)

    print("Running test...")
    test_results = trainer.test(model, datamodule=datamodule)

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    if val_results:
        for key, val in sorted(val_results[0].items()):
            print(f"  {key}: {val:.4f}")
    if test_results:
        for key, val in sorted(test_results[0].items()):
            print(f"  {key}: {val:.4f}")


if __name__ == "__main__":
    main()
