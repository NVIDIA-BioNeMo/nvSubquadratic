#!/usr/bin/env python3

"""Evaluate a JiT diffusion checkpoint with the repository's built-in FID routine."""

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from omegaconf import OmegaConf


# Ensure repository root is importable when running `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nvsubquadratic.lazy_config import instantiate


def _load_config(config_path: str, config_fn: str) -> Any:
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    spec = importlib.util.spec_from_file_location("jit_fid_eval_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import config module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, config_fn, None)
    if fn is None or not callable(fn):
        raise AttributeError(f"Config module {path} has no callable '{config_fn}'")

    return fn()


def _instantiate_network(config: Any) -> torch.nn.Module:
    try:
        return instantiate(config.net)
    except Exception as exc:
        # Some configs (e.g. ccnn_jit_baseline) use self-references like ${net.in_channels}
        # inside cfg.net. Resolve those by providing cfg.net as a top-level OmegaConf node.
        message = str(exc)
        if "Interpolation key 'net." not in message:
            raise
        net_root = OmegaConf.create({"net": config.net})
        OmegaConf.resolve(net_root)
        return instantiate(net_root.net)


def _infer_example_input_shape(cfg: Any, explicit_image_size: int | None, channels: int) -> torch.Size:
    if explicit_image_size is not None:
        size = int(explicit_image_size)
        return torch.Size((size, size, channels))

    dataset_cfg = getattr(cfg, "dataset", None)
    if dataset_cfg is not None:
        final_image_size = getattr(dataset_cfg, "final_image_size", None)
        if final_image_size is not None:
            size = int(final_image_size)
            return torch.Size((size, size, channels))

    return torch.Size((64, 64, channels))


def _checkpoint_has_ema_weights(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("_ema_model.") for key in state_dict)


def _latest_wandb_run_file(run_dir: Path, rel_path: str) -> Path | None:
    candidates = list(run_dir.glob(f"wandb/run-*/files/{rel_path}"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _checkpoint_training_commit(run_dir: Path) -> tuple[str | None, Path | None]:
    metadata_path = _latest_wandb_run_file(run_dir, "wandb-metadata.json")
    if metadata_path is None:
        return None, None
    try:
        payload = json.loads(metadata_path.read_text())
    except Exception:
        return None, metadata_path
    commit = payload.get("git", {}).get("commit")
    return (str(commit) if commit else None), metadata_path


def _head_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _print_logged_reference_metrics(run_dir: Path) -> None:
    summary_path = _latest_wandb_run_file(run_dir, "wandb-summary.json")
    if summary_path is None:
        return
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        return

    print(f"[ref] summary_file: {summary_path}")
    for key in (
        "global_step",
        "train/loss_step",
        "train/loss_epoch",
        "val/loss",
        "metrics/fid_online",
        "metrics/is_online",
    ):
        value = summary.get(key)
        if value is not None:
            print(f"[ref] {key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FID for a JiT baseline diffusion checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/imagenet_diffusion/jit_baseline.py",
        help="Path to a config python file with a get_config() function.",
    )
    parser.add_argument(
        "--config-fn",
        type=str,
        default="get_config",
        help="Name of the config factory function in --config.",
    )
    parser.add_argument(
        "--fid-stats",
        type=str,
        default=None,
        help="Override path to FID reference stats (.npz). Defaults to value from config.",
    )
    parser.add_argument(
        "--fid-num-samples",
        type=int,
        default=None,
        help="Override number of generated samples used for FID.",
    )
    parser.add_argument(
        "--fid-batch-size",
        type=int,
        default=None,
        help="Override generation batch size for FID sampling.",
    )
    parser.add_argument(
        "--fid-num-inference-steps",
        type=int,
        default=None,
        help="Override sampling steps used during FID generation.",
    )
    parser.add_argument(
        "--ema-mode",
        choices=["auto", "always", "never"],
        default="auto",
        help="EMA usage for sampling: auto enables EMA if EMA weights are present in checkpoint.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional generated image size (H=W) used to seed wrapper.example_input_shape.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Number of image channels in generated samples.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Directory used for temporary generated FID images. Defaults to checkpoint run directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string (default: cuda).",
    )
    parser.add_argument(
        "--non-strict-load",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the denoiser network with torch.compile before running FID sampling.",
    )
    parser.add_argument(
        "--compile-backend",
        type=str,
        default=None,
        help="Optional torch.compile backend (e.g. inductor, aot_eager).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default=None,
        help="Optional torch.compile mode (e.g. default, reduce-overhead, max-autotune).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    run_dir = checkpoint_path.parent.parent

    trained_commit, metadata_path = _checkpoint_training_commit(run_dir)
    head_commit = _head_commit()
    if trained_commit is not None:
        print(f"[ckpt] recorded training commit: {trained_commit}")
        if metadata_path is not None:
            print(f"[ckpt] metadata source: {metadata_path}")
    if head_commit is not None:
        print(f"[env] current HEAD commit: {head_commit}")
    if trained_commit is not None and head_commit is not None and trained_commit != head_commit:
        print(
            "[warn] Checkpoint was trained with a different commit than current checkout. "
            "Loss/FID may differ if diffusion or datamodule code changed."
        )

    _print_logged_reference_metrics(run_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type != "cuda":
        raise RuntimeError(
            "This script requires CUDA because DiffusionWrapper._run_jit_online_eval uses CUDA-only "
            "feature extraction in torch-fidelity."
        )

    cfg = _load_config(args.config, args.config_fn)
    network = _instantiate_network(cfg)
    wrapper = instantiate(cfg.lightning_wrapper_class, network=network, cfg=cfg)

    ckpt_obj = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" not in ckpt_obj:
        raise ValueError(f"Checkpoint does not contain a 'state_dict': {checkpoint_path}")

    state_dict = ckpt_obj["state_dict"]
    strict = not args.non_strict_load
    load_result = wrapper.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(
            f"[load] strict=False missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}"
        )

    wrapper.example_input_shape = _infer_example_input_shape(cfg, args.image_size, args.channels)

    if args.fid_stats is not None:
        wrapper.fid_stats_file = args.fid_stats
    if wrapper.fid_stats_file is None:
        raise ValueError("No FID stats file configured. Set --fid-stats or configure diffusion.fid_stats_file.")
    fid_stats_path = Path(wrapper.fid_stats_file).expanduser().resolve()
    if not fid_stats_path.exists():
        raise FileNotFoundError(f"FID stats file not found: {fid_stats_path}")
    wrapper.fid_stats_file = str(fid_stats_path)

    if args.fid_num_samples is not None:
        wrapper.fid_num_samples = int(args.fid_num_samples)
    if args.fid_batch_size is not None:
        wrapper.fid_batch_size = int(args.fid_batch_size)
    if args.fid_num_inference_steps is not None:
        wrapper.fid_num_inference_steps = int(args.fid_num_inference_steps)

    checkpoint_has_ema = _checkpoint_has_ema_weights(state_dict)
    if wrapper.ema_enabled and wrapper._ema_model is not None:
        if args.ema_mode == "always":
            wrapper._ema_has_been_updated = True
        elif args.ema_mode == "never":
            wrapper._ema_has_been_updated = False
        else:
            wrapper._ema_has_been_updated = checkpoint_has_ema

    if args.output_root is not None:
        output_root = Path(args.output_root).expanduser().resolve()
    else:
        # Typical checkpoint layout: runs/<run_name>/checkpoints/<epoch...>.ckpt
        output_root = checkpoint_path.parent.parent
    output_root.mkdir(parents=True, exist_ok=True)

    wrapper = wrapper.to(device)
    if args.torch_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build.")
        compile_kwargs: dict[str, str] = {}
        if args.compile_backend:
            compile_kwargs["backend"] = args.compile_backend
        if args.compile_mode:
            compile_kwargs["mode"] = args.compile_mode
        wrapper.network = torch.compile(wrapper.network, **compile_kwargs)
    wrapper.log = lambda *unused_args, **unused_kwargs: None
    wrapper._trainer = SimpleNamespace(
        default_root_dir=str(output_root),
        world_size=1,
        global_rank=0,
        global_step=int(ckpt_obj.get("global_step", 0)),
        logger=None,
    )

    print(f"[fid] checkpoint: {checkpoint_path}")
    print(f"[fid] stats: {wrapper.fid_stats_file}")
    print(f"[fid] num_samples: {wrapper.fid_num_samples}")
    print(f"[fid] batch_size: {wrapper.fid_batch_size}")
    print(f"[fid] inference_steps: {wrapper.fid_num_inference_steps}")
    print(f"[fid] example_input_shape: {tuple(wrapper.example_input_shape)}")
    print(f"[fid] ema_enabled: {wrapper.ema_enabled}, use_ema_for_sampling: {wrapper._ema_has_been_updated}")
    print(f"[fid] torch_compile: {args.torch_compile}")
    print(f"[fid] output_root: {output_root}")

    wrapper._run_jit_online_eval()


if __name__ == "__main__":
    main()
