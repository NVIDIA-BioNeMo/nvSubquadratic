#!/usr/bin/env python
"""Benchmark GPU memory/time for the ImageNet diffusion model (batch size = 1)."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, replace
from typing import Callable, Iterable, List

import torch

from experiments.utils.cli import apply_config_overrides, load_config_from_file
from nvsubquadratic.lazy_config import instantiate


RESOLUTIONS = [64, 128, 256, 1024]
BATCH_SIZE = 1


@dataclass(frozen=True)
class ModelSpec:
    """Simple descriptor for each benchmarked model size."""

    name: str
    hidden_dim: int
    num_layers: int
    num_params: int | None = None


MODEL_SPECS: tuple[ModelSpec, ModelSpec, ModelSpec] = (
    ModelSpec(name="tiny", hidden_dim=512, num_layers=8),
    ModelSpec(name="base", hidden_dim=768, num_layers=12),
    ModelSpec(name="large", hidden_dim=1024, num_layers=16),
)


def _ensure_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for this benchmark.")
    return torch.device("cuda")


def _clone_config(cfg):
    return copy.deepcopy(cfg)


def _prepare_config(base_cfg, spec: ModelSpec, image_size: int):
    cfg = _clone_config(base_cfg)
    overrides = [
        f"net.hidden_dim={spec.hidden_dim}",
        f"net.num_blocks={spec.num_layers}",
        f"diffusion.time_embed_dim={spec.hidden_dim}",
        f"diffusion.cosine_schedule_image_resolution={image_size}",
        f"diffusion.cosine_schedule_noise_res_low={max(32, image_size // 2)}",
        f"diffusion.cosine_schedule_noise_res_high={image_size}",
    ]
    overrides.append(f"dataset.image_size={image_size}")
    overrides.append(f"dataset.final_image_size={image_size}")
    return apply_config_overrides(cfg, overrides)


def _instantiate_wrapper(cfg, device: torch.device, dtype: torch.dtype):
    network = instantiate(cfg.net, in_channels=3, out_channels=3)
    wrapper = instantiate(cfg.lightning_wrapper_class, network=network, cfg=cfg)
    wrapper = wrapper.to(device=device, dtype=dtype)
    return wrapper


def _make_images(resolution: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.rand((BATCH_SIZE, resolution, resolution, 3), device=device, dtype=dtype) * 2.0 - 1.0


def _make_labels(device: torch.device) -> torch.Tensor:
    return torch.zeros((BATCH_SIZE,), device=device, dtype=torch.long)


def _measure(fn, device: torch.device, repeat: int) -> tuple[float, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) / repeat
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return elapsed, peak_mb


def _inference_fn(wrapper, resolution: int, dtype: torch.dtype):
    device = next(wrapper.parameters()).device
    images = _make_images(resolution, device, dtype)
    labels = _make_labels(device)
    timesteps = torch.randint(
        0,
        wrapper.scheduler.config.num_train_timesteps,
        (BATCH_SIZE,),
        device=device,
        dtype=torch.long,
    )

    def _run():
        wrapper.eval()
        with torch.no_grad():
            condition = wrapper._condition_from_timesteps(timesteps, labels=labels)
            wrapper.network({"input": images, "condition": condition})

    return _run


def _training_fn(wrapper, resolution: int, dtype: torch.dtype):
    device = next(wrapper.parameters()).device
    wrapper.train()
    images = _make_images(resolution, device, dtype)
    labels = _make_labels(device)
    batch = {"input": images, "label": labels, "condition": None}
    optimizer = torch.optim.AdamW(wrapper.parameters(), lr=1e-4)

    def _run():
        optimizer.zero_grad(set_to_none=True)
        loss = wrapper._shared_step(batch)
        loss.backward()
        optimizer.step()

    return _run


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message


def _run_mode(
    *,
    mode: str,
    build_fn: Callable[[], Callable[[], None]],
    spec: ModelSpec,
    image_size: int,
    device: torch.device,
    repeat: int,
    dtype_name: str,
) -> dict:
    result = {
        "mode": mode,
        "model": spec.name,
        "hidden_dim": spec.hidden_dim,
        "num_layers": spec.num_layers,
        "image_size": image_size,
        "batch_size": BATCH_SIZE,
        "dtype": dtype_name,
        "num_params": spec.num_params,
    }
    try:
        fn = build_fn()
        fn()  # warmup
        elapsed, peak = _measure(fn, device, repeat)
        result["time_ms"] = elapsed * 1e3
        result["peak_memory_mb"] = peak
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if _is_oom_error(exc):
            torch.cuda.empty_cache()
            result["error"] = "OOM"
        else:
            raise
    return result


def benchmark_spec(
    base_cfg,
    spec: ModelSpec,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    repeat: int,
    dtype_name: str,
) -> list[dict]:
    cfg = _prepare_config(base_cfg, spec, image_size)
    wrapper = _instantiate_wrapper(cfg, device, dtype)
    if spec.num_params is None:
        param_count = sum(p.numel() for p in wrapper.parameters())
        spec = replace(spec, num_params=param_count)
    results = []

    results.append(
        _run_mode(
            mode="inference",
            build_fn=lambda: _inference_fn(wrapper, image_size, dtype),
            spec=spec,
            image_size=image_size,
            device=device,
            repeat=repeat,
            dtype_name=dtype_name,
        )
    )

    results.append(
        _run_mode(
            mode="training",
            build_fn=lambda: _training_fn(wrapper, image_size, dtype),
            spec=spec,
            image_size=image_size,
            device=device,
            repeat=repeat,
            dtype_name=dtype_name,
        )
    )
    del wrapper
    torch.cuda.empty_cache()
    return results


def _print_table(rows: Iterable[dict]) -> None:
    header = "{:<10} {:<8} {:>5} {:>5} {:<6} {:>10} {:>10} {:>6} {:>6} {:>10} {:<8}"
    print(header.format("mode", "model", "res", "bs", "dtype", "time_ms", "mem_mb", "hidden", "layers", "params", "status"))
    for row in rows:
        time_val = row.get("time_ms")
        mem_val = row.get("peak_memory_mb")
        time_str = f"{time_val:10.2f}" if isinstance(time_val, (int, float)) else f"{'--':>10}"
        mem_str = f"{mem_val:10.1f}" if isinstance(mem_val, (int, float)) else f"{'--':>10}"
        status = row.get("error", "ok")
        params = row.get("num_params")
        params_str = f"{params/1e6:10.2f}M" if isinstance(params, (int, float)) else f"{'--':>10}"
        print(
            f"{row['mode']:<10} {row['model']:<8} {row['image_size']:>5} {row['batch_size']:>5} {row['dtype']:<6} "
            f"{time_str} {mem_str} {row['hidden_dim']:>6} {row['num_layers']:>6} {params_str} {status:<8}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ImageNet diffusion model memory/time.")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/imagenet_diffusion/ccnn_12_768_hyena_rope_qknorm.py",
        help="Path to the ImageNet diffusion config file.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of iterations to average for each measurement.",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["fp32", "bf16", "fp16"],
        help="List of precisions to benchmark (choices: fp32 bf16 fp16).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save the raw metrics as JSON.",
    )
    return parser.parse_args()


def _dtype_from_string(name: str) -> torch.dtype:
    mapping = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    return mapping[name.lower()]


def main() -> None:
    args = parse_args()
    base_cfg = load_config_from_file(args.config)
    device = _ensure_cuda()

    results: List[dict] = []
    repeat = max(1, args.repeat)

    for dtype_name in args.dtypes:
        dtype = _dtype_from_string(dtype_name)
        for spec in MODEL_SPECS:
            for res in RESOLUTIONS:
                spec_results = benchmark_spec(
                    base_cfg=base_cfg,
                    spec=spec,
                    image_size=res,
                    device=device,
                    dtype=dtype,
                    repeat=repeat,
                    dtype_name=dtype_name,
                )
                results.extend(spec_results)

    _print_table(results)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
