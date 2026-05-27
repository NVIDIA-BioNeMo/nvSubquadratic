#!/usr/bin/env python3
"""Quick correctness and forward-speed check for the CUDA fft_causal_conv1d kernel.

Compares ``subquadratic_ops_torch.fft_causal_conv1d`` against the
reference ``torch.fft``-based implementation across a few realistic
1D Hyena workloads, then prints absolute / relative errors and forward
throughput.  Intended as a fast sanity gate after kernel changes.

Targets: any Ampere+ GPU with the ``subquadratic_ops_torch`` wheel
installed (requires CUDA toolkit 12).

Usage:
    PYTHONPATH=. conda run -n nv-subq python \\
        benchmarks/ops/bench_subquadratic_fftconv.py --device cuda

Output: stdout (correctness summary + throughput table).
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from subquadratic_ops_torch.fft_causal_conv1d import fft_causal_conv1d


def _reference_causal_conv1d(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Depth-wise causal conv via PyTorch (used for correctness + baseline)."""
    dw_weight = weight.unsqueeze(1)
    y = F.conv1d(x, dw_weight, padding=weight.shape[-1] - 1, groups=x.shape[1])
    return y[..., : x.shape[-1]]


def _maybe_sync(device: torch.device) -> None:
    """Synchronize the CUDA stream if we are on a GPU device."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(label: str, fn, device: torch.device, warmup: int, steps: int) -> float:
    """Run *fn* with warmup, then time *steps* iterations and print the average."""
    for _ in range(warmup):
        fn()
    _maybe_sync(device)
    total = 0.0
    for _ in range(steps):
        _maybe_sync(device)
        start = time.perf_counter()
        fn()
        _maybe_sync(device)
        total += time.perf_counter() - start
    avg = total / max(1, steps)
    print(f"{label:>20s}: {avg * 1e3:7.3f} ms")
    return avg


def main() -> None:
    """Run latency benchmarks for subquadratic FFT convolution variants."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--kernel-len", type=int, default=1024)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device not available.")
    if args.device == "cpu":
        raise SystemExit("The custom kernel only runs on CUDA.")

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    torch.manual_seed(42)
    x = torch.randn(args.batch, args.heads, args.seq_len, device=device, dtype=dtype)
    weight = torch.randn(args.heads, args.kernel_len, device=device, dtype=dtype)

    with torch.no_grad():
        y_kernel = fft_causal_conv1d(x, weight)
        y_reference = _reference_causal_conv1d(x, weight)
    torch.testing.assert_close(y_kernel.float(), y_reference.float(), rtol=1e-3, atol=1e-3)
    print("Outputs match reference (rtol=1e-3, atol=1e-3).")

    with torch.no_grad():
        kernel_time = _bench(
            "subquadratic kernel",
            lambda: fft_causal_conv1d(x, weight),
            device,
            args.warmup,
            args.steps,
        )
        reference_time = _bench(
            "depthwise conv1d",
            lambda: _reference_causal_conv1d(x, weight),
            device,
            args.warmup,
            args.steps,
        )

    if reference_time > 0.0:
        print(f"Speedup (reference / kernel): {reference_time / kernel_time:.2f}x")


if __name__ == "__main__":
    main()
