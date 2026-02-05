#!/usr/bin/env python3
"""Benchmark ShortCausalConv1dKernel: speed (CUDA vs PyTorch) and accuracy by precision.

By default runs three kernel sizes (short=7, medium=64, long=128), all three
precisions (fp32, fp16, bf16), and:
  - Plots speed: CUDA vs PyTorch time (ms) for each kernel size and precision
  - Prints accuracy: max absolute difference vs reference for all precisions

Example usage:
    # Default: 3 kernel sizes, 3 precisions, plot + accuracy table
    python scripts/bench_short_causalconv1d_kernel.py

    # Save plot to file
    python scripts/bench_short_causalconv1d_kernel.py --output benchmark.png

    # Custom kernel sizes or seq length
    python scripts/bench_short_causalconv1d_kernel.py --kernel-sizes 3 32 256 --seq-len 2048
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from nvsubquadratic.ops.short_causalconv1d_kernel import (
    ShortCausalConv1dKernel,
    is_cuda_kernel_available,
)


# Default: short, medium, long kernel sizes
DEFAULT_KERNEL_SIZES = [7, 64, 128]
DEFAULT_KERNEL_LABELS = ["Short (7)", "Medium (64)", "Long (128)"]

# Default sequence length (medium)
DEFAULT_SEQ_LEN = 4096

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def pytorch_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch reference implementation with causal padding."""
    kernel_size = weight.shape[1]
    weight_conv = weight.unsqueeze(1)
    left_pad = kernel_size - 1
    x_padded = F.pad(x, (left_pad, 0))
    return F.conv1d(x_padded, weight_conv, bias=bias, groups=weight.shape[0])


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one_benchmark(
    kernel_size: int,
    seq_len: int,
    dtype: torch.dtype,
    dtype_name: str,
    args: Any,
) -> dict[str, Any]:
    """Run benchmark for one (kernel_size, seq_len, dtype). Returns cuda_ms, pytorch_ms, max_diff."""
    device = torch.device("cuda")
    torch.manual_seed(42)

    conv = ShortCausalConv1dKernel(
        channels=args.channels,
        kernel_size=kernel_size,
        bias=not args.no_bias,
        activation="identity",
    ).to(device=device, dtype=dtype)

    x = torch.randn(
        args.batch,
        args.channels,
        seq_len,
        device=device,
        dtype=dtype,
    )

    # Correctness
    with torch.no_grad():
        y_cuda = conv(x)
        y_pytorch = pytorch_causal_conv1d(x, conv.weight, conv.bias)
    max_diff = (y_cuda - y_pytorch).abs().max().item()

    # Timing
    for _ in range(args.warmup):
        conv(x)
        pytorch_causal_conv1d(x, conv.weight, conv.bias)
    _maybe_sync(device)

    _maybe_sync(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        conv(x)
    _maybe_sync(device)
    cuda_ms = (time.perf_counter() - start) / args.steps * 1000

    _maybe_sync(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        pytorch_causal_conv1d(x, conv.weight, conv.bias)
    _maybe_sync(device)
    pytorch_ms = (time.perf_counter() - start) / args.steps * 1000

    return {
        "kernel_size": kernel_size,
        "seq_len": seq_len,
        "dtype": dtype_name,
        "cuda_ms": cuda_ms,
        "pytorch_ms": pytorch_ms,
        "speedup": pytorch_ms / cuda_ms if cuda_ms > 0 else 0.0,
        "max_diff": max_diff,
    }


def print_accuracy_table(
    results: list[dict[str, Any]],
    kernel_sizes: list[int],
    kernel_labels: list[str],
    dtypes: list[str],
) -> None:
    """Print accuracy (max_diff) for all precisions at the chosen kernel sizes."""
    print()
    print("=" * 70)
    print("Accuracy (max absolute diff vs PyTorch reference)")
    print("=" * 70)
    print(
        f"  Sequence length: {results[0]['seq_len']}  (batch={results[0].get('batch', '?')}, channels={results[0].get('channels', '?')})"
    )
    print()
    # Header
    header = f"{'Precision':<10}"
    for lab in kernel_labels:
        header += f" {lab:>14}"
    print(header)
    print("-" * (10 + 15 * len(kernel_labels)))
    for dtype in dtypes:
        row = f"{dtype:<10}"
        for ks in kernel_sizes:
            r = next((x for x in results if x["dtype"] == dtype and x["kernel_size"] == ks), None)
            if r is not None:
                row += f" {r['max_diff']:>13.2e}"
            else:
                row += "            —"
        print(row)
    print()


def plot_speed(
    results: list[dict[str, Any]],
    kernel_sizes: list[int],
    kernel_labels: list[str],
    dtypes: list[str],
    output_path: str | None,
) -> None:
    """Plot CUDA vs PyTorch time (ms) for each kernel size and precision."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping plot. Install with: pip install matplotlib", file=sys.stderr)
        return

    n_k = len(kernel_sizes)
    n_d = len(dtypes)
    x = np.arange(n_k)
    width = 0.8 / (2 * n_d + 1)  # bar width so CUDA/PyTorch pairs don't overlap

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_dtype = {"fp32": "#1f77b4", "fp16": "#ff7f0e", "bf16": "#2ca02c"}
    for i, dtype in enumerate(dtypes):
        cuda_vals = []
        pytorch_vals = []
        for ks in kernel_sizes:
            r = next((x for x in results if x["dtype"] == dtype and x["kernel_size"] == ks), None)
            if r is not None:
                cuda_vals.append(r["cuda_ms"])
                pytorch_vals.append(r["pytorch_ms"])
            else:
                cuda_vals.append(0.0)
                pytorch_vals.append(0.0)
        off = (2 * i - n_d + 1) * width
        ax.bar(x + off, cuda_vals, width, label=f"{dtype} CUDA", color=colors_dtype.get(dtype, "gray"), alpha=0.9)
        ax.bar(
            x + off + width,
            pytorch_vals,
            width,
            label=f"{dtype} PyTorch",
            color=colors_dtype.get(dtype, "gray"),
            alpha=0.5,
            hatch="//",
        )

    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Kernel size")
    ax.set_title("ShortCausalConv1dKernel: CUDA vs PyTorch (lower is faster)")
    ax.set_xticks(x)
    ax.set_xticklabels(kernel_labels)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        fig.savefig("bench_short_causalconv1d_kernel.png", dpi=150)
        print("Plot saved to bench_short_causalconv1d_kernel.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ShortCausalConv1dKernel: default = 3 kernel sizes (short/medium/long), plot speed, print accuracy"
    )
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--channels", type=int, default=64, help="Channels")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help=f"Sequence length (default: {DEFAULT_SEQ_LEN})",
    )
    parser.add_argument(
        "--kernel-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_KERNEL_SIZES,
        help=f"Kernel sizes to test (default: {DEFAULT_KERNEL_SIZES})",
    )
    parser.add_argument(
        "--kernel-labels",
        type=str,
        nargs="+",
        default=None,
        help="Labels for kernel sizes (default: Short (7), Medium (64), Long (128))",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("fp32", "fp16", "bf16"),
        default=("fp32", "fp16", "bf16"),
        help="Precisions to test (default: fp32 fp16 bf16)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--steps", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--no-bias", action="store_true", help="Disable bias")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="FILE",
        help="Output PNG path (default: bench_short_causalconv1d_kernel.png)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate plot",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    if not is_cuda_kernel_available():
        raise SystemExit("CUDA kernel not available. Install subquadratic-ops-torch-cu12.")

    kernel_sizes = args.kernel_sizes
    if args.kernel_labels is not None:
        kernel_labels = args.kernel_labels
    else:
        kernel_labels = [f"K={k}" for k in kernel_sizes]
        if kernel_sizes == DEFAULT_KERNEL_SIZES:
            kernel_labels = DEFAULT_KERNEL_LABELS

    print("=" * 70)
    print("Benchmark: ShortCausalConv1dKernel (speed + accuracy)")
    print("=" * 70)
    print(f"  Kernel sizes: {kernel_sizes}  ({', '.join(kernel_labels)})")
    print(f"  Seq length:   {args.seq_len}")
    print(f"  Precisions:   {list(args.dtypes)}")
    print(f"  Batch:        {args.batch}  Channels: {args.channels}")
    print()

    results = []
    for dtype_name in args.dtypes:
        dtype = DTYPE_MAP[dtype_name]
        for kernel_size in kernel_sizes:
            r = run_one_benchmark(
                kernel_size=kernel_size,
                seq_len=args.seq_len,
                dtype=dtype,
                dtype_name=dtype_name,
                args=args,
            )
            r["batch"] = args.batch
            r["channels"] = args.channels
            results.append(r)

    # Print accuracy for all three precisions at these settings
    print_accuracy_table(results, kernel_sizes, kernel_labels, list(args.dtypes))

    # Speed summary
    print("=" * 70)
    print("Speed summary (CUDA vs PyTorch)")
    print("=" * 70)
    print(f"{'Precision':<10} {'Kernel':<12} {'CUDA (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['dtype']:<10} {r['kernel_size']:<12} {r['cuda_ms']:<12.3f} {r['pytorch_ms']:<14.3f} {r['speedup']:.2f}x"
        )
    print()

    # Plot speed by default
    if not args.no_plot:
        plot_speed(
            results,
            kernel_sizes,
            kernel_labels,
            list(args.dtypes),
            args.output,
        )


if __name__ == "__main__":
    main()
