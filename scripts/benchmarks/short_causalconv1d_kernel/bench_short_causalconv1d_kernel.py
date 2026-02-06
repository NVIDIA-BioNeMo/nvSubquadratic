#!/usr/bin/env python3
"""Benchmark ShortCausalConv1dKernel: speed (CUDA vs PyTorch) and accuracy by precision.

By default runs three kernel sizes (short=7, medium=64, long=128), three
sequence lengths (128, 4096, 16k), all three precisions (fp32, fp16, bf16), and:
  - Plots speed: one subplot per seq length , CUDA vs PyTorch
  - Prints accuracy: max absolute diff vs reference for all precisions at each setting

Example usage:
    # Default: 3 kernel sizes, 3 seq lengths, 3 precisions, 3-panel plot + accuracy
    python scripts/bench_short_causalconv1d_kernel.py

    # Save plot to file
    python scripts/bench_short_causalconv1d_kernel.py --output benchmark.png

    # Custom seq lengths
    python scripts/bench_short_causalconv1d_kernel.py --seq-lens 256 2048 8192
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nvsubquadratic.ops.short_causalconv1d_kernel import (
    ShortCausalConv1dKernel,
    is_cuda_kernel_available,
)


# Directory where this script lives (plot is saved here by default)
SCRIPT_DIR = Path(__file__).resolve().parent

# Default: short, medium, long kernel sizes
DEFAULT_KERNEL_SIZES = [7, 64, 128]
DEFAULT_KERNEL_LABELS = ["Short (7)", "Medium (64)", "Long (128)"]

# Default sequence lengths: short, medium, long (16k = 16384)
DEFAULT_SEQ_LENS = [128, 4096, 16384]

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
    seq_lens: list[int],
    dtypes: list[str],
) -> None:
    """Print accuracy (max_diff) for all precisions at each (kernel size, seq_len)."""
    print()
    print("=" * 70)
    print("Accuracy (max absolute diff vs PyTorch reference)")
    print("=" * 70)
    batch = results[0].get("batch", "?") if results else "?"
    channels = results[0].get("channels", "?") if results else "?"
    print(f"  Batch: {batch}  Channels: {channels}")
    print()
    for seq_len in seq_lens:
        subset = [r for r in results if r["seq_len"] == seq_len]
        if not subset:
            continue
        print(f"  Seq length {seq_len}:")
        header = f"    {'Precision':<10}"
        for lab in kernel_labels:
            header += f" {lab:>14}"
        print(header)
        print("    " + "-" * (10 + 15 * len(kernel_labels)))
        for dtype in dtypes:
            row = f"    {dtype:<10}"
            for ks in kernel_sizes:
                r = next((x for x in subset if x["dtype"] == dtype and x["kernel_size"] == ks), None)
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
    seq_lens: list[int],
    dtypes: list[str],
    output_path: str | None,
    batch: int,
    channels: int,
) -> None:
    """Single plot: x = seq length, bar groups per kernel size (CUDA + PyTorch). fp32 only. Saves to script dir."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed; skipping plot. Install with: pip install matplotlib", file=sys.stderr)
        return

    results = [r for r in results if r["dtype"] == "fp32"]
    if not results:
        return

    n_k = len(kernel_sizes)
    n_s = len(seq_lens)
    x_pos = np.arange(n_s)
    colors_kernel = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    if n_k > 3:
        try:
            cmap = matplotlib.colormaps["tab10"]
        except AttributeError:
            cmap = matplotlib.cm.get_cmap("tab10")
        colors_kernel = [cmap(i % 10) for i in range(n_k)]

    # At each seq length: 3 kernel sizes × 2 (CUDA, PyTorch) = 6 bars
    n_bars_per_x = n_k * 2
    total_width = 0.8
    bar_width = total_width / n_bars_per_x

    fig, ax = plt.subplots(figsize=(8, 5))

    for s_idx, seq_len in enumerate(seq_lens):
        subset = [r for r in results if r["seq_len"] == seq_len]
        for k_idx, ks in enumerate(kernel_sizes):
            r = next((x for x in subset if x["kernel_size"] == ks), None)
            if r is None:
                continue
            # Bar positions: grouped by kernel size (CUDA then PyTorch side by side)
            off = (k_idx * 2 - (n_bars_per_x - 1) / 2) * bar_width
            lab_cuda = f"{kernel_labels[k_idx]} CUDA" if s_idx == 0 else ""
            lab_pytorch = f"{kernel_labels[k_idx]} PyTorch" if s_idx == 0 else ""
            # CUDA bar (solid)
            ax.bar(
                x_pos[s_idx] + off,
                r["cuda_ms"],
                bar_width,
                color=colors_kernel[k_idx % len(colors_kernel)],
                edgecolor="k",
                linewidth=0.6,
                label=lab_cuda,
            )
            # PyTorch bar (hatched)
            ax.bar(
                x_pos[s_idx] + off + bar_width,
                r["pytorch_ms"],
                bar_width,
                color=colors_kernel[k_idx % len(colors_kernel)],
                edgecolor="k",
                linewidth=0.6,
                hatch="//",
                alpha=0.85,
                label=lab_pytorch,
            )

    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Sequence length")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in seq_lens])
    ax.set_title(f"ShortCausalConv1dKernel (fp32, batch={batch}, channels={channels}): CUDA vs PyTorch by kernel size")
    ax.set_yscale("log")
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    # Always save to script directory
    if output_path:
        path = Path(output_path)
        if not path.is_absolute():
            path = SCRIPT_DIR / path
    else:
        path = SCRIPT_DIR / "bench_short_causalconv1d_kernel.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"Plot saved to {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ShortCausalConv1dKernel: default = 3 kernel sizes (short/medium/long), plot speed, print accuracy"
    )
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--channels", type=int, default=128, help="Channels")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENS,
        help=f"Sequence lengths (default: {DEFAULT_SEQ_LENS})",
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
        help="Output PNG path (default: script_dir/bench_short_causalconv1d_kernel.png)",
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

    seq_lens = args.seq_lens

    print("=" * 70)
    print("Benchmark: ShortCausalConv1dKernel (speed + accuracy)")
    print("=" * 70)
    print(f"  Kernel sizes: {kernel_sizes}  ({', '.join(kernel_labels)})")
    print(f"  Seq lengths:  {seq_lens}")
    print(f"  Precisions:   {list(args.dtypes)}")
    print(f"  Batch:        {args.batch}  Channels: {args.channels}")
    print()

    results = []
    for dtype_name in args.dtypes:
        dtype = DTYPE_MAP[dtype_name]
        for kernel_size in kernel_sizes:
            for seq_len in seq_lens:
                r = run_one_benchmark(
                    kernel_size=kernel_size,
                    seq_len=seq_len,
                    dtype=dtype,
                    dtype_name=dtype_name,
                    args=args,
                )
                r["batch"] = args.batch
                r["channels"] = args.channels
                results.append(r)

    # Print accuracy for all precisions at each (kernel size, seq_len)
    print_accuracy_table(results, kernel_sizes, kernel_labels, seq_lens, list(args.dtypes))

    # Speed summary (include seq_len)
    print("=" * 70)
    print("Speed summary (CUDA vs PyTorch)")
    print("=" * 70)
    print(f"{'Precision':<10} {'SeqLen':<8} {'Kernel':<8} {'CUDA (ms)':<12} {'PyTorch (ms)':<14} {'Speedup'}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['dtype']:<10} {r['seq_len']:<8} {r['kernel_size']:<8} {r['cuda_ms']:<12.3f} {r['pytorch_ms']:<14.3f} {r['speedup']:.2f}x"
        )
    print()

    # Plot speed: one subplot per seq length
    if not args.no_plot:
        plot_speed(
            results,
            kernel_sizes,
            kernel_labels,
            seq_lens,
            list(args.dtypes),
            args.output,
            args.batch,
            args.channels,
        )


if __name__ == "__main__":
    main()
