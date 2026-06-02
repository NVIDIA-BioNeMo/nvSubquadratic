# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Benchmark: torch.fft fftconv2d vs subquadratic_ops_torch CUDA kernel.

Compares latency (forward and forward+backward) of the torch.fft-based
``fftconv2d_fp32_bhl`` against the CUDA-accelerated
``subquadratic_ops_torch.fft_conv2d`` across realistic ViT/Hyena workloads.

Variants tested:
  - torch.fft eager
  - torch.fft + torch.compile (default)
  - torch.fft + torch.compile (max-autotune)
  - subquadratic_ops_torch CUDA kernel (eager)

Usage (requires GPU — run inside SLURM):
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python benchmarks/ops/bench_fftconv2d.py

    # With custom config:
    srun --gres=gpu:1 -c 16 --partition low \\
        conda run -n nv-subq python benchmarks/ops/bench_fftconv2d.py --iters 200 --warmup 50
"""

import argparse
import sys
import time

import torch


sys.path.insert(0, ".")

from nvsubquadratic.ops.fftconv import fftconv2d_fp32_bhl


def _has_subq_ops() -> bool:
    try:
        from subquadratic_ops_torch.fft_conv2d import fft_conv2d  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _time_fn(fn, warmup: int, iters: int) -> float:
    """Time a function, returning median ms per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def _time_fwd_bwd(fn, x, kernel, warmup: int, iters: int) -> float:
    """Time forward + backward, returning median ms per call."""
    for _ in range(warmup):
        x.grad = None
        kernel.grad = None
        y = fn()
        y.sum().backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        x.grad = None
        kernel.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = fn()
        y.sum().backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Workloads: realistic ViT/Hyena configurations
# ---------------------------------------------------------------------------

WORKLOADS = {
    "ViT-S 14x14 k=14": (4, 384, 14, 14, 14, 14),
    "ViT-S 14x14 k=7": (4, 384, 14, 14, 7, 7),
    "Hyena 32x32 k=7": (4, 128, 32, 32, 7, 7),
    "Hyena 32x32 k=13": (4, 256, 32, 32, 13, 13),
    "Hyena 64x64 k=7": (8, 128, 64, 64, 7, 7),
    "FiLM 32x32 k=7": (4, 128, 32, 32, 7, 7),
    "FiLM 14x14 k=14": (4, 384, 14, 14, 14, 14),
    "Large H 14x14 k=7": (4, 768, 14, 14, 7, 7),
}

# FiLM workloads use batched kernels [B, H, Kx, Ky]; others use shared [1, H, Kx, Ky]
FILM_WORKLOADS = {"FiLM 32x32 k=7", "FiLM 14x14 k=14"}


def run_benchmarks(warmup: int, iters: int) -> None:
    """Run all fftconv2d benchmarks across workloads and implementations."""
    device = "cuda"
    has_subq = _has_subq_ops()

    if has_subq:
        from subquadratic_ops_torch.fft_conv2d import fft_conv2d as subq_fft_conv2d

    # -----------------------------------------------------------------------
    # Build compiled variants once
    # -----------------------------------------------------------------------
    fftconv2d_compiled_default = torch.compile(fftconv2d_fp32_bhl)
    fftconv2d_compiled_max = torch.compile(fftconv2d_fp32_bhl, mode="max-autotune")

    # -----------------------------------------------------------------------
    # Forward-only benchmarks
    # -----------------------------------------------------------------------
    print("=" * 90)
    print("FORWARD ONLY (median ms, lower is better)")
    print("=" * 90)

    header = f"{'Workload':<25} {'Eager':>8} {'Compile':>8} {'MaxAuto':>8}"
    if has_subq:
        header += f" {'SubqOps':>8} {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for name, (B, H, X, Y, Kx, Ky) in WORKLOADS.items():
        is_film = name in FILM_WORKLOADS
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
        k_bhl = torch.randn(B if is_film else 1, H, Kx, Ky, device=device, dtype=torch.float32)
        k_subq = k_bhl.squeeze(0) if not is_film else k_bhl

        t_eager = _time_fn(lambda: fftconv2d_fp32_bhl(x, k_bhl, None), warmup, iters)
        t_compiled = _time_fn(lambda: fftconv2d_compiled_default(x, k_bhl, None), warmup, iters)
        t_max = _time_fn(lambda: fftconv2d_compiled_max(x, k_bhl, None), warmup, iters)

        row = f"{name:<25} {t_eager:>7.3f}ms {t_compiled:>7.3f}ms {t_max:>7.3f}ms"

        if has_subq:
            t_subq = _time_fn(lambda: subq_fft_conv2d(x, k_subq), warmup, iters)
            best_torch = min(t_eager, t_compiled, t_max)
            speedup = best_torch / t_subq
            row += f" {t_subq:>7.3f}ms {speedup:>7.2f}x"

        print(row)

    # -----------------------------------------------------------------------
    # Forward + backward benchmarks
    # -----------------------------------------------------------------------
    print()
    print("=" * 90)
    print("FORWARD + BACKWARD (median ms, lower is better)")
    print("=" * 90)

    header = f"{'Workload':<25} {'Eager':>8} {'Compile':>8} {'MaxAuto':>8}"
    if has_subq:
        header += f" {'SubqOps':>8} {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for name, (B, H, X, Y, Kx, Ky) in WORKLOADS.items():
        is_film = name in FILM_WORKLOADS
        x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32, requires_grad=True)
        k_bhl = torch.randn(B if is_film else 1, H, Kx, Ky, device=device, dtype=torch.float32, requires_grad=True)
        k_subq = (
            torch.randn(
                B if is_film else H,
                *([H, Kx, Ky] if is_film else [Kx, Ky]),
                device=device,
                dtype=torch.float32,
                requires_grad=True,
            )
            if has_subq
            else None
        )

        t_eager = _time_fwd_bwd(lambda: fftconv2d_fp32_bhl(x, k_bhl, None), x, k_bhl, warmup, iters)
        t_compiled = _time_fwd_bwd(lambda: fftconv2d_compiled_default(x, k_bhl, None), x, k_bhl, warmup, iters)
        t_max = _time_fwd_bwd(lambda: fftconv2d_compiled_max(x, k_bhl, None), x, k_bhl, warmup, iters)

        row = f"{name:<25} {t_eager:>7.3f}ms {t_compiled:>7.3f}ms {t_max:>7.3f}ms"

        if has_subq and k_subq is not None:
            x_subq = x.detach().clone().requires_grad_(True)
            t_subq = _time_fwd_bwd(lambda: subq_fft_conv2d(x_subq, k_subq), x_subq, k_subq, warmup, iters)
            best_torch = min(t_eager, t_compiled, t_max)
            speedup = best_torch / t_subq
            row += f" {t_subq:>7.3f}ms {speedup:>7.2f}x"

        print(row)

    # -----------------------------------------------------------------------
    # Memory comparison
    # -----------------------------------------------------------------------
    if has_subq:
        print()
        print("=" * 90)
        print("PEAK MEMORY (MB, forward only)")
        print("=" * 90)
        print(f"{'Workload':<25} {'torch.fft':>10} {'SubqOps':>10} {'Savings':>10}")
        print("-" * 55)

        for name, (B, H, X, Y, Kx, Ky) in WORKLOADS.items():
            is_film = name in FILM_WORKLOADS

            # torch.fft
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
            k_bhl = torch.randn(B if is_film else 1, H, Kx, Ky, device=device, dtype=torch.float32)
            _ = fftconv2d_fp32_bhl(x, k_bhl, None)
            torch.cuda.synchronize()
            mem_torch = torch.cuda.max_memory_allocated() / 1024**2

            # subq_ops
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(B, H, X, Y, device=device, dtype=torch.float32)
            k_subq = (
                torch.randn(B, H, Kx, Ky, device=device, dtype=torch.float32)
                if is_film
                else torch.randn(H, Kx, Ky, device=device, dtype=torch.float32)
            )
            _ = subq_fft_conv2d(x, k_subq)
            torch.cuda.synchronize()
            mem_subq = torch.cuda.max_memory_allocated() / 1024**2

            savings = (1 - mem_subq / mem_torch) * 100 if mem_torch > 0 else 0
            print(f"{name:<25} {mem_torch:>9.1f}MB {mem_subq:>9.1f}MB {savings:>9.1f}%")


def main():
    """Parse arguments and run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark fftconv2d implementations")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"subquadratic_ops_torch: {'available' if _has_subq_ops() else 'NOT INSTALLED'}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print()

    run_benchmarks(args.warmup, args.iters)


if __name__ == "__main__":
    main()
