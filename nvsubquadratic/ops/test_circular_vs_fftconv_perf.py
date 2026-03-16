# TODO: Add license header here


"""Performance and memory comparison: circular vs conventional FFT conv2d (BHL).

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/test_circular_vs_fftconv_perf.py
"""

import time

import torch
from einops import rearrange

from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_bhl,
    circular_fftconv2d_bhl,
    circular_fftconv3d_bhl,
)
from nvsubquadratic.ops.fftconv import (
    fftconv1d_fp32_bhl,
    fftconv2d_fp32_bhl,
    fftconv3d_fp32_bhl,
)


def bench_op(fn, label: str, repetitions: int, device: str, warmup: int = 10):
    # Warmup
    for _ in range(warmup):
        _ = fn()
    # Measure
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        torch.cuda.synchronize()
    total = 0.0
    for _ in range(repetitions):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total += t1 - t0
    avg = total / repetitions
    peak_mem = None
    if device == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(torch.cuda.current_device())
    print(f"{label}: {avg:.6f}s avg" + (f", peak_mem {peak_mem / 1e6:.1f} MB" if peak_mem is not None else ""))
    return avg, peak_mem


def test_perf_mem_circular_vs_fftconv1d():
    """Compares speed and peak memory of circular vs conventional fftconv1d_fp32_bhl across lengths."""
    print("🚀 Circular vs Conventional FFTConv1D Perf/Memory (multi-length)")
    # Configs
    B, H = 8, 64
    lengths = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}, B={B}, H={H}")

    for N in lengths:
        L = N
        K = N  # full-size kernel
        warmup = 10
        repetitions = 20
        print(f"\nLength: {L} (kernel={K}, warmup={warmup}, reps={repetitions})")
        try:
            x = torch.randn(B, H, L, dtype=dtype, device=device)
            k = torch.randn(H, K, dtype=dtype, device=device)
            k_bhl = rearrange(k, "h k -> 1 h k")

            avg_circ, mem_circ = bench_op(
                lambda x=x, k_bhl=k_bhl: circular_fftconv1d_bhl(x, k_bhl),
                "circular_fftconv1d_bhl",
                repetitions,
                device,
                warmup,
            )
            avg_fft, mem_fft = bench_op(
                lambda x=x, k_bhl=k_bhl: fftconv1d_fp32_bhl(x, k_bhl),
                "fftconv1d_fp32_bhl (conventional)",
                repetitions,
                device,
                warmup,
            )
            if avg_circ > 0:
                print(f"Speedup (conventional / circular): {avg_fft / avg_circ:.2f}x")
            if mem_circ is not None and mem_fft is not None and mem_circ > 0:
                print(f"Peak memory ratio (conventional / circular): {mem_fft / mem_circ:.2f}x")
        except RuntimeError as e:
            print(f"Skipped L={L} due to error: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()
        finally:
            del x, k
            if device == "cuda":
                torch.cuda.empty_cache()
    print("-" * 50)


def test_perf_mem_circular_vs_fftconv2d():
    """Compares speed and peak memory of circular vs conventional fftconv2d_fp32_bhl across sizes."""
    print("🚀 Circular vs Conventional FFTConv2D Perf/Memory (multi-size)")
    # Configs
    B, H = 4, 64
    sizes = [64, 128, 256, 512, 1024, 2048]
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}, B={B}, H={H}")

    for N in sizes:
        X_in = Y_in = N
        Kx = Ky = N  # full-size kernel
        # Adjust warmup/reps for larger sizes
        warmup = 10
        repetitions = 20
        print(f"\nSize: {N}x{N} (kernel={Kx}x{Ky}, warmup={warmup}, reps={repetitions})")
        try:
            x = torch.randn(B, H, X_in, Y_in, dtype=dtype, device=device)
            k = torch.randn(H, Kx, Ky, dtype=dtype, device=device)
            k_bhl = rearrange(k, "h kx ky -> 1 h kx ky")

            avg_circ, mem_circ = bench_op(
                lambda x=x, k_bhl=k_bhl: circular_fftconv2d_bhl(x, k_bhl),
                "circular_fftconv2d_bhl",
                repetitions,
                device,
                warmup,
            )
            avg_fft, mem_fft = bench_op(
                lambda x=x, k_bhl=k_bhl: fftconv2d_fp32_bhl(x, k_bhl),
                "fftconv2d_fp32_bhl (conventional)",
                repetitions,
                device,
                warmup,
            )
            if avg_circ > 0:
                print(f"Speedup (conventional / circular): {avg_fft / avg_circ:.2f}x")
            if mem_circ is not None and mem_fft is not None and mem_circ > 0:
                print(f"Peak memory ratio (conventional / circular): {mem_fft / mem_circ:.2f}x")
        except RuntimeError as e:
            print(f"Skipped {N}x{N} due to error: {e}")
            # Try to release memory between sizes
            if device == "cuda":
                torch.cuda.empty_cache()
        finally:
            del x, k
            if device == "cuda":
                torch.cuda.empty_cache()
    print("-" * 50)


def test_perf_mem_circular_vs_fftconv3d():
    """Compares speed and peak memory of circular vs conventional fftconv3d_fp32_bhl across cube sizes."""
    print("🚀 Circular vs Conventional FFTConv3D Perf/Memory (multi-cube)")
    # Configs
    B, H = 2, 16
    cubes = [16, 24, 32, 48, 64, 96, 128]
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}, B={B}, H={H}")

    for N in cubes:
        X = Y = Z = N
        Kx = Ky = Kz = N  # full-size kernel
        warmup = 10
        repetitions = 20
        print(f"\nCube: {N}^3 (kernel={Kx}x{Ky}x{Kz}, warmup={warmup}, reps={repetitions})")
        try:
            x = torch.randn(B, H, X, Y, Z, dtype=dtype, device=device)
            k = torch.randn(H, Kx, Ky, Kz, dtype=dtype, device=device)
            k_bhl = rearrange(k, "h kx ky kz -> 1 h kx ky kz")

            avg_circ, mem_circ = bench_op(
                lambda x=x, k_bhl=k_bhl: circular_fftconv3d_bhl(x, k_bhl),
                "circular_fftconv3d_bhl",
                repetitions,
                device,
                warmup,
            )
            avg_fft, mem_fft = bench_op(
                lambda x=x, k_bhl=k_bhl: fftconv3d_fp32_bhl(x, k_bhl),
                "fftconv3d_fp32_bhl (conventional)",
                repetitions,
                device,
                warmup,
            )
            if avg_circ > 0:
                print(f"Speedup (conventional / circular): {avg_fft / avg_circ:.2f}x")
            if mem_circ is not None and mem_fft is not None and mem_circ > 0:
                print(f"Peak memory ratio (conventional / circular): {mem_fft / mem_circ:.2f}x")
        except RuntimeError as e:
            print(f"Skipped {N}^3 due to error: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()
        finally:
            del x, k
            if device == "cuda":
                torch.cuda.empty_cache()
    print("-" * 50)


if __name__ == "__main__":
    """
    🚀 Circular vs Conventional FFTConv1D Perf/Memory (multi-length)
    Using device: cuda, dtype: torch.float32, B=8, H=64

    Length: 1024 (kernel=1024, warmup=10, reps=20)
    circular_fftconv1d_bhl: 0.000076s avg, peak_mem 11.0 MB
    fftconv1d_fp32_bhl (conventional): 0.000110s avg, peak_mem 15.3 MB
    Speedup (conventional / circular): 1.44x
    Peak memory ratio (conventional / circular): 1.39x

    Length: 2048 (kernel=2048, warmup=10, reps=20)
    circular_fftconv1d_bhl: 0.000070s avg, peak_mem 22.0 MB
    fftconv1d_fp32_bhl (conventional): 0.000098s avg, peak_mem 30.7 MB
    Speedup (conventional / circular): 1.40x
    Peak memory ratio (conventional / circular): 1.39x

    Length: 4096 (kernel=4096, warmup=10, reps=20)
    circular_fftconv1d_bhl: 0.000073s avg, peak_mem 44.1 MB
    fftconv1d_fp32_bhl (conventional): 0.000116s avg, peak_mem 61.4 MB
    Speedup (conventional / circular): 1.58x
    Peak memory ratio (conventional / circular): 1.39x

    Length: 8192 (kernel=8192, warmup=10, reps=20)
    circular_fftconv1d_bhl: 0.000088s avg, peak_mem 88.2 MB
    fftconv1d_fp32_bhl (conventional): 0.000184s avg, peak_mem 122.8 MB
    Speedup (conventional / circular): 2.09x
    Peak memory ratio (conventional / circular): 1.39x

    Length: 16384 (kernel=16384, warmup=10, reps=20)
    circular_fftconv1d_bhl: 0.000170s avg, peak_mem 176.3 MB
    fftconv1d_fp32_bhl (conventional): 0.000387s avg, peak_mem 245.5 MB
    Speedup (conventional / circular): 2.27x
    Peak memory ratio (conventional / circular): 1.39x

    Length: 32768 (kernel=32768, warmup=10, reps=20)
    circular_fftconv1d_bhl: 0.000356s avg, peak_mem 352.6 MB
    fftconv1d_fp32_bhl (conventional): 0.000761s avg, peak_mem 491.0 MB
    Speedup (conventional / circular): 2.14x
    Peak memory ratio (conventional / circular): 1.39x

    Length: 65536 (kernel=65536, warmup=10, reps=20)
    circular_fftconv1d_bhl: 0.000751s avg, peak_mem 705.2 MB
    fftconv1d_fp32_bhl (conventional): 0.001945s avg, peak_mem 982.0 MB
    Speedup (conventional / circular): 2.59x
    Peak memory ratio (conventional / circular): 1.39x
    --------------------------------------------------
    🚀 Circular vs Conventional FFTConv2D Perf/Memory (multi-size)
    Using device: cuda, dtype: torch.float32, B=4, H=64

    Size: 64x64 (kernel=64x64, warmup=10, reps=20)
    circular_fftconv2d_bhl: 0.000083s avg, peak_mem 24.0 MB
    fftconv2d_fp32_bhl (conventional): 0.000123s avg, peak_mem 46.5 MB
    Speedup (conventional / circular): 1.48x
    Peak memory ratio (conventional / circular): 1.94x

    Size: 128x128 (kernel=128x128, warmup=10, reps=20)
    circular_fftconv2d_bhl: 0.000113s avg, peak_mem 93.7 MB
    fftconv2d_fp32_bhl (conventional): 0.000304s avg, peak_mem 183.3 MB
    Speedup (conventional / circular): 2.70x
    Peak memory ratio (conventional / circular): 1.96x

    Size: 256x256 (kernel=256x256, warmup=10, reps=20)
    circular_fftconv2d_bhl: 0.000440s avg, peak_mem 371.7 MB
    fftconv2d_fp32_bhl (conventional): 0.001242s avg, peak_mem 729.0 MB
    Speedup (conventional / circular): 2.82x
    Peak memory ratio (conventional / circular): 1.96x

    Size: 512x512 (kernel=512x512, warmup=10, reps=20)
    circular_fftconv2d_bhl: 0.001661s avg, peak_mem 1484.9 MB
    fftconv2d_fp32_bhl (conventional): 0.005082s avg, peak_mem 2911.1 MB
    Speedup (conventional / circular): 3.06x
    Peak memory ratio (conventional / circular): 1.96x

    Size: 1024x1024 (kernel=1024x1024, warmup=10, reps=20)
    circular_fftconv2d_bhl: 0.006314s avg, peak_mem 5918.5 MB
    fftconv2d_fp32_bhl (conventional): 0.022736s avg, peak_mem 11629.3 MB
    Speedup (conventional / circular): 3.60x
    Peak memory ratio (conventional / circular): 1.96x

    Size: 2048x2048 (kernel=2048x2048, warmup=10, reps=20)
    circular_fftconv2d_bhl: 0.027740s avg, peak_mem 23658.9 MB
    fftconv2d_fp32_bhl (conventional): 0.088777s avg, peak_mem 46482.7 MB
    Speedup (conventional / circular): 3.20x
    Peak memory ratio (conventional / circular): 1.96x
    --------------------------------------------------
    🚀 Circular vs Conventional FFTConv3D Perf/Memory (multi-cube)
    Using device: cuda, dtype: torch.float32, B=2, H=16

    Cube: 16^3 (kernel=16x16x16, warmup=10, reps=20)
    circular_fftconv3d_bhl: 0.000093s avg, peak_mem 26.3 MB
    fftconv3d_fp32_bhl (conventional): 0.000126s avg, peak_mem 32.9 MB
    Speedup (conventional / circular): 1.35x
    Peak memory ratio (conventional / circular): 1.25x

    Cube: 24^3 (kernel=24x24x24, warmup=10, reps=20)
    circular_fftconv3d_bhl: 0.000093s avg, peak_mem 35.0 MB
    fftconv3d_fp32_bhl (conventional): 0.000133s avg, peak_mem 54.0 MB
    Speedup (conventional / circular): 1.44x
    Peak memory ratio (conventional / circular): 1.54x

    Cube: 32^3 (kernel=32x32x32, warmup=10, reps=20)
    circular_fftconv3d_bhl: 0.000098s avg, peak_mem 49.2 MB
    fftconv3d_fp32_bhl (conventional): 0.000171s avg, peak_mem 95.2 MB
    Speedup (conventional / circular): 1.75x
    Peak memory ratio (conventional / circular): 1.93x

    Cube: 48^3 (kernel=48x48x48, warmup=10, reps=20)
    circular_fftconv3d_bhl: 0.000155s avg, peak_mem 110.6 MB
    fftconv3d_fp32_bhl (conventional): 0.000527s avg, peak_mem 264.5 MB
    Speedup (conventional / circular): 3.41x
    Peak memory ratio (conventional / circular): 2.39x

    Cube: 64^3 (kernel=64x64x64, warmup=10, reps=20)
    circular_fftconv3d_bhl: 0.000311s avg, peak_mem 229.7 MB
    fftconv3d_fp32_bhl (conventional): 0.001339s avg, peak_mem 592.9 MB
    Speedup (conventional / circular): 4.30x
    Peak memory ratio (conventional / circular): 2.58x

    Cube: 96^3 (kernel=96x96x96, warmup=10, reps=20)
    circular_fftconv3d_bhl: 0.001268s avg, peak_mem 716.0 MB
    fftconv3d_fp32_bhl (conventional): 0.003504s avg, peak_mem 1938.0 MB
    Speedup (conventional / circular): 2.76x
    Peak memory ratio (conventional / circular): 2.71x

    Cube: 128^3 (kernel=128x128x128, warmup=10, reps=20)
    circular_fftconv3d_bhl: 0.002313s avg, peak_mem 1662.1 MB
    fftconv3d_fp32_bhl (conventional): 0.008021s avg, peak_mem 4552.5 MB
    Speedup (conventional / circular): 3.47x
    Peak memory ratio (conventional / circular): 2.74x
    --------------------------------------------------
    """
    test_perf_mem_circular_vs_fftconv1d()
    test_perf_mem_circular_vs_fftconv2d()
    test_perf_mem_circular_vs_fftconv3d()
