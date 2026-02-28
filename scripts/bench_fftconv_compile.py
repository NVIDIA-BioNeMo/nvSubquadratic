"""Benchmark: torch.compile speed for fftconv2d_bhl with batch-dependent kernels.

Compares three strategies for handling the complex64 multiply:
  1. out-of-place (current fix) -- full compile
  2. torch.compiler.disable on just the multiply -- rest compiles
  3. eager (no compile at all) -- baseline
"""

import time
import torch

B, H, X, Y = 4, 384, 15, 14
WARMUP, REPS = 20, 200


def fftconv2d_outofplace(x, kernel, shortcut):
    """Current fix: out-of-place multiply, fully compiled."""
    _, _, K_x, K_y = kernel.shape
    fft_shape = (min(X + (K_x + 1) // 2, 2 * X), min(Y + (K_y + 1) // 2, 2 * Y))
    fft_x = torch.fft.rfft2(x, s=fft_shape, dim=(2, 3))
    fft_kernel = torch.fft.rfft2(kernel, s=fft_shape, dim=(2, 3))
    fft_x = fft_x * fft_kernel
    crop_x, crop_y = K_x // 2, K_y // 2
    y = torch.fft.irfft2(fft_x, s=fft_shape, dim=(2, 3))[..., crop_x:crop_x + X, crop_y:crop_y + Y]
    if shortcut is not None:
        y = y + shortcut.view(1, -1, 1, 1) * x
    return y


@torch.compiler.disable
def _complex_mul_disabled(a, b):
    return a * b


def fftconv2d_disable_mul(x, kernel, shortcut):
    """Only the complex multiply is excluded from compile."""
    _, _, K_x, K_y = kernel.shape
    fft_shape = (min(X + (K_x + 1) // 2, 2 * X), min(Y + (K_y + 1) // 2, 2 * Y))
    fft_x = torch.fft.rfft2(x, s=fft_shape, dim=(2, 3))
    fft_kernel = torch.fft.rfft2(kernel, s=fft_shape, dim=(2, 3))
    fft_x = _complex_mul_disabled(fft_x, fft_kernel)
    crop_x, crop_y = K_x // 2, K_y // 2
    y = torch.fft.irfft2(fft_x, s=fft_shape, dim=(2, 3))[..., crop_x:crop_x + X, crop_y:crop_y + Y]
    if shortcut is not None:
        y = y + shortcut.view(1, -1, 1, 1) * x
    return y


def bench(fn, label, x, kernel, shortcut):
    for _ in range(WARMUP):
        fn(x, kernel, shortcut)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn(x, kernel, shortcut)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / REPS * 1000
    print(f"  {label:40s}  {elapsed:.3f} ms/call")
    return elapsed


def bench_backward(fn, label, x, kernel, shortcut):
    for _ in range(WARMUP):
        x_ = x.detach().requires_grad_(True)
        k_ = kernel.detach().requires_grad_(True)
        y = fn(x_, k_, shortcut)
        y.sum().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        x_ = x.detach().requires_grad_(True)
        k_ = kernel.detach().requires_grad_(True)
        y = fn(x_, k_, shortcut)
        y.sum().backward()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / REPS * 1000
    print(f"  {label:40s}  {elapsed:.3f} ms/call")
    return elapsed


if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Shape: x=[{B},{H},{X},{Y}], kernel=[{B},{H},{X},{Y}] (batch-dependent)")
    print(f"Warmup={WARMUP}, Reps={REPS}\n")

    x = torch.randn(B, H, X, Y, device="cuda", dtype=torch.float32)
    kernel = torch.randn(B, H, X, Y, device="cuda", dtype=torch.float32)
    shortcut = torch.randn(H, device="cuda", dtype=torch.float32)

    compiled_outofplace = torch.compile(fftconv2d_outofplace, mode="max-autotune")
    compiled_disable_mul = torch.compile(fftconv2d_disable_mul, mode="max-autotune")

    print("=== Forward only ===")
    bench(fftconv2d_outofplace, "eager (no compile)", x, kernel, shortcut)
    bench(compiled_outofplace, "compile out-of-place *", x, kernel, shortcut)
    bench(compiled_disable_mul, "compile + disable(mul)", x, kernel, shortcut)

    print("\n=== Forward + Backward ===")
    bench_backward(fftconv2d_outofplace, "eager (no compile)", x, kernel, shortcut)
    bench_backward(compiled_outofplace, "compile out-of-place *", x, kernel, shortcut)
    bench_backward(compiled_disable_mul, "compile + disable(mul)", x, kernel, shortcut)
