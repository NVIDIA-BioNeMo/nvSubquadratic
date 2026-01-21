"""Test script comparing standard FFT 2D convolution with custom CUDA FFT 2D convolution.

This script compares the outputs of:
- nvsubquadratic.ops.fftconv.fftconv2d_bhl (standard PyTorch FFT, linear convolution)
- nvsubquadratic.ops.fftconv_custom.fftconv2d_bhl (custom CUDA kernel, linear convolution)

Both implementations perform LINEAR convolution (not circular), so their outputs should match
when using full-size kernels (kernel size == input size).

Note: The custom CUDA kernel requires kernel spatial dimensions to match input spatial dimensions.
"""

import argparse
import time

import torch

from nvsubquadratic.ops.fftconv import fftconv2d_bhl as fftconv2d_bhl_standard
from nvsubquadratic.ops.fftconv import fftconv2d_blh as fftconv2d_blh_standard
from nvsubquadratic.ops.fftconv_custom import fftconv2d_bhl as fftconv2d_bhl_custom
from nvsubquadratic.ops.fftconv_custom import fftconv2d_blh as fftconv2d_blh_custom


def test_correctness_bhl(
    batch_size: int = 4,
    hidden_dim: int = 64,
    x_in: int = 32,
    y_in: int = 32,
    test_shortcut: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    device: str = "cuda",
) -> bool:
    """Test that standard and custom FFT convolutions produce matching outputs (BHL layout).

    Note: Kernel size is set to match input size as required by the custom CUDA kernel.

    Args:
        batch_size: Batch size for test tensors.
        hidden_dim: Number of hidden channels.
        x_in: Input height.
        y_in: Input width.
        test_shortcut: Whether to test with shortcut tensor.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        device: Device to run tests on.

    Returns:
        True if outputs match within tolerance, False otherwise.
    """
    # Kernel size must match input size for custom kernel
    k_x, k_y = x_in, y_in

    print(f"\n{'=' * 60}")
    print("Testing BHL layout (B, H, X, Y)")
    print(f"{'=' * 60}")
    print(f"  Input shape:  ({batch_size}, {hidden_dim}, {x_in}, {y_in})")
    print(f"  Kernel shape: (1, {hidden_dim}, {k_x}, {k_y})")

    # Create test tensors
    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_dim, x_in, y_in, dtype=torch.float32, device=device)
    kernel = torch.randn(1, hidden_dim, k_x, k_y, dtype=torch.float32, device=device)
    shortcut = torch.randn(hidden_dim, dtype=torch.float32, device=device) if test_shortcut else None

    # Run standard FFT convolution
    y_standard = fftconv2d_bhl_standard(x, kernel, shortcut)

    # Run custom CUDA FFT convolution
    y_custom = fftconv2d_bhl_custom(x, kernel, shortcut)

    # Compare outputs
    max_abs_diff = (y_standard - y_custom).abs().max().item()
    max_rel_diff = ((y_standard - y_custom).abs() / (y_standard.abs() + 1e-8)).max().item()
    is_close = torch.allclose(y_standard, y_custom, atol=atol, rtol=rtol)

    print(f"\n  Results (shortcut={'Yes' if test_shortcut else 'No'}):")
    print(f"    Max absolute difference: {max_abs_diff:.2e}")
    print(f"    Max relative difference: {max_rel_diff:.2e}")
    print(f"    Outputs match (atol={atol}, rtol={rtol}): {'✓ PASS' if is_close else '✗ FAIL'}")

    return is_close


def test_correctness_blh(
    batch_size: int = 4,
    hidden_dim: int = 64,
    x_in: int = 32,
    y_in: int = 32,
    test_shortcut: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    device: str = "cuda",
) -> bool:
    """Test that standard and custom FFT convolutions produce matching outputs (BLH layout).

    Note: Kernel size is set to match input size as required by the custom CUDA kernel.

    Args:
        batch_size: Batch size for test tensors.
        hidden_dim: Number of hidden channels.
        x_in: Input height.
        y_in: Input width.
        test_shortcut: Whether to test with shortcut tensor.
        atol: Absolute tolerance for comparison.
        rtol: Relative tolerance for comparison.
        device: Device to run tests on.

    Returns:
        True if outputs match within tolerance, False otherwise.
    """
    # Kernel size must match input size for custom kernel
    k_x, k_y = x_in, y_in

    print(f"\n{'=' * 60}")
    print("Testing BLH layout (B, X, Y, H)")
    print(f"{'=' * 60}")
    print(f"  Input shape:  ({batch_size}, {x_in}, {y_in}, {hidden_dim})")
    print(f"  Kernel shape: (1, {k_x}, {k_y}, {hidden_dim})")

    # Create test tensors
    torch.manual_seed(42)
    x = torch.randn(batch_size, x_in, y_in, hidden_dim, dtype=torch.float32, device=device)
    kernel = torch.randn(1, k_x, k_y, hidden_dim, dtype=torch.float32, device=device)
    shortcut = torch.randn(hidden_dim, dtype=torch.float32, device=device) if test_shortcut else None

    # Run standard FFT convolution
    y_standard = fftconv2d_blh_standard(x, kernel, shortcut)

    # Run custom CUDA FFT convolution
    y_custom = fftconv2d_blh_custom(x, kernel, shortcut)

    # Compare outputs
    max_abs_diff = (y_standard - y_custom).abs().max().item()
    max_rel_diff = ((y_standard - y_custom).abs() / (y_standard.abs() + 1e-8)).max().item()
    is_close = torch.allclose(y_standard, y_custom, atol=atol, rtol=rtol)

    print(f"\n  Results (shortcut={'Yes' if test_shortcut else 'No'}):")
    print(f"    Max absolute difference: {max_abs_diff:.2e}")
    print(f"    Max relative difference: {max_rel_diff:.2e}")
    print(f"    Outputs match (atol={atol}, rtol={rtol}): {'✓ PASS' if is_close else '✗ FAIL'}")

    return is_close


def benchmark(
    batch_size: int = 8,
    hidden_dim: int = 256,
    x_in: int = 64,
    y_in: int = 64,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    device: str = "cuda",
) -> None:
    """Benchmark standard vs custom FFT convolution performance.

    Args:
        batch_size: Batch size for benchmark.
        hidden_dim: Number of hidden channels.
        x_in: Input height.
        y_in: Input width.
        warmup_iters: Number of warmup iterations.
        benchmark_iters: Number of benchmark iterations.
        device: Device to run benchmark on.
    """
    # Kernel size must match input size for custom kernel
    k_x, k_y = x_in, y_in

    print(f"\n{'=' * 60}")
    print("Performance Benchmark")
    print(f"{'=' * 60}")
    print(f"  Input shape:  ({batch_size}, {hidden_dim}, {x_in}, {y_in})")
    print(f"  Kernel shape: (1, {hidden_dim}, {k_x}, {k_y})")
    print(f"  Warmup iters: {warmup_iters}")
    print(f"  Benchmark iters: {benchmark_iters}")

    # Create test tensors
    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_dim, x_in, y_in, dtype=torch.float32, device=device)
    kernel = torch.randn(1, hidden_dim, k_x, k_y, dtype=torch.float32, device=device)
    shortcut = torch.randn(hidden_dim, dtype=torch.float32, device=device)

    # Warmup
    print("\n  Warming up...")
    for _ in range(warmup_iters):
        _ = fftconv2d_bhl_standard(x, kernel, shortcut)
        _ = fftconv2d_bhl_custom(x, kernel, shortcut)
    torch.cuda.synchronize()

    # Benchmark standard
    print("  Benchmarking standard FFT convolution...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        _ = fftconv2d_bhl_standard(x, kernel, shortcut)
    torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) / benchmark_iters * 1000  # ms

    # Benchmark custom
    print("  Benchmarking custom CUDA FFT convolution...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        _ = fftconv2d_bhl_custom(x, kernel, shortcut)
    torch.cuda.synchronize()
    custom_time = (time.perf_counter() - start) / benchmark_iters * 1000  # ms

    # Report results
    speedup = standard_time / custom_time
    print("\n  Results:")
    print(f"    Standard FFT:     {standard_time:.3f} ms/iter")
    print(f"    Custom CUDA FFT:  {custom_time:.3f} ms/iter")
    print(f"    Speedup:          {speedup:.2f}x {'(custom faster)' if speedup > 1 else '(standard faster)'}")


def main():
    parser = argparse.ArgumentParser(description="Compare standard and custom FFT 2D convolutions")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for tests")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Number of hidden channels")
    parser.add_argument("--x-in", type=int, default=32, help="Input height (kernel height will match)")
    parser.add_argument("--y-in", type=int, default=32, help="Input width (kernel width will match)")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Custom kernel requires CUDA.")
        return 1

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print("\nNote: Both implementations perform LINEAR convolution (with 2x FFT padding).")
    print("      The custom CUDA kernel requires kernel size == input size.")

    # Run correctness tests
    all_passed = True

    # Test BHL layout without shortcut
    all_passed &= test_correctness_bhl(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        x_in=args.x_in,
        y_in=args.y_in,
        test_shortcut=False,
        atol=args.atol,
        rtol=args.rtol,
        device=args.device,
    )

    # Test BHL layout with shortcut
    all_passed &= test_correctness_bhl(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        x_in=args.x_in,
        y_in=args.y_in,
        test_shortcut=True,
        atol=args.atol,
        rtol=args.rtol,
        device=args.device,
    )

    # Test BLH layout without shortcut
    all_passed &= test_correctness_blh(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        x_in=args.x_in,
        y_in=args.y_in,
        test_shortcut=False,
        atol=args.atol,
        rtol=args.rtol,
        device=args.device,
    )

    # Test BLH layout with shortcut
    all_passed &= test_correctness_blh(
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        x_in=args.x_in,
        y_in=args.y_in,
        test_shortcut=True,
        atol=args.atol,
        rtol=args.rtol,
        device=args.device,
    )

    # Run benchmark if requested
    if args.benchmark:
        benchmark(
            batch_size=args.batch_size * 2,  # Use larger batch for benchmark
            hidden_dim=args.hidden_dim * 4,
            x_in=args.x_in * 2,
            y_in=args.y_in * 2,
            device=args.device,
        )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    if all_passed:
        print("  All correctness tests: ✓ PASSED")
        return 0
    else:
        print("  Some correctness tests: ✗ FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
