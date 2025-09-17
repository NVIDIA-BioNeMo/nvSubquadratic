# David W. Romero, 2025-09-09

"""Test the FFT-based 1D (causal & non-causal), 2D, and 3D convolutions against the reference F.conv1d, F.conv2d, and F.conv3d.

We only compare against the *_bhl variants, as the fastest *_blh variants use this implementation internally.

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/fftconv_test.py
"""

import time

import torch
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.ops.fftconv import causal_fftconv1d_bhl, fftconv1d_bhl, fftconv2d_bhl, fftconv3d_bhl


def test_fftconv1d():
    """Tests the 1D FFT-based convolution against the reference F.conv1d."""
    print("🚀 Running 1D 'Same' Padding Test...")
    # Setup test parameters
    B, H, L = 4, 16, 256 * 256  # 2^16
    K = L
    repetitions = 20

    dtype = torch.float32  # Using float32 for more realistic performance comparison
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: {B, H, L}")

    # Create random input and kernel tensors
    input_tensor = torch.randn(B, H, L, dtype=dtype, device=device)
    kernel = torch.randn(H, K, dtype=dtype, device=device)

    # --- 1. Reference: Spatial 1D Convolution with 'same' padding ---
    # F.conv1d computes cross-correlation. To compute convolution, we flip the kernel.
    kernel_for_conv1d = kernel.unsqueeze(1)
    kernel_flipped = torch.flip(kernel_for_conv1d, dims=[-1])

    # Warm-up
    for _ in range(5):
        _ = F.conv1d(input_tensor, kernel_flipped, groups=H, padding="same")

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ref_output = F.conv1d(input_tensor, kernel_flipped, groups=H, padding="same")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    # Average time per iteration
    average_time_ndconv = total_time / repetitions
    print(f"F.conv1d took: {average_time_ndconv:.4f} seconds on average")

    # --- 2. FFT-based implementation ---
    kernel = rearrange(kernel, "h k -> 1 h k")
    # Warm-up
    for _ in range(5):
        _ = fftconv1d_bhl(input_tensor, kernel)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fft_output = fftconv1d_bhl(input_tensor, kernel)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    # Average time per iteration
    average_time_fftconv_nd = total_time / repetitions
    print(f"fftconv1d_bhl took: {average_time_fftconv_nd:.4f} seconds on average")

    print(f"Speedup: {average_time_ndconv / average_time_fftconv_nd:.2f}x")

    # --- 3. Comparison ---
    print("\nReference output shape:", ref_output.shape)
    print("FFT output shape:    ", fft_output.shape)

    assert ref_output.shape == fft_output.shape, "Shape mismatch!"

    max_abs_diff = torch.max(torch.abs(ref_output - fft_output)).item()
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Reference: max {ref_output.abs().max()}, mean {ref_output.abs().mean()}")
    relative_diff = max_abs_diff / ref_output.abs().max()
    print(f"Relative difference: {relative_diff:.2e}")
    is_close = relative_diff < 5e-5
    if is_close:
        print("\n✅ Test PASSED: Outputs match.")
    else:
        print("\n❌ Test FAILED: Outputs do not match.")

    print("-" * 50)


def test_causal_fftconv1d():
    """Tests the 1D causal FFT-based convolution against the reference F.conv1d (with causal padding)."""
    print("🚀 Running 1D Causal Padding Test...")
    # Setup test parameters
    B, H, L = 4, 16, 256 * 256  # 2^16
    K = L
    repetitions = 20

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: {B, H, L}")

    # Create random input and kernel tensors
    input_tensor = torch.randn(B, H, L, dtype=dtype, device=device)
    kernel = torch.randn(H, K, dtype=dtype, device=device)

    # --- 1. Reference: Spatial 1D Convolution with causal padding ---
    kernel_for_conv1d = kernel.unsqueeze(1)
    kernel_flipped = torch.flip(kernel_for_conv1d, dims=[-1])

    # Causal padding
    padding = K - 1
    padded_input = F.pad(input_tensor, (padding, 0))

    # Warm-up
    for _ in range(5):
        _ = F.conv1d(padded_input, kernel_flipped, groups=H, padding="valid")

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ref_output = F.conv1d(padded_input, kernel_flipped, groups=H, padding="valid")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    # Average time per iteration
    average_time_ndconv = total_time / repetitions
    print(f"F.conv1d (causal) took: {average_time_ndconv:.4f} seconds on average")

    # --- 2. FFT-based implementation ---
    # The fftconv function expects kernel shape (1, h, k)
    kernel_fft = rearrange(kernel, "h k -> 1 h k")
    # Warm-up
    for _ in range(5):
        _ = causal_fftconv1d_bhl(input_tensor, kernel_fft)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fft_output = causal_fftconv1d_bhl(input_tensor, kernel_fft)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    # Average time per iteration
    average_time_fftconv_nd = total_time / repetitions
    print(f"causal_fftconv1d_bhl took: {average_time_fftconv_nd:.4f} seconds on average")

    print(f"Speedup: {average_time_ndconv / average_time_fftconv_nd:.2f}x")

    # --- 3. Comparison ---
    print("\nReference output shape:", ref_output.shape)
    print("FFT output shape:    ", fft_output.shape)

    assert ref_output.shape == fft_output.shape, "Shape mismatch!"

    max_abs_diff = torch.max(torch.abs(ref_output - fft_output)).item()
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Reference: max {ref_output.abs().max()}, mean {ref_output.abs().mean()}")
    relative_diff = max_abs_diff / ref_output.abs().max()
    print(f"Relative difference: {relative_diff:.2e}")
    is_close = relative_diff < 5e-5
    if is_close:
        print("\n✅ Test PASSED: Outputs match.")
    else:
        print("\n❌ Test FAILED: Outputs do not match.")

    print("-" * 50)


def test_fftconv2d():
    """Tests the 2D FFT-based convolution against the reference F.conv2d."""
    print("🚀 Running 2D 'Same' Padding Test...")
    # Setup test parameters
    B, H, X_in, Y_in = 4, 16, 256, 256
    Kx, Ky = X_in, Y_in
    repetitions = 20

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: {B, H, X_in, Y_in}")

    # Create random input and kernel tensors
    input_tensor = torch.randn(B, H, X_in, Y_in, dtype=dtype, device=device)
    kernel = torch.randn(H, Kx, Ky, dtype=dtype, device=device)

    # --- 1. Reference: Spatial 2D Convolution with 'same' padding ---
    kernel_for_conv2d = kernel.unsqueeze(1)
    kernel_flipped = torch.flip(kernel_for_conv2d, dims=[-1, -2])

    # Warm-up
    for _ in range(5):
        _ = F.conv2d(input_tensor, kernel_flipped, groups=H, padding="same")

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ref_output = F.conv2d(input_tensor, kernel_flipped, groups=H, padding="same")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time_ndconv = total_time / repetitions
    print(f"F.conv2d took: {average_time_ndconv:.4f} seconds on average")

    # --- 2. FFT-based implementation ---
    kernel_fft = rearrange(kernel, "h kx ky -> 1 h kx ky")
    # Warm-up
    for _ in range(5):
        _ = fftconv2d_bhl(input_tensor, kernel_fft)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fft_output = fftconv2d_bhl(input_tensor, kernel_fft)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time_fftconv_nd = total_time / repetitions
    print(f"fftconv2d_bhl took: {average_time_fftconv_nd:.4f} seconds on average")

    print(f"Speedup: {average_time_ndconv / average_time_fftconv_nd:.2f}x")

    # --- 3. Comparison ---
    print("\nReference output shape:", ref_output.shape)
    print("FFT output shape:    ", fft_output.shape)

    assert ref_output.shape == fft_output.shape, "Shape mismatch!"

    max_abs_diff = torch.max(torch.abs(ref_output - fft_output)).item()
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Reference: max {ref_output.abs().max()}, mean {ref_output.abs().mean()}")
    relative_diff = max_abs_diff / ref_output.abs().max()
    print(f"Relative difference: {relative_diff:.2e}")
    is_close = relative_diff < 5e-5
    if is_close:
        print("\n✅ Test PASSED: Outputs match.")
    else:
        print("\n❌ Test FAILED: Outputs do not match.")

    print("-" * 50)


def test_fftconv3d():
    """Tests the 3D FFT-based convolution against the reference F.conv3d."""
    print("🚀 Running 3D 'Same' Padding Test...")
    # Setup test parameters
    B, H, X_in, Y_in, Z_in = 4, 16, 32, 32, 32
    Kx, Ky, Kz = X_in, Y_in, Z_in
    repetitions = 20

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: {B, H, X_in, Y_in, Z_in}")

    # Create random input and kernel tensors
    input_tensor = torch.randn(B, H, X_in, Y_in, Z_in, dtype=dtype, device=device)
    kernel = torch.randn(H, Kx, Ky, Kz, dtype=dtype, device=device)

    # --- 1. Reference: Spatial 3D Convolution with 'same' padding ---
    kernel_for_conv3d = kernel.unsqueeze(1)
    kernel_flipped = torch.flip(kernel_for_conv3d, dims=[-1, -2, -3])

    # Warm-up
    for _ in range(5):
        _ = F.conv3d(input_tensor, kernel_flipped, groups=H, padding="same")

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ref_output = F.conv3d(input_tensor, kernel_flipped, groups=H, padding="same")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    # Average time per iteration
    average_time_ndconv = total_time / repetitions
    print(f"F.conv3d took: {average_time_ndconv:.4f} seconds on average")

    # --- 2. FFT-based implementation ---
    kernel_fft = rearrange(kernel, "h kx ky kz -> 1 h kx ky kz")
    # Warm-up
    for _ in range(5):
        _ = fftconv3d_bhl(input_tensor, kernel_fft)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fft_output = fftconv3d_bhl(input_tensor, kernel_fft)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time_fftconv_nd = total_time / repetitions
    print(f"fftconv3d_bhl took: {average_time_fftconv_nd:.4f} seconds on average")

    print(f"Speedup: {average_time_ndconv / average_time_fftconv_nd:.2f}x")

    # --- 3. Comparison ---
    print("\nReference output shape:", ref_output.shape)
    print("FFT output shape:    ", fft_output.shape)

    assert ref_output.shape == fft_output.shape, "Shape mismatch!"

    max_abs_diff = torch.max(torch.abs(ref_output - fft_output)).item()
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Reference: max {ref_output.abs().max()}, mean {ref_output.abs().mean()}")
    relative_diff = max_abs_diff / ref_output.abs().max()
    print(f"Relative difference: {relative_diff:.2e}")
    is_close = relative_diff < 5e-5
    if is_close:
        print("\n✅ Test PASSED: Outputs match.")
    else:
        print("\n❌ Test FAILED: Outputs do not match.")

    print("-" * 50)


if __name__ == "__main__":
    test_causal_fftconv1d()
    test_fftconv1d()
    test_fftconv2d()
    test_fftconv3d()
