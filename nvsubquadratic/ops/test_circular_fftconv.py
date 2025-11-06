# TODO: Add license header here


"""Test the 2D circular FFT convolution against reference circular padded conv2d.

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/test_circular_fftconv.py
"""

import time

import torch
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_bhl,
    circular_fftconv2d_bhl,
    circular_fftconv3d_bhl,
)


def test_circular_fftconv1d():
    """Tests the 1D circular FFT-based convolution against circular padded F.conv1d."""
    print("🚀 Running 1D Circular Convolution Test...")
    # Setup test parameters
    B, H, L = 4, 16, 8192
    K = L
    repetitions = 20

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: {B, H, L}; Kernel: {K}")

    # Create random input and kernel tensors
    input_tensor = torch.randn(B, H, L, dtype=dtype, device=device)
    kernel = torch.randn(H, K, dtype=dtype, device=device)

    # --- 1. Reference: Spatial 1D Convolution with circular padding and 'same' output ---
    # F.conv1d computes cross-correlation; flip to get convolution
    kernel_for_conv1d = kernel.unsqueeze(1)  # [H, 1, K]
    kernel_flipped = torch.flip(kernel_for_conv1d, dims=[-1])

    pad_left = K // 2
    pad_right = K - 1 - K // 2

    # Warm-up
    for _ in range(5):
        padded = F.pad(input_tensor, (pad_left, pad_right), mode="circular")
        _ = F.conv1d(padded, kernel_flipped, groups=H, padding=0)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        padded = F.pad(input_tensor, (pad_left, pad_right), mode="circular")
        ref_output = F.conv1d(padded, kernel_flipped, groups=H, padding=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time_ndconv = total_time / repetitions
    print(f"F.conv1d (circular) took: {average_time_ndconv:.4f} seconds on average")

    # --- 2. FFT-based circular implementation ---
    kernel_fft = rearrange(kernel, "h k -> 1 h k")
    # Warm-up
    for _ in range(5):
        _ = circular_fftconv1d_bhl(input_tensor, kernel_fft)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fft_output = circular_fftconv1d_bhl(input_tensor, kernel_fft)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time_fftconv_nd = total_time / repetitions
    print(f"circular_fftconv1d_bhl took: {average_time_fftconv_nd:.4f} seconds on average")
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


def test_circular_fftconv2d():
    """Tests the 2D circular FFT-based convolution against circular padded F.conv2d."""
    print("🚀 Running 2D Circular Convolution Test...")
    # Setup test parameters
    B, H, X_in, Y_in = 4, 16, 256, 256
    Kx, Ky = X_in, Y_in
    repetitions = 20

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: {B, H, X_in, Y_in}; Kernel: {Kx}x{Ky}")

    # Create random input and kernel tensors
    input_tensor = torch.randn(B, H, X_in, Y_in, dtype=dtype, device=device)
    kernel = torch.randn(H, Kx, Ky, dtype=dtype, device=device)

    # --- 1. Reference: Spatial 2D Convolution with circular padding and 'same' output ---
    # F.conv2d computes cross-correlation; flip to get convolution
    kernel_for_conv2d = kernel.unsqueeze(1)  # [H, 1, Kx, Ky]
    kernel_flipped = torch.flip(kernel_for_conv2d, dims=[-1, -2])

    # Circular padding to obtain same-sized output
    pad_w_left = Ky // 2
    pad_w_right = Ky - 1 - Ky // 2
    pad_h_top = Kx // 2
    pad_h_bottom = Kx - 1 - Kx // 2

    # Warm-up
    for _ in range(5):
        padded = F.pad(input_tensor, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode="circular")
        _ = F.conv2d(padded, kernel_flipped, groups=H, padding=0)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        padded = F.pad(input_tensor, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode="circular")
        ref_output = F.conv2d(padded, kernel_flipped, groups=H, padding=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time_ndconv = total_time / repetitions
    print(f"F.conv2d (circular) took: {average_time_ndconv:.4f} seconds on average")

    # --- 2. FFT-based circular implementation ---
    kernel_fft = rearrange(kernel, "h kx ky -> 1 h kx ky")
    # Warm-up
    for _ in range(5):
        _ = circular_fftconv2d_bhl(input_tensor, kernel_fft)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fft_output = circular_fftconv2d_bhl(input_tensor, kernel_fft)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time_fftconv_nd = total_time / repetitions
    print(f"circular_fftconv2d_bhl took: {average_time_fftconv_nd:.4f} seconds on average")
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


def test_circular_fftconv3d():
    """Tests the 3D circular FFT-based convolution against circular padded F.conv3d."""
    print("🚀 Running 3D Circular Convolution Test...")
    # Setup test parameters
    B, H, X_in, Y_in, Z_in = 2, 8, 64, 64, 64
    Kx, Ky, Kz = X_in, Y_in, Z_in
    repetitions = 10

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: {B, H, X_in, Y_in, Z_in}; Kernel: {Kx}x{Ky}x{Kz}")

    # Create random input and kernel tensors
    input_tensor = torch.randn(B, H, X_in, Y_in, Z_in, dtype=dtype, device=device)
    kernel = torch.randn(H, Kx, Ky, Kz, dtype=dtype, device=device)

    # --- 1. Reference: Spatial 3D Convolution with circular padding and 'same' output ---
    kernel_for_conv3d = kernel.unsqueeze(1)  # [H, 1, Kx, Ky, Kz]
    kernel_flipped = torch.flip(kernel_for_conv3d, dims=[-1, -2, -3])

    pad_w_left = Kz // 2
    pad_w_right = Kz - 1 - Kz // 2
    pad_h_top = Ky // 2
    pad_h_bottom = Ky - 1 - Ky // 2
    pad_d_front = Kx // 2
    pad_d_back = Kx - 1 - Kx // 2

    # Warm-up
    for _ in range(3):
        padded = F.pad(
            input_tensor,
            (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back),
            mode="circular",
        )
        _ = F.conv3d(padded, kernel_flipped, groups=H, padding=0)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        padded = F.pad(
            input_tensor,
            (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back),
            mode="circular",
        )
        ref_output = F.conv3d(padded, kernel_flipped, groups=H, padding=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time_ndconv = total_time / repetitions
    print(f"F.conv3d (circular) took: {average_time_ndconv:.4f} seconds on average")

    # --- 2. FFT-based circular implementation ---
    kernel_fft = rearrange(kernel, "h kx ky kz -> 1 h kx ky kz")
    # Warm-up
    for _ in range(3):
        _ = circular_fftconv3d_bhl(input_tensor, kernel_fft)

    total_time = 0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fft_output = circular_fftconv3d_bhl(input_tensor, kernel_fft)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        total_time += end_time - start_time

    average_time_fftconv_nd = total_time / repetitions
    print(f"circular_fftconv3d_bhl took: {average_time_fftconv_nd:.4f} seconds on average")
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
    test_circular_fftconv1d()
    test_circular_fftconv2d()
    test_circular_fftconv3d()
