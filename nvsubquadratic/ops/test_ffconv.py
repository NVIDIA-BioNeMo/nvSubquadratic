# TODO: Add license header here


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


def test_fftconv2d_non_depthwise():
    """Tests the 2D non-depthwise FFT-based convolution against the reference F.conv2d.

    Non-depthwise convolution: each output channel is a weighted sum of ALL input channels.
    - F.conv2d with no groups: kernel shape (H_out, H_in, Kx, Ky)
    - fftconv2d_bhl with is_depthwise=False: kernel shape (H_out, H_in, Kx, Ky)

    fftconv2d_bhl should match F.conv2d exactly (with kernel flip for conv vs cross-correlation).
    """
    print("🚀 Running 2D Non-Depthwise 'Same' Padding Test...")
    # Setup test parameters - smaller sizes for non-depthwise (memory intensive)
    B, H_in, H_out, X_in, Y_in = 2, 8, 16, 32, 32
    Kx, Ky = 15, 15  # Smaller kernel for speed

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: (B={B}, H_in={H_in}, X={X_in}, Y={Y_in})")
    print(f"Kernel shape: (H_out={H_out}, H_in={H_in}, Kx={Kx}, Ky={Ky})")

    # Create random input and kernel tensors
    input_tensor = torch.randn(B, H_in, X_in, Y_in, dtype=dtype, device=device)
    # Kernel for F.conv2d: (out_channels, in_channels, Kx, Ky)
    kernel = torch.randn(H_out, H_in, Kx, Ky, dtype=dtype, device=device)

    # --- 1. Reference: Spatial 2D Convolution (non-depthwise) with 'same' padding ---
    # F.conv2d computes cross-correlation. To compute convolution, we flip the kernel.
    kernel_flipped = torch.flip(kernel, dims=[-1, -2])

    ref_output = F.conv2d(input_tensor, kernel_flipped, groups=1, padding="same")
    print(f"Reference output shape: {ref_output.shape}")

    # --- 2. FFT-based implementation (non-depthwise) ---
    # fftconv2d_bhl expects kernel shape (H_out, H_in, Kx, Ky) for non-depthwise
    fft_output = fftconv2d_bhl(input_tensor, kernel, is_depthwise=False)
    print(f"FFT output shape: {fft_output.shape}")

    # --- 3. Comparison ---
    assert ref_output.shape == fft_output.shape, f"Shape mismatch! {ref_output.shape} vs {fft_output.shape}"

    max_abs_diff = torch.max(torch.abs(ref_output - fft_output)).item()
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Reference: max {ref_output.abs().max():.2e}, mean {ref_output.abs().mean():.2e}")
    relative_diff = max_abs_diff / ref_output.abs().max()
    print(f"Relative difference: {relative_diff:.2e}")

    is_close = relative_diff < 1e-4
    if is_close:
        print("\n✅ Test PASSED: fftconv2d_bhl matches F.conv2d exactly.")
    else:
        print("\n❌ Test FAILED: Outputs do not match.")
        print(f"Sample ref[0,0,0,:5]: {ref_output[0, 0, 0, :5]}")
        print(f"Sample fft[0,0,0,:5]: {fft_output[0, 0, 0, :5]}")

    print("-" * 50)
    return is_close


def test_fftconv2d_non_depthwise_variance_scaling():
    """Tests variance scaling behavior for non-depthwise convolution.

    For unit-variance inputs and unit-variance kernels (without proper initialization):
    - Output variance ~ H_in * Kx * Ky * input_var (scales with H_in!)

    This demonstrates why proper kernel initialization (scale 1/sqrt(H_in)) is needed
    to prevent variance explosion in non-depthwise convolutions.
    """
    print("🚀 Running 2D Non-Depthwise Variance Scaling Test...")
    B, H_in, H_out, X_in, Y_in = 32, 64, 64, 16, 16
    Kx, Ky = 7, 7

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: (B={B}, H_in={H_in}, X={X_in}, Y={Y_in})")

    # Unit variance inputs
    input_tensor = torch.randn(B, H_in, X_in, Y_in, dtype=dtype, device=device)
    # Unit variance kernel (NOT properly initialized - demonstrates the problem)
    kernel = torch.randn(H_out, H_in, Kx, Ky, dtype=dtype, device=device)

    input_var = input_tensor.var().item()
    print(f"Input variance: {input_var:.4f}")

    # FFT convolution (no normalization - matches F.conv2d)
    fft_output = fftconv2d_bhl(input_tensor, kernel, is_depthwise=False)
    fft_var = fft_output.var().item()

    # Without proper initialization, variance scales with H_in * Kx * Ky
    expected_var_unscaled = H_in * Kx * Ky * input_var

    print(f"FFT output variance: {fft_var:.4f}")
    print(f"Expected variance (~ H_in * Kx * Ky * input_var): {expected_var_unscaled:.4f}")
    print(f"Ratio (actual/expected): {fft_var / expected_var_unscaled:.4f}")

    # The variance should scale with H_in * Kx * Ky
    ratio = fft_var / expected_var_unscaled
    is_reasonable = 0.5 < ratio < 2.0

    if is_reasonable:
        print("\n✅ Test PASSED: Variance scales as expected (~ H_in * Kx * Ky).")
        print("   → This confirms proper initialization (scale 1/sqrt(H_in)) is needed!")
    else:
        print("\n❌ Test FAILED: Variance scaling is unexpected.")

    print("-" * 50)
    return is_reasonable


def test_fftconv2d_non_depthwise_einsum_correctness():
    """Tests that non-depthwise convolution einsum matches F.conv2d exactly (without normalization).

    This test verifies the mathematical correctness of the einsum operation,
    separate from the variance-preserving normalization.
    """
    print("🚀 Running 2D Non-Depthwise Einsum Correctness Test...")
    # Setup test parameters
    B, H_in, H_out, X_in, Y_in = 2, 4, 8, 16, 16
    Kx, Ky = 7, 7

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: (B={B}, H_in={H_in}, X={X_in}, Y={Y_in})")
    print(f"Kernel shape: (H_out={H_out}, H_in={H_in}, Kx={Kx}, Ky={Ky})")

    # Create random input and kernel tensors
    input_tensor = torch.randn(B, H_in, X_in, Y_in, dtype=dtype, device=device)
    kernel = torch.randn(H_out, H_in, Kx, Ky, dtype=dtype, device=device)

    # --- Reference: F.conv2d ---
    kernel_flipped = torch.flip(kernel, dims=[-1, -2])
    ref_output = F.conv2d(input_tensor, kernel_flipped, groups=1, padding="same")

    # --- Manual FFT convolution (WITHOUT normalization to verify einsum) ---
    fft_shape = (
        min(X_in + (Kx + 1) // 2, 2 * X_in),
        min(Y_in + (Ky + 1) // 2, 2 * Y_in),
    )

    fft_x = torch.fft.rfft2(input_tensor, s=fft_shape, dim=(2, 3))
    fft_kernel = torch.fft.rfft2(kernel, s=fft_shape, dim=(2, 3))

    # Non-depthwise: einsum over input channels (NO normalization for this test)
    fft_result = torch.einsum("b i x y, o i x y -> b o x y", fft_x, fft_kernel)

    crop_start_x = Kx // 2
    crop_start_y = Ky // 2

    fft_output = torch.fft.irfft2(fft_result, s=fft_shape, dim=(2, 3))[
        ..., crop_start_x : crop_start_x + X_in, crop_start_y : crop_start_y + Y_in
    ]

    # --- Comparison ---
    assert ref_output.shape == fft_output.shape, f"Shape mismatch! {ref_output.shape} vs {fft_output.shape}"

    max_abs_diff = torch.max(torch.abs(ref_output - fft_output)).item()
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Reference: max {ref_output.abs().max():.2e}, mean {ref_output.abs().mean():.2e}")
    relative_diff = max_abs_diff / ref_output.abs().max()
    print(f"Relative difference: {relative_diff:.2e}")

    is_close = relative_diff < 1e-4
    if is_close:
        print("\n✅ Test PASSED: Einsum-based FFT convolution matches F.conv2d exactly.")
    else:
        print("\n❌ Test FAILED: Outputs do not match.")
        print(f"Sample ref[0,0,0,:5]: {ref_output[0, 0, 0, :5]}")
        print(f"Sample fft[0,0,0,:5]: {fft_output[0, 0, 0, :5]}")

    print("-" * 50)
    return is_close


def test_fftconv2d_bhl_w_reshape_non_depthwise_variance():
    """Tests that fftconv2d_bhl_w_reshape with is_depthwise=False properly scales by 1/sqrt(H_in).

    The variance-preserving scaling is applied in fftconv2d_bhl_w_reshape (BLH wrapper)
    and CKConvND.apply_convolution (BHL direct path).
    """
    from nvsubquadratic.ops.fftconv import fftconv2d_bhl_w_reshape

    print("🚀 Running 2D Non-Depthwise Variance Preservation Test (via fftconv2d_bhl_w_reshape)...")
    B, H_in, H_out, X_in, Y_in = 32, 64, 64, 16, 16
    Kx, Ky = 7, 7

    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, dtype: {dtype}")
    print(f"Input shape: (B={B}, X={X_in}, Y={Y_in}, H_in={H_in}) [BLH format]")

    # Unit variance inputs (BLH format: batch, height, width, hidden)
    input_tensor = torch.randn(B, X_in, Y_in, H_in, dtype=dtype, device=device)
    # Kernel in BLH format: (1, Kx, Ky, c_out*c_in)
    kernel = torch.randn(1, Kx, Ky, H_out * H_in, dtype=dtype, device=device)

    input_var = input_tensor.var().item()
    print(f"Input variance: {input_var:.4f}")

    # FFT convolution via wrapper (which includes 1/sqrt(H_in) scaling)
    fft_output = fftconv2d_bhl_w_reshape(input_tensor, kernel, is_depthwise=False)
    fft_var = fft_output.var().item()

    # With 1/sqrt(H_in) scaling, variance should scale with Kx * Ky (not H_in * Kx * Ky)
    expected_var_with_scaling = Kx * Ky * input_var

    print(f"FFT output variance: {fft_var:.4f}")
    print(f"Expected variance (~ Kx * Ky * input_var): {expected_var_with_scaling:.4f}")
    print(f"Ratio (actual/expected): {fft_var / expected_var_with_scaling:.4f}")

    # The variance should be roughly proportional to Kx * Ky (not H_in * Kx * Ky)
    ratio = fft_var / expected_var_with_scaling
    is_reasonable = 0.5 < ratio < 2.0

    if is_reasonable:
        print("\n✅ Test PASSED: Variance is preserved (scales with Kx*Ky, not H_in*Kx*Ky).")
    else:
        print("\n❌ Test FAILED: Variance scaling is unexpected.")

    print("-" * 50)
    return is_reasonable


if __name__ == "__main__":
    print("=" * 60)
    print("DEPTHWISE CONVOLUTION TESTS")
    print("=" * 60 + "\n")
    test_causal_fftconv1d()
    test_fftconv1d()
    test_fftconv2d()
    test_fftconv3d()

    print("\n" + "=" * 60)
    print("NON-DEPTHWISE CONVOLUTION TESTS")
    print("=" * 60 + "\n")
    test_fftconv2d_non_depthwise_einsum_correctness()
    test_fftconv2d_non_depthwise()
    test_fftconv2d_non_depthwise_variance_scaling()
    test_fftconv2d_bhl_w_reshape_non_depthwise_variance()
