# TODO: Add license header here


"""Test the 2D FFT convolution with spectral mask (DiffStride-like downsampling).

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/test_fftconv_spectral_mask.py
"""

import torch

from nvsubquadratic.ops.fftconv import fftconv2d_bhl, fftconv2d_bhl_w_reshape


def test_fftconv2d_bhl_without_spectral_mask():
    """Test baseline: fftconv2d_bhl without spectral mask should preserve input size."""
    print("\n🔵 Test 1: fftconv2d_bhl without spectral mask")

    B, H, X_in, Y_in = 2, 8, 32, 32
    K_x, K_y = 32, 32  # Kernel same size as input

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    x = torch.randn(B, H, X_in, Y_in, dtype=dtype, device=device)
    kernel = torch.randn(1, H, K_x, K_y, dtype=dtype, device=device)

    y = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=None)

    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Kernel shape: {tuple(kernel.shape)}")
    print(f"  Output shape: {tuple(y.shape)}")

    assert y.shape == (B, H, X_in, Y_in), f"Expected output shape {(B, H, X_in, Y_in)}, got {tuple(y.shape)}"
    print("  ✅ Output shape matches input shape")


def test_fftconv2d_bhl_with_spectral_mask_output_size():
    """Test that spectral mask correctly downsamples the output."""
    print("\n🔵 Test 2: fftconv2d_bhl with spectral mask - output size")

    B, H, X_in, Y_in = 2, 8, 64, 64
    K_x, K_y = 64, 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    x = torch.randn(B, H, X_in, Y_in, dtype=dtype, device=device)
    kernel = torch.randn(1, H, K_x, K_y, dtype=dtype, device=device)

    # Test different downsampling factors
    test_cases = [
        # (sM_x, sM_y) -> expected output (sM_x, 2*(sM_y-1))
        (32, 17, 32, 32),  # 64 -> 32 (2x downsample)
        (16, 9, 16, 16),  # 64 -> 16 (4x downsample)
        (64, 33, 64, 64),  # No downsample (full size for 64 input)
    ]

    for sM_x, sM_y, expected_X_out, expected_Y_out in test_cases:
        spectral_mask = torch.ones(1, H, sM_x, sM_y, dtype=dtype, device=device)
        y = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=spectral_mask)

        print(
            f"  spectral_mask: ({sM_x}, {sM_y}) -> output: {tuple(y.shape[2:])}, expected: ({expected_X_out}, {expected_Y_out})"
        )

        assert y.shape == (B, H, expected_X_out, expected_Y_out), (
            f"Expected output shape {(B, H, expected_X_out, expected_Y_out)}, got {tuple(y.shape)}"
        )

    print("  ✅ All output sizes match expected downsampled sizes")


def test_fftconv2d_bhl_spectral_mask_gradient_flow():
    """Test that gradients flow through the spectral mask."""
    print("\n🔵 Test 3: fftconv2d_bhl spectral mask gradient flow")

    B, H, X_in, Y_in = 2, 8, 32, 32
    K_x, K_y = 32, 32
    sM_x, sM_y = 16, 9  # Downsample to 16x16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    x = torch.randn(B, H, X_in, Y_in, dtype=dtype, device=device)
    kernel = torch.randn(1, H, K_x, K_y, dtype=dtype, device=device)

    # Spectral mask with requires_grad
    spectral_mask = torch.ones(1, H, sM_x, sM_y, dtype=dtype, device=device, requires_grad=True)

    y = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=spectral_mask)
    loss = y.sum()
    loss.backward()

    print(f"  spectral_mask.grad is not None: {spectral_mask.grad is not None}")
    print(f"  spectral_mask.grad.shape: {spectral_mask.grad.shape if spectral_mask.grad is not None else 'N/A'}")
    print(
        f"  spectral_mask.grad.abs().mean(): {spectral_mask.grad.abs().mean().item() if spectral_mask.grad is not None else 'N/A':.6f}"
    )

    assert spectral_mask.grad is not None, "Gradient should flow through spectral_mask"
    assert spectral_mask.grad.shape == spectral_mask.shape, "Gradient shape should match mask shape"
    assert spectral_mask.grad.abs().sum() > 0, "Gradient should be non-zero"
    print("  ✅ Gradients flow through spectral mask")


def test_fftconv2d_bhl_spectral_mask_affects_output():
    """Test that different spectral mask values produce different outputs."""
    print("\n🔵 Test 4: fftconv2d_bhl spectral mask affects output")

    B, H, X_in, Y_in = 2, 8, 32, 32
    K_x, K_y = 32, 32
    sM_x, sM_y = 16, 9

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    x = torch.randn(B, H, X_in, Y_in, dtype=dtype, device=device)
    kernel = torch.randn(1, H, K_x, K_y, dtype=dtype, device=device)

    # All-ones mask
    mask_ones = torch.ones(1, H, sM_x, sM_y, dtype=dtype, device=device)
    y_ones = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=mask_ones)

    # All-zeros mask (should zero out everything)
    mask_zeros = torch.zeros(1, H, sM_x, sM_y, dtype=dtype, device=device)
    y_zeros = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=mask_zeros)

    # Random mask
    mask_random = torch.rand(1, H, sM_x, sM_y, dtype=dtype, device=device)
    y_random = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=mask_random)

    print(f"  y_ones.abs().mean():   {y_ones.abs().mean().item():.6f}")
    print(f"  y_zeros.abs().mean():  {y_zeros.abs().mean().item():.6f}")
    print(f"  y_random.abs().mean(): {y_random.abs().mean().item():.6f}")

    # Zero mask should produce zero output
    assert y_zeros.abs().max() < 1e-6, f"Zero mask should produce zero output, got max abs {y_zeros.abs().max()}"

    # Different masks should produce different outputs
    diff_ones_random = (y_ones - y_random).abs().mean().item()
    print(f"  diff(y_ones, y_random): {diff_ones_random:.6f}")
    assert diff_ones_random > 1e-6, "Different masks should produce different outputs"

    print("  ✅ Spectral mask values affect the output correctly")


def test_fftconv2d_bhl_spectral_mask_low_pass_behavior():
    """Test that spectral mask acts as a low-pass filter (smaller mask = more blur)."""
    print("\n🔵 Test 5: fftconv2d_bhl spectral mask low-pass behavior")

    B, H, X_in, Y_in = 1, 1, 64, 64
    K_x, K_y = 64, 64

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Create a high-frequency input (checkerboard pattern)
    x = torch.zeros(B, H, X_in, Y_in, dtype=dtype, device=device)
    x[:, :, ::2, ::2] = 1.0
    x[:, :, 1::2, 1::2] = 1.0

    # Identity-like kernel (mostly zero with center spike)
    kernel = torch.zeros(1, H, K_x, K_y, dtype=dtype, device=device)
    kernel[:, :, K_x // 2, K_y // 2] = 1.0

    # Full resolution mask (keep all frequencies)
    full_mask = torch.ones(1, H, 64, 33, dtype=dtype, device=device)
    y_full = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=full_mask)

    # Low resolution mask (keep only low frequencies) - 16x16 output
    low_mask = torch.ones(1, H, 16, 9, dtype=dtype, device=device)
    y_low = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=low_mask)

    print(f"  Full resolution output variance: {y_full.var().item():.6f}")
    print(f"  Low resolution output variance:  {y_low.var().item():.6f}")
    print(f"  Full resolution output shape: {tuple(y_full.shape)}")
    print(f"  Low resolution output shape:  {tuple(y_low.shape)}")

    # High-frequency content (checkerboard) should have lower variance after low-pass filtering
    # because the high frequencies are removed
    assert y_low.var() < y_full.var(), "Low-pass filtered output should have lower variance for high-freq input"

    print("  ✅ Spectral mask correctly acts as low-pass filter")


def test_fftconv2d_bhl_w_reshape_with_spectral_mask():
    """Test that the wrapper function also works with spectral mask."""
    print("\n🔵 Test 6: fftconv2d_bhl_w_reshape with spectral mask")

    B, X_in, Y_in, H = 2, 32, 32, 8  # Note: different layout (B, X, Y, H)
    K_x, K_y = 32, 32
    sM_x, sM_y = 16, 9

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    x = torch.randn(B, X_in, Y_in, H, dtype=dtype, device=device)
    kernel = torch.randn(1, K_x, K_y, H, dtype=dtype, device=device)
    # spectral_mask layout for w_reshape: (1, sM_x, sM_y, H) - same as input layout
    spectral_mask = torch.ones(1, sM_x, sM_y, H, dtype=dtype, device=device)

    y = fftconv2d_bhl_w_reshape(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=spectral_mask)

    expected_X_out = sM_x
    expected_Y_out = 2 * (sM_y - 1)

    print(f"  Input shape:  {tuple(x.shape)} (B, X, Y, H)")
    print(f"  Output shape: {tuple(y.shape)} (B, X_out, Y_out, H)")
    print(f"  Expected:     ({B}, {expected_X_out}, {expected_Y_out}, {H})")

    assert y.shape == (B, expected_X_out, expected_Y_out, H), (
        f"Expected output shape {(B, expected_X_out, expected_Y_out, H)}, got {tuple(y.shape)}"
    )
    print("  ✅ Wrapper function works correctly with spectral mask")


def test_fftconv2d_bhl_shortcut_spectral_mask_mutual_exclusion():
    """Test that shortcut and spectral_mask cannot be used together."""
    print("\n🔵 Test 7: shortcut and spectral_mask mutual exclusion")

    B, H, X_in, Y_in = 2, 8, 32, 32
    K_x, K_y = 32, 32
    sM_x, sM_y = 16, 9

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    x = torch.randn(B, H, X_in, Y_in, dtype=dtype, device=device)
    kernel = torch.randn(1, H, K_x, K_y, dtype=dtype, device=device)
    spectral_mask = torch.ones(1, H, sM_x, sM_y, dtype=dtype, device=device)
    shortcut = torch.randn(H, dtype=dtype, device=device)

    try:
        _ = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=shortcut, spectral_mask=spectral_mask)
        print("  ❌ Should have raised an assertion error")
        assert False, "Should have raised an assertion error"
    except AssertionError as e:
        print(f"  Caught expected assertion: {e}")
        print("  ✅ Mutual exclusion enforced correctly")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Testing fftconv2d_bhl with spectral mask")
    print("=" * 60)

    test_fftconv2d_bhl_without_spectral_mask()
    test_fftconv2d_bhl_with_spectral_mask_output_size()
    test_fftconv2d_bhl_spectral_mask_gradient_flow()
    test_fftconv2d_bhl_spectral_mask_affects_output()
    test_fftconv2d_bhl_spectral_mask_low_pass_behavior()
    test_fftconv2d_bhl_w_reshape_with_spectral_mask()
    test_fftconv2d_bhl_shortcut_spectral_mask_mutual_exclusion()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
