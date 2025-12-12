"""Test to verify that spectral_downsampling2d_bhl doesn't have circular convolution artifacts.

The issue: Without zero-padding, the FFT treats the signal as periodic, causing content
from one border to smear into the opposite border during low-pass filtering.

The test: Create an image with a bright region on one side and black on the other.
After downsampling, the black region should remain black (no smearing from the bright side).
"""

import matplotlib.pyplot as plt
import torch

from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND
from nvsubquadratic.ops.spectral_masking import spectral_downsampling2d_bhl


def test_no_border_smearing():
    """Test that the borders don't smear into each other during spectral downsampling."""
    print("\n" + "=" * 80)
    print("Test: Verifying NO circular convolution artifacts (border smearing)")
    print("=" * 80)

    # Create a test image: bright on the left half, black on the right half
    H, W = 64, 64
    x = torch.zeros(1, 1, H, W)
    x[:, :, :, : W // 2] = 1.0  # Left half is bright (1.0)
    # Right half remains 0.0

    print("\nInput image:")
    print(f"  Shape: {x.shape}")
    print(f"  Left half (cols 0-{W // 2 - 1}): mean = {x[:, :, :, : W // 2].mean():.4f}")
    print(f"  Right half (cols {W // 2}-{W - 1}): mean = {x[:, :, :, W // 2 :].mean():.4f}")

    # Create a spectral mask for 2x downsampling
    stride = 2.0
    target_H, target_W = int(H / stride), int(W / stride)

    # Create a simple Gaussian mask manually for testing
    mask_module = SpectralGaussianMaskND(
        data_dim=2,
        clip_value=0.1,
        init_stride_value=stride,
    )
    spectral_mask = mask_module((H, W)).unsqueeze(1)  # Shape: (1, 1, sM_x, sM_y)
    print(f"\nSpectral mask shape: {spectral_mask.shape}")
    print(f"  Expected target output: ({target_H}, {target_W})")

    # Apply spectral downsampling
    y = spectral_downsampling2d_bhl(x, spectral_mask)

    print("\nOutput image:")
    print(f"  Shape: {y.shape}")
    print(f"  Left half (cols 0-{target_W // 2 - 1}): mean = {y[:, :, :, : target_W // 2].mean():.4f}")
    print(f"  Right half (cols {target_W // 2}-{target_W - 1}): mean = {y[:, :, :, target_W // 2 :].mean():.4f}")

    # Check for border smearing
    # The right half should be very close to 0 (no smearing from the left)
    # Allow some tolerance for the Gaussian rolloff at the boundary
    right_half_max = y[:, :, :, target_W // 2 :].abs().max().item()
    right_border_max = y[:, :, :, -2:].abs().max().item()  # Just the rightmost 2 columns

    print("\nBorder analysis:")
    print(f"  Max value in right half: {right_half_max:.6f}")
    print(f"  Max value in rightmost 2 columns: {right_border_max:.6f}")

    # The key test: the rightmost columns should not have significant values
    # With circular convolution, the bright left edge would wrap around to the right edge
    if right_border_max < 0.01:
        print("\n✅ PASS: No significant border smearing detected!")
    else:
        print(f"\n⚠️  WARNING: Possible border smearing detected (right border max = {right_border_max:.4f})")

    # Create a visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(x[0, 0].numpy(), cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title(f"Input ({H}x{W})")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")

    axes[1].imshow(y[0, 0].detach().numpy(), cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title(f"Output ({target_H}x{target_W})")
    axes[1].set_xlabel("Width")
    axes[1].set_ylabel("Height")

    # Show a horizontal slice through the middle
    mid_row = target_H // 2
    axes[2].plot(y[0, 0, mid_row].detach().numpy(), label="Output")
    axes[2].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Value")
    axes[2].set_title(f"Horizontal slice (row {mid_row})")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(
        "/home/david.romero/projects/nvSubquadratic-private/nvsubquadratic/ops/circular_convolution_fix_test.png",
        dpi=150,
    )
    print("\n✅ Visualization saved to: circular_convolution_fix_test.png")

    return right_border_max < 0.01


def test_top_bottom_border():
    """Test vertical border smearing as well."""
    print("\n" + "=" * 80)
    print("Test: Verifying NO vertical border smearing")
    print("=" * 80)

    # Create a test image: bright on the top half, black on the bottom half
    H, W = 64, 64
    x = torch.zeros(1, 1, H, W)
    x[:, :, : H // 2, :] = 1.0  # Top half is bright

    print("\nInput image:")
    print(f"  Top half (rows 0-{H // 2 - 1}): mean = {x[:, :, : H // 2, :].mean():.4f}")
    print(f"  Bottom half (rows {H // 2}-{H - 1}): mean = {x[:, :, H // 2 :, :].mean():.4f}")

    stride = 2.0
    target_H, target_W = int(H / stride), int(W / stride)

    mask_module = SpectralGaussianMaskND(
        data_dim=2,
        clip_value=0.1,
        init_stride_value=stride,
    )
    spectral_mask = mask_module((H, W)).unsqueeze(1)

    y = spectral_downsampling2d_bhl(x, spectral_mask)

    print("\nOutput image:")
    print(f"  Top half mean: {y[:, :, : target_H // 2, :].mean():.4f}")
    print(f"  Bottom half mean: {y[:, :, target_W // 2 :, :].mean():.4f}")

    bottom_border_max = y[:, :, -2:, :].abs().max().item()
    print(f"  Max value in bottom 2 rows: {bottom_border_max:.6f}")

    if bottom_border_max < 0.01:
        print("\n✅ PASS: No significant vertical border smearing!")
    else:
        print(f"\n⚠️  WARNING: Possible vertical border smearing (max = {bottom_border_max:.4f})")

    return bottom_border_max < 0.01


if __name__ == "__main__":
    test1_passed = test_no_border_smearing()
    test2_passed = test_top_bottom_border()

    print("\n" + "=" * 80)
    if test1_passed and test2_passed:
        print("✅ All tests PASSED!")
    else:
        print("⚠️  Some tests did not pass - check output above")
    print("=" * 80)
