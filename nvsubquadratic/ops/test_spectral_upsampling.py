"""Test spectral upsampling as inverse of spectral downsampling.

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/test_spectral_upsampling.py
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND
from nvsubquadratic.ops.spectral_masking import (
    spectral_downsampling2d_bhl,
    spectral_upsampling2d_bhl,
)


def get_cameraman_image(size: int | None = None) -> np.ndarray:
    """Get the classic cameraman test image, optionally resized.

    Args:
        size: Target size for resizing. If None, returns original resolution (512x512).

    Returns a numpy array of shape (size, size) with values in [0, 1].
    """
    try:
        from skimage import data

        img_np = data.camera()  # Classic cameraman image (512x512)
        if size is not None and size != img_np.shape[0]:
            img = Image.fromarray(img_np, mode="L")
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            img_np = np.array(img)
        return img_np.astype(np.float32) / 255.0
    except ImportError:
        pass

    try:
        from scipy.datasets import face

        # Use scipy's raccoon face image and convert to grayscale
        img_np = face(gray=True).astype(np.float32)
        # Crop to square and resize
        h, w = img_np.shape
        min_size = min(h, w)
        start_h, start_w = (h - min_size) // 2, (w - min_size) // 2
        img_np = img_np[start_h : start_h + min_size, start_w : start_w + min_size]
        target_size = size if size is not None else 512
        img = Image.fromarray(img_np.astype(np.uint8), mode="L")
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        pass

    # Fallback: generate a test pattern with gradients and edges
    print("scipy/skimage not available, generating test pattern...")
    target_size = size if size is not None else 512
    x = np.linspace(0, 1, target_size)
    y = np.linspace(0, 1, target_size)
    xx, yy = np.meshgrid(x, y)

    # Combine different patterns
    pattern = (
        0.3 * xx  # horizontal gradient
        + 0.3 * yy  # vertical gradient
        + 0.2 * np.sin(xx * 20 * np.pi)  # horizontal stripes
        + 0.2 * np.sin(yy * 20 * np.pi)  # vertical stripes
    )
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    return pattern.astype(np.float32)


def create_test_image(size: int | None = None, device: str = "cuda") -> torch.Tensor:
    """Create a test image using the cameraman image.

    Args:
        size: Target size. If None, uses original resolution (512x512).
        device: Device to place tensor on.

    Returns a (1, 1, size, size) tensor with values in [0, 1].
    """
    img_np = get_cameraman_image(size)
    return torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, size, size)


def test_upsampling_basic():
    """Test basic upsampling without mask (ideal sinc interpolation)."""
    print("\n" + "=" * 60)
    print("Test 1: Basic upsampling without mask (sinc interpolation)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a small image (128x128, will upsample to 512x512)
    x_small = create_test_image(size=128, device=device)
    print(f"Input shape: {x_small.shape}")

    # Upsample to original cameraman resolution
    target_shape = (512, 512)
    x_up = spectral_upsampling2d_bhl(x_small, target_shape=target_shape)
    print(f"Upsampled shape: {x_up.shape}")

    assert x_up.shape == (1, 1, 512, 512), f"Expected (1, 1, 512, 512), got {x_up.shape}"
    print("✓ Shape check passed")

    # Check that values are finite
    assert torch.isfinite(x_up).all(), "Output contains non-finite values"
    print("✓ Finite values check passed")

    return x_small, x_up


def test_downsampling_upsampling_roundtrip():
    """Test that downsample -> upsample approximately recovers the input."""
    print("\n" + "=" * 60)
    print("Test 2: Downsampling -> Upsampling roundtrip")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a test image at original cameraman resolution (512x512)
    x_orig = create_test_image(size=None, device=device)  # Original 512x512
    print(f"Original shape: {x_orig.shape}")

    # Create spectral mask for downsampling with stride=2
    stride = 2.0
    mask_module = SpectralGaussianMaskND(
        data_dim=2,
        clip_value=0.1,
        init_stride_value=stride,
        min_stride_value=1.0,
    ).to(device)

    # Get the mask for downsampling
    spatial_dims = (512, 512)
    spectral_mask = mask_module(spatial_dims=spatial_dims)
    print(f"Spectral mask shape (from module): {spectral_mask.shape}")
    # Mask is [1, sM_x, sM_y, 1] (BLH format)
    # We need [1, C, sM_x, sM_y] (BHL format) for spectral_downsampling2d_bhl
    # Rearrange and expand for C=1 channel
    from einops import rearrange

    spectral_mask_bhl = rearrange(spectral_mask, "b x y c -> b c x y")  # [1, 1, sM_x, sM_y]
    print(f"Spectral mask shape (BHL): {spectral_mask_bhl.shape}")

    # Downsample
    x_down = spectral_downsampling2d_bhl(x_orig, spectral_mask_bhl)
    print(f"Downsampled shape: {x_down.shape}")

    # Upsample back to original size (without mask for now)
    x_recovered = spectral_upsampling2d_bhl(x_down, target_shape=(512, 512))
    print(f"Recovered shape: {x_recovered.shape}")

    # Compute reconstruction error
    # Note: Perfect recovery is not expected due to information loss in downsampling
    # (the Gaussian mask attenuates high frequencies, so they're lost)
    mse = torch.mean((x_orig - x_recovered) ** 2).item()
    rel_error = mse / torch.mean(x_orig**2).item()
    print(f"MSE: {mse:.6f}")
    print(f"Relative error: {rel_error:.4%}")

    # The relative error should be reasonable (not perfect, but not too bad)
    # With stride=2 and soft mask, we expect some loss but reasonable recovery
    assert rel_error < 0.5, f"Relative error too high: {rel_error:.4%}"
    print("✓ Reconstruction error within acceptable bounds")

    return x_orig, x_down, x_recovered, spectral_mask_bhl


def test_identity_with_stride_1():
    """Test that stride=1 (no downsampling) gives near-identity."""
    print("\n" + "=" * 60)
    print("Test 3: Identity with stride=1 (no downsampling)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_orig = create_test_image(size=None, device=device)  # Original 512x512
    print(f"Original shape: {x_orig.shape}")

    # Create spectral mask with stride=1 (no downsampling)
    mask_module = SpectralGaussianMaskND(
        data_dim=2,
        clip_value=0.8,
        init_stride_value=1.0,
        min_stride_value=1.0,
    ).to(device)

    # Get the mask
    spectral_mask = mask_module(spatial_dims=(512, 512))
    from einops import rearrange

    spectral_mask_bhl = rearrange(spectral_mask, "b x y c -> b c x y")
    print(f"Spectral mask shape (BHL): {spectral_mask_bhl.shape}")

    # Downsample with stride=1 should give same or very similar size
    x_down = spectral_downsampling2d_bhl(x_orig, spectral_mask_bhl)
    print(f"Downsampled shape: {x_down.shape}")

    # With stride=1, the mask should cover most frequencies,
    # so downsampling should be near-identity (just slight attenuation at edges)
    # Note: With higher clip_value (e.g., 0.5), output may be slightly smaller
    # because the Gaussian mask reaches the threshold sooner
    orig_H, orig_W = x_orig.shape[2], x_orig.shape[3]
    down_H, down_W = x_down.shape[2], x_down.shape[3]

    if orig_H == down_H and orig_W == down_W:
        # Same size - compare directly
        mse = torch.mean((x_orig - x_down) ** 2).item()
        rel_error = mse / torch.mean(x_orig**2).item()
    else:
        # Different size - compare overlapping center region
        # Center crop both to the smaller size
        min_H, min_W = min(orig_H, down_H), min(orig_W, down_W)
        start_orig_H, start_orig_W = (orig_H - min_H) // 2, (orig_W - min_W) // 2
        start_down_H, start_down_W = (down_H - min_H) // 2, (down_W - min_W) // 2

        x_orig_crop = x_orig[:, :, start_orig_H : start_orig_H + min_H, start_orig_W : start_orig_W + min_W]
        x_down_crop = x_down[:, :, start_down_H : start_down_H + min_H, start_down_W : start_down_W + min_W]

        mse = torch.mean((x_orig_crop - x_down_crop) ** 2).item()
        rel_error = mse / torch.mean(x_orig_crop**2).item()
        print(f"  (Comparing center {min_H}x{min_W} region due to size mismatch)")

    print(f"MSE: {mse:.6f}")
    print(f"Relative error: {rel_error:.4%}")

    # With stride=1, we expect very good preservation
    assert rel_error < 0.1, f"Relative error too high for stride=1: {rel_error:.4%}"
    print("✓ Near-identity check passed")

    return x_orig, x_down


def test_upsampling_with_mask():
    """Test upsampling with a spectral mask (smooth interpolation)."""
    print("\n" + "=" * 60)
    print("Test 4: Upsampling with spectral mask")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a small image (128x128, will upsample to 512x512)
    x_small = create_test_image(size=128, device=device)
    print(f"Input shape: {x_small.shape}")

    # Create spectral mask for the small input
    mask_module = SpectralGaussianMaskND(
        data_dim=2,
        clip_value=0.1,
        init_stride_value=1.0,  # stride=1 for the input size
        min_stride_value=1.0,
    ).to(device)

    spectral_mask = mask_module(spatial_dims=(128, 128))
    from einops import rearrange

    spectral_mask_bhl = rearrange(spectral_mask, "b x y c -> b c x y")
    print(f"Spectral mask shape (BHL): {spectral_mask_bhl.shape}")

    # Upsample with mask
    target_shape = (512, 512)
    x_up_masked = spectral_upsampling2d_bhl(x_small, target_shape=target_shape, spectral_mask=spectral_mask_bhl)
    print(f"Upsampled (with mask) shape: {x_up_masked.shape}")

    # Upsample without mask for comparison
    x_up_no_mask = spectral_upsampling2d_bhl(x_small, target_shape=target_shape, spectral_mask=None)
    print(f"Upsampled (no mask) shape: {x_up_no_mask.shape}")

    # Check shapes
    assert x_up_masked.shape == (1, 1, 512, 512)
    assert x_up_no_mask.shape == (1, 1, 512, 512)
    print("✓ Shape checks passed")

    # Check that values are finite
    assert torch.isfinite(x_up_masked).all(), "Masked output contains non-finite values"
    assert torch.isfinite(x_up_no_mask).all(), "Unmasked output contains non-finite values"
    print("✓ Finite values checks passed")

    return x_small, x_up_masked, x_up_no_mask


def visualize_results():
    """Run all tests and visualize results."""
    print("\n" + "=" * 60)
    print("Running all tests with visualization")
    print("=" * 60)

    # Run tests
    x_small, x_up = test_upsampling_basic()
    x_orig, x_down, x_recovered, mask = test_downsampling_upsampling_roundtrip()
    x_orig_id, x_down_id = test_identity_with_stride_1()
    x_small2, x_up_masked, x_up_no_mask = test_upsampling_with_mask()

    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Basic upsampling
    axes[0, 0].imshow(x_small[0, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Small input ({x_small.shape[2]}x{x_small.shape[3]})")
    axes[0, 1].imshow(x_up[0, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Upsampled ({x_up.shape[2]}x{x_up.shape[3]})")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    # Row 2: Roundtrip
    axes[1, 0].imshow(x_orig[0, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Original ({x_orig.shape[2]}x{x_orig.shape[3]})")
    axes[1, 1].imshow(x_down[0, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Downsampled ({x_down.shape[2]}x{x_down.shape[3]})")
    axes[1, 2].imshow(x_recovered[0, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title(f"Recovered ({x_recovered.shape[2]}x{x_recovered.shape[3]})")
    diff = (x_orig - x_recovered)[0, 0].detach().cpu().numpy()
    axes[1, 3].imshow(diff, cmap="RdBu", vmin=-0.2, vmax=0.2)
    axes[1, 3].set_title("Difference")

    # Row 3: Upsampling with/without mask
    axes[2, 0].imshow(x_small2[0, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2, 0].set_title(f"Small input ({x_small2.shape[2]}x{x_small2.shape[3]})")
    axes[2, 1].imshow(x_up_no_mask[0, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2, 1].set_title(f"Upsampled no mask ({x_up_no_mask.shape[2]}x{x_up_no_mask.shape[3]})")
    axes[2, 2].imshow(x_up_masked[0, 0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2, 2].set_title(f"Upsampled with mask ({x_up_masked.shape[2]}x{x_up_masked.shape[3]})")
    diff2 = (x_up_no_mask - x_up_masked)[0, 0].detach().cpu().numpy()
    axes[2, 3].imshow(diff2, cmap="RdBu", vmin=-0.1, vmax=0.1)
    axes[2, 3].set_title("Difference (no mask - with mask)")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("nvsubquadratic/ops/test_spectral_upsampling.png", dpi=150)
    print("\n✓ Visualization saved to nvsubquadratic/ops/test_spectral_upsampling.png")
    plt.close()


def main():
    """Run all tests."""
    torch.manual_seed(42)

    # Run individual tests
    test_upsampling_basic()
    test_downsampling_upsampling_roundtrip()
    test_identity_with_stride_1()
    test_upsampling_with_mask()

    # Visualize
    visualize_results()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
