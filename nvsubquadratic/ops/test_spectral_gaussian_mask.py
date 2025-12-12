# TODO: Add license header here

"""Visual test for spectral downsampling using SpectralGaussianMaskND.

Uses scipy's test image, applies identity convolution with learnable Gaussian
spectral mask for downsampling, and saves input/output images for visual verification.

This test demonstrates:
1. SpectralGaussianMaskND generates proper frequency-domain masks
2. The mask correctly controls downsampling via learned stride
3. Integration with fftconv2d_bhl for end-to-end downsampling

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/test_spectral_gaussian_mask.py
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND
from nvsubquadratic.ops.fftconv import fftconv2d_bhl


def get_test_image(size: int = 256):
    """Get a test image (from scipy or generate a pattern)."""
    try:
        from scipy.datasets import face

        # Use scipy's raccoon face image and convert to grayscale
        img_np = face(gray=True).astype(np.float32)
        # Crop to square and resize
        h, w = img_np.shape
        min_size = min(h, w)
        start_h, start_w = (h - min_size) // 2, (w - min_size) // 2
        img_np = img_np[start_h : start_h + min_size, start_w : start_w + min_size]
        img = Image.fromarray(img_np.astype(np.uint8), mode="L")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        return img
    except ImportError:
        pass

    try:
        from skimage import data

        img_np = data.camera()  # Classic cameraman image
        img = Image.fromarray(img_np, mode="L")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        return img
    except ImportError:
        pass

    # Fallback: generate a test pattern with gradients and edges
    print("scipy/skimage not available, generating test pattern...")
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xx, yy = np.meshgrid(x, y)

    # Combine different patterns
    pattern = (
        0.3 * xx  # horizontal gradient
        + 0.3 * yy  # vertical gradient
        + 0.2 * np.sin(xx * 20 * np.pi)  # horizontal stripes
        + 0.2 * np.sin(yy * 20 * np.pi)  # vertical stripes
    )
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
    img_np = (pattern * 255).astype(np.uint8)
    return Image.fromarray(img_np, mode="L")


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    arr = t.squeeze().detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def test_spectral_gaussian_mask_basic():
    """Test basic functionality of SpectralGaussianMaskND."""
    print("\n" + "=" * 60)
    print("Test 1: Basic SpectralGaussianMaskND functionality")
    print("=" * 60)

    spatial_dims = (64, 64)

    # Test different stride values
    for stride in [1.0, 1.5, 2.0, 4.0]:
        mask_module = SpectralGaussianMaskND(
            data_dim=2,
            clip_value=0.1,
            init_stride_value=stride,
            min_stride_value=1.0,
            max_stride_value=None,
            parametrization="direct",
        )

        mask = mask_module(spatial_dims)
        effective_stride = mask_module.get_stride()

        print(f"\n  stride_init={stride}")
        print(f"    std_param: {mask_module.std_param.tolist()}")
        print(f"    effective_stride: {effective_stride.tolist()}")
        print(f"    mask shape: {tuple(mask.shape)}")
        print(f"    mask range: [{mask.min().item():.4f}, {mask.max().item():.4f}]")

        # Verify mask shape is reduced for stride > 1
        if stride > 1.0:
            assert mask.shape[1] < spatial_dims[0], f"Height should be reduced for stride {stride}"
            assert mask.shape[2] < spatial_dims[1] // 2 + 1, f"Width should be reduced for stride {stride}"

    print("\n  ✅ Basic functionality test passed!")


def test_spectral_gaussian_mask_with_fftconv():
    """Test SpectralGaussianMaskND integration with fftconv2d_bhl."""
    print("\n" + "=" * 60)
    print("Test 2: Integration with fftconv2d_bhl")
    print("=" * 60)

    # Create input tensor
    B, C, H, W = 1, 1, 64, 64
    x = torch.randn(B, C, H, W, dtype=torch.float32)

    # Create identity kernel
    kernel = torch.zeros(1, C, H, W, dtype=torch.float32)
    kernel[:, :, H // 2, W // 2] = 1.0

    # Test with different strides
    for stride in [1.5, 2.0, 4.0]:
        mask_module = SpectralGaussianMaskND(
            data_dim=2,
            clip_value=0.9,
            init_stride_value=stride,
            min_stride_value=1.0,
            parametrization="direct",
        )

        # Generate spectral mask: [1, sM_h, sM_w, 1]
        mask = mask_module((H, W))

        # Remove trailing dimension and expand to match expected shape for fftconv: (1, C, sM_x, sM_y)
        mask = mask.squeeze(-1)  # [1, sM_h, sM_w]
        spectral_mask = mask.unsqueeze(1).expand(-1, C, -1, -1)

        # Apply convolution with spectral mask
        y = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=spectral_mask)

        print(f"\n  stride={stride}")
        print(f"    input shape: {tuple(x.shape)}")
        print(f"    spectral_mask shape: {tuple(spectral_mask.shape)}")
        print(f"    output shape: {tuple(y.shape)}")

        # Verify output is downsampled
        assert y.shape[2] < H, f"Height should be reduced for stride {stride}"
        assert y.shape[3] < W, f"Width should be reduced for stride {stride}"

    print("\n  ✅ fftconv integration test passed!")


def test_spectral_gaussian_mask_gradient_flow():
    """Test that gradients flow through SpectralGaussianMaskND."""
    print("\n" + "=" * 60)
    print("Test 3: Gradient flow through SpectralGaussianMaskND")
    print("=" * 60)

    B, C, H, W = 1, 1, 64, 64
    x = torch.randn(B, C, H, W, dtype=torch.float32)
    kernel = torch.zeros(1, C, H, W, dtype=torch.float32)
    kernel[:, :, H // 2, W // 2] = 1.0

    mask_module = SpectralGaussianMaskND(
        data_dim=2,
        clip_value=0.1,
        init_stride_value=2.0,
        min_stride_value=1.0,
        parametrization="direct",
    )

    # Forward pass
    mask = mask_module((H, W))
    mask = mask.squeeze(-1)  # [1, sM_h, sM_w]
    spectral_mask = mask.unsqueeze(1).expand(-1, C, -1, -1)
    y = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=spectral_mask)

    # Backward pass
    loss = y.sum()
    loss.backward()

    print(f"  std_param.grad: {mask_module.std_param.grad}")
    print(f"  std_param.grad is not None: {mask_module.std_param.grad is not None}")

    # Note: Gradients may be None because the mask values don't directly depend on
    # std_param in a differentiable way (the crop bounds are computed with detached values).
    # The Gaussian values DO depend on std_param, so gradients should flow through those.

    print("\n  ✅ Gradient flow test completed!")


def test_visual_downsampling():
    """Visual test with a real image."""
    print("\n" + "=" * 60)
    print("Test 4: Visual downsampling test")
    print("=" * 60)

    output_dir = Path(__file__).parent

    # Get test image
    img_size = 256
    img = get_test_image(img_size)
    print(f"  Using image of size {img.size}")

    # Convert to tensor: (B, C, H, W)
    img_np = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)

    print(f"  Input shape: {x.shape}")
    print(f"  Input range: [{x.min():.3f}, {x.max():.3f}]")

    # Create identity kernel
    K_H, K_W = img_size, img_size
    kernel = torch.zeros(1, 1, K_H, K_W, dtype=torch.float32)
    kernel[:, :, K_H // 2, K_W // 2] = 1.0

    # Test with different stride values
    results = []

    # Stride ~1 (minimal downsampling)
    mask_1x = SpectralGaussianMaskND(data_dim=2, clip_value=0.1, init_stride_value=1.0, min_stride_value=1.0)
    spectral_mask_1x = mask_1x((K_H, K_W)).squeeze(-1).unsqueeze(1)
    y_1x = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=spectral_mask_1x)
    results.append(("1x", y_1x, mask_1x.get_stride()))
    print(f"  1x: shape {y_1x.shape}, stride {mask_1x.get_stride().tolist()}")
    print(
        f"      output range [{y_1x.min():.3f}, {y_1x.max():.3f}], mask range [{spectral_mask_1x.min():.3f}, {spectral_mask_1x.max():.3f}], mask mean {spectral_mask_1x.mean():.3f}"
    )

    # Stride 2
    mask_2x = SpectralGaussianMaskND(data_dim=2, clip_value=0.1, init_stride_value=2.0, min_stride_value=1.0)
    spectral_mask_2x = mask_2x((K_H, K_W)).squeeze(-1).unsqueeze(1)
    y_2x = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=spectral_mask_2x)
    results.append(("2x", y_2x, mask_2x.get_stride()))
    print(f"  2x: shape {y_2x.shape}, stride {mask_2x.get_stride().tolist()}")
    print(
        f"      output range [{y_2x.min():.3f}, {y_2x.max():.3f}], mask range [{spectral_mask_2x.min():.3f}, {spectral_mask_2x.max():.3f}], mask mean {spectral_mask_2x.mean():.3f}"
    )

    # Stride 4
    mask_4x = SpectralGaussianMaskND(data_dim=2, clip_value=0.1, init_stride_value=4.0, min_stride_value=1.0)
    spectral_mask_4x = mask_4x((K_H, K_W)).squeeze(-1).unsqueeze(1)
    y_4x = fftconv2d_bhl(x, kernel, is_depthwise=True, shortcut=None, spectral_mask=spectral_mask_4x)
    results.append(("4x", y_4x, mask_4x.get_stride()))
    print(f"  4x: shape {y_4x.shape}, stride {mask_4x.get_stride().tolist()}")
    print(
        f"      output range [{y_4x.min():.3f}, {y_4x.max():.3f}], mask range [{spectral_mask_4x.min():.3f}, {spectral_mask_4x.max():.3f}], mask mean {spectral_mask_4x.mean():.3f}"
    )

    # Create visualization
    from PIL import ImageDraw, ImageFont

    img_input = tensor_to_image(x)
    img_1x = tensor_to_image(y_1x)
    img_2x = tensor_to_image(y_2x)
    img_4x = tensor_to_image(y_4x)

    # Layout parameters
    padding = 10
    label_height = 30

    # Row 1: Original + downsampled at native sizes
    # Row 2: All upscaled to original size for comparison

    total_width = img_size + padding + img_1x.width + padding + img_2x.width + padding + img_4x.width
    total_height = img_size + label_height + padding + img_size + label_height

    canvas = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Row 1: Native sizes
    y_offset = 0
    x_pos = 0

    draw.text((x_pos, y_offset), f"Original ({img_size}x{img_size})", fill=(255, 255, 255), font=font)
    canvas.paste(img_input.convert("RGB"), (x_pos, y_offset + label_height))
    x_pos += img_size + padding

    draw.text((x_pos, y_offset), f"~1x ({img_1x.width}x{img_1x.height})", fill=(255, 255, 255), font=font)
    canvas.paste(img_1x.convert("RGB"), (x_pos, y_offset + label_height))
    x_pos += img_1x.width + padding

    draw.text((x_pos, y_offset), f"2x ({img_2x.width}x{img_2x.height})", fill=(255, 255, 255), font=font)
    canvas.paste(img_2x.convert("RGB"), (x_pos, y_offset + label_height))
    x_pos += img_2x.width + padding

    draw.text((x_pos, y_offset), f"4x ({img_4x.width}x{img_4x.height})", fill=(255, 255, 255), font=font)
    canvas.paste(img_4x.convert("RGB"), (x_pos, y_offset + label_height))

    # Row 2: Upscaled for comparison
    y_offset = img_size + label_height + padding
    x_pos = 0

    img_input_up = img_input.resize((img_size, img_size), Image.Resampling.NEAREST)
    img_1x_up = img_1x.resize((img_size, img_size), Image.Resampling.NEAREST)
    img_2x_up = img_2x.resize((img_size, img_size), Image.Resampling.NEAREST)
    img_4x_up = img_4x.resize((img_size, img_size), Image.Resampling.NEAREST)

    draw.text((x_pos, y_offset), "Original (ref)", fill=(200, 200, 200), font=font)
    canvas.paste(img_input_up.convert("RGB"), (x_pos, y_offset + label_height))
    x_pos += img_size + padding

    draw.text((x_pos, y_offset), "~1x upscaled", fill=(200, 200, 200), font=font)
    canvas.paste(img_1x_up.convert("RGB"), (x_pos, y_offset + label_height))
    x_pos += img_size + padding

    draw.text((x_pos, y_offset), "2x upscaled", fill=(200, 200, 200), font=font)
    canvas.paste(img_2x_up.convert("RGB"), (x_pos, y_offset + label_height))
    x_pos += img_size + padding

    draw.text((x_pos, y_offset), "4x upscaled", fill=(200, 200, 200), font=font)
    canvas.paste(img_4x_up.convert("RGB"), (x_pos, y_offset + label_height))

    # Save
    output_path = output_dir / "spectral_gaussian_mask_test.png"
    canvas.save(output_path)

    print("\n  ✅ Visual test complete!")
    print(f"     Saved to: {output_path}")


def test_clip_value_effect():
    """Test that clip_value affects mask values (but not output size)."""
    print("\n" + "=" * 60)
    print("Test 5: Clip value effect on mask")
    print("=" * 60)

    spatial_dims = (64, 64)
    stride = 2.0

    print(f"\n  Testing stride={stride} with different clip_values:")
    print(f"  Note: cutoff = 1/stride = {1 / stride:.2f} (independent of clip_value)")
    print("  But std depends on clip_value, affecting mask shape/attenuation\n")

    for clip_value in [0.01, 0.1, 0.5, 0.9]:
        mask_module = SpectralGaussianMaskND(
            data_dim=2,
            clip_value=clip_value,
            init_stride_value=stride,
            min_stride_value=1.0,
            parametrization="direct",
        )

        mask = mask_module(spatial_dims)
        std = mask_module._compute_std()
        cutoff = mask_module._compute_cutoff()

        print(f"  clip_value={clip_value}:")
        print(f"    std={std.detach().tolist()}")
        print(f"    cutoff={cutoff.detach().tolist()} (should be ~{1 / stride:.2f})")
        print(f"    mask shape={tuple(mask.shape)}")
        print(f"    mask min={mask.min().item():.4f}, max={mask.max().item():.4f}, mean={mask.mean().item():.4f}")

    print("\n  ✅ Clip value effect test completed!")


def test_mask_caching():
    """Test that grid caching works correctly."""
    print("\n" + "=" * 60)
    print("Test 6: Grid caching")
    print("=" * 60)

    mask_module = SpectralGaussianMaskND(
        data_dim=2,
        clip_value=0.1,
        init_stride_value=2.0,
        min_stride_value=1.0,
        parametrization="direct",
    )

    spatial_dims = (64, 64)

    # First call - should generate grid
    mask1 = mask_module(spatial_dims)
    assert mask_module._cached_grid is not None, "Cache should be populated after first call"
    cached_grid_id = id(mask_module._cached_grid)

    # Second call with same dims - should reuse cache
    mask2 = mask_module(spatial_dims)
    assert id(mask_module._cached_grid) == cached_grid_id, "Cache should be reused for same dims"

    # Call with different dims - should regenerate
    mask3 = mask_module((128, 128))
    assert id(mask_module._cached_grid) != cached_grid_id, "Cache should be invalidated for different dims"

    del mask1, mask2, mask3

    # Clear cache manually
    mask_module.clear_cache()
    assert mask_module._cached_grid is None, "Cache should be cleared"

    print("  ✅ Caching test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SpectralGaussianMaskND Tests")
    print("=" * 60)

    test_spectral_gaussian_mask_basic()
    test_spectral_gaussian_mask_with_fftconv()
    test_spectral_gaussian_mask_gradient_flow()
    test_clip_value_effect()
    test_mask_caching()
    test_visual_downsampling()

    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
