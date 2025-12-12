# TODO: Add license header here

"""Visual test for spectral downsampling with a real image.

Uses scipy's cameraman test image, applies identity convolution with spectral downsampling,
and saves input/output images for visual verification.

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/test_spectral_visual.py
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nvsubquadratic.ops.fftconv import fftconv2d_bhl


def get_test_image():
    """Get a test image (cameraman from scipy or generate a pattern)."""
    try:
        from scipy.datasets import face

        # Use scipy's raccoon face image and convert to grayscale
        img_np = face(gray=True).astype(np.float32)
        # Crop to square and resize
        h, w = img_np.shape
        size = min(h, w)
        start_h, start_w = (h - size) // 2, (w - size) // 2
        img_np = img_np[start_h : start_h + size, start_w : start_w + size]
        img = Image.fromarray(img_np.astype(np.uint8), mode="L")
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return img
    except ImportError:
        pass

    try:
        from skimage import data

        img_np = data.camera()  # Classic cameraman image
        img = Image.fromarray(img_np, mode="L")
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        return img
    except ImportError:
        pass

    # Fallback: generate a test pattern with gradients and edges
    print("scipy/skimage not available, generating test pattern...")
    size = 256
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


def main():
    output_dir = Path(__file__).parent

    # Get test image
    img = get_test_image()
    print(f"Using image of size {img.size}")

    # Convert to tensor: (B, C, H, W)
    img_np = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)

    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")

    # Create identity kernel (1 at center, 0 elsewhere)
    # Kernel size same as input for global convolution
    K_H, K_W = 256, 256
    kernel = torch.zeros(1, 1, K_H, K_W, dtype=torch.float32)
    kernel[:, :, K_H // 2, K_W // 2] = 1.0

    # Test 1: No spectral mask (should preserve input)
    y_no_mask = fftconv2d_bhl(x, kernel, shortcut=None, spectral_mask=None)
    print(f"No mask output shape: {y_no_mask.shape}")

    # Test 2: Spectral mask for 2x downsampling (256 -> 128)
    target_H, target_W = 128, 128
    sM_x = target_H
    sM_y = target_W // 2 + 1  # For rfft2
    spectral_mask_2x = torch.ones(1, 1, sM_x, sM_y, dtype=torch.float32)
    y_2x = fftconv2d_bhl(x, kernel, shortcut=None, spectral_mask=spectral_mask_2x)
    print(f"2x downsample output shape: {y_2x.shape}")

    # Test 3: Spectral mask for 4x downsampling (256 -> 64)
    target_H, target_W = 64, 64
    sM_x = target_H
    sM_y = target_W // 2 + 1
    spectral_mask_4x = torch.ones(1, 1, sM_x, sM_y, dtype=torch.float32)
    y_4x = fftconv2d_bhl(x, kernel, shortcut=None, spectral_mask=spectral_mask_4x)
    print(f"4x downsample output shape: {y_4x.shape}")

    # Convert tensors to images
    def tensor_to_image(t):
        """Convert tensor to PIL Image."""
        arr = t.squeeze().numpy()
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")

    # Create a single combined comparison image
    # Layout: 2 rows x 2 columns
    # Row 1: Original (256x256) | 2x downsampled (128x128, shown at native size)
    # Row 2: 4x downsampled (64x64, shown at native size) | All upscaled to 256 for comparison

    img_input = tensor_to_image(x)  # 256x256
    img_2x = tensor_to_image(y_2x)  # 128x128
    img_4x = tensor_to_image(y_4x)  # 64x64

    # Create comparison with native sizes + upscaled versions
    padding = 10
    label_height = 25

    # Top row: Original + 2x native + 4x native
    top_width = 256 + padding + 128 + padding + 64
    # Bottom row: All three upscaled to 256 for direct comparison
    bottom_width = 256 * 3 + padding * 2

    total_width = max(top_width, bottom_width)
    total_height = 256 + label_height + padding + 256 + label_height

    # Create canvas (RGB for colored labels)
    from PIL import ImageDraw, ImageFont

    canvas = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    # Try to get a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Row 1: Native resolution images
    y_offset = 0

    # Original 256x256
    x_pos = 0
    draw.text((x_pos, y_offset), "Original (256x256)", fill=(255, 255, 255), font=font)
    canvas.paste(img_input.convert("RGB"), (x_pos, y_offset + label_height))

    # 2x downsampled 128x128 (native)
    x_pos = 256 + padding
    draw.text((x_pos, y_offset), "2x down (128x128)", fill=(255, 255, 255), font=font)
    canvas.paste(img_2x.convert("RGB"), (x_pos, y_offset + label_height))

    # 4x downsampled 64x64 (native)
    x_pos = 256 + padding + 128 + padding
    draw.text((x_pos, y_offset), "4x down (64x64)", fill=(255, 255, 255), font=font)
    canvas.paste(img_4x.convert("RGB"), (x_pos, y_offset + label_height))

    # Row 2: All upscaled to 256x256 for comparison
    y_offset = 256 + label_height + padding

    img_input_up = img_input.resize((256, 256), Image.Resampling.NEAREST)
    img_2x_up = img_2x.resize((256, 256), Image.Resampling.NEAREST)
    img_4x_up = img_4x.resize((256, 256), Image.Resampling.NEAREST)

    x_pos = 0
    draw.text((x_pos, y_offset), "Original (reference)", fill=(200, 200, 200), font=font)
    canvas.paste(img_input_up.convert("RGB"), (x_pos, y_offset + label_height))

    x_pos = 256 + padding
    draw.text((x_pos, y_offset), "2x upscaled to 256", fill=(200, 200, 200), font=font)
    canvas.paste(img_2x_up.convert("RGB"), (x_pos, y_offset + label_height))

    x_pos = 256 * 2 + padding * 2
    draw.text((x_pos, y_offset), "4x upscaled to 256", fill=(200, 200, 200), font=font)
    canvas.paste(img_4x_up.convert("RGB"), (x_pos, y_offset + label_height))

    # Save the single combined image
    output_path = output_dir / "spectral_downsampling_test.png"
    canvas.save(output_path)

    print("\n✅ Visual test complete!")
    print(f"   Saved to: {output_path}")


if __name__ == "__main__":
    main()
