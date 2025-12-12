# TODO: Add license header here

"""Ablation study for clip_value in SpectralGaussianMaskND.

Creates a grid visualization comparing different clip_values (rows)
across different strides (columns) to help determine the optimal clip_value.

Each downsampled image is upscaled back to original size for visual comparison.

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/test_clip_value_ablation.py
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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


def apply_spectral_downsampling(
    x: torch.Tensor,
    kernel: torch.Tensor,
    stride: float,
    clip_value: float,
) -> tuple[torch.Tensor, torch.Tensor, SpectralGaussianMaskND]:
    """Apply spectral downsampling with given stride and clip_value.

    Returns:
        Tuple of (output_tensor, spectral_mask, mask_module)
    """
    H, W = x.shape[2], x.shape[3]

    mask_module = SpectralGaussianMaskND(
        data_dim=2,
        clip_value=clip_value,
        init_stride_value=stride,
        min_stride_value=1.0,
        max_stride_value=None,
        parametrization="direct",
    )

    mask = mask_module((H, W))
    spectral_mask = mask.unsqueeze(1)  # (1, 1, sM_x, sM_y)

    y = fftconv2d_bhl(x, kernel, shortcut=None, spectral_mask=spectral_mask)

    return y, spectral_mask, mask_module


def run_clip_value_ablation():
    """Generate ablation grid comparing clip_values across strides."""
    print("=" * 70)
    print("Clip Value Ablation Study")
    print("=" * 70)

    output_dir = Path(__file__).parent

    # Parameters
    img_size = 256
    strides = [2.0, 4.0, 8.0, 16.0]
    clip_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Get test image
    img = get_test_image(img_size)
    print(f"Using image of size {img.size}")

    # Convert to tensor: (B, C, H, W)
    img_np = np.array(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)

    # Create identity kernel
    kernel = torch.zeros(1, 1, img_size, img_size, dtype=torch.float32)
    kernel[:, :, img_size // 2, img_size // 2] = 1.0

    # Store results: dict[(clip_value, stride)] -> (output_img, mask_mean, output_shape)
    results = {}

    print("\nProcessing grid...")
    for clip_value in clip_values:
        for stride in strides:
            y, mask, mask_module = apply_spectral_downsampling(x, kernel, stride, clip_value)

            output_img = tensor_to_image(y)
            mask_mean = mask.mean().item()
            output_shape = (y.shape[2], y.shape[3])

            results[(clip_value, stride)] = (output_img, mask_mean, output_shape)

            print(f"  clip={clip_value:.2f}, stride={stride:.1f}: output={output_shape}, mask_mean={mask_mean:.4f}")

    # Create visualization grid
    print("\nCreating visualization...")

    # Layout parameters
    cell_size = 128  # Size of each upscaled image
    padding = 5
    label_height = 20
    row_label_width = 100
    col_label_height = 25

    n_rows = len(clip_values) + 1  # +1 for original row
    n_cols = len(strides) + 1  # +1 for original column

    total_width = row_label_width + n_cols * (cell_size + padding)
    total_height = col_label_height + n_rows * (cell_size + label_height + padding)

    canvas = Image.new("RGB", (total_width, total_height), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    # Column headers (strides)
    x_pos = row_label_width
    draw.text((x_pos, 5), "Original", fill=(255, 255, 255), font=font)
    x_pos += cell_size + padding

    for stride in strides:
        draw.text((x_pos, 5), f"stride={stride:.0f}x", fill=(255, 255, 255), font=font)
        x_pos += cell_size + padding

    # First row: original image repeated
    y_pos = col_label_height
    x_pos = 0

    draw.text((x_pos, y_pos + cell_size // 2), "Original", fill=(200, 200, 200), font=font)
    x_pos = row_label_width

    original_img = tensor_to_image(x).resize((cell_size, cell_size), Image.Resampling.NEAREST)
    for _ in range(n_cols):
        canvas.paste(original_img.convert("RGB"), (x_pos, y_pos))
        x_pos += cell_size + padding

    # Rows for each clip_value
    for row_idx, clip_value in enumerate(clip_values):
        y_pos = col_label_height + (row_idx + 1) * (cell_size + label_height + padding)
        x_pos = 0

        # Row label
        draw.text((x_pos, y_pos + cell_size // 2 - 10), f"clip={clip_value}", fill=(200, 200, 200), font=font)
        x_pos = row_label_width

        # Original image in first column
        canvas.paste(original_img.convert("RGB"), (x_pos, y_pos))
        x_pos += cell_size + padding

        # Downsampled images for each stride
        for stride in strides:
            output_img, mask_mean, output_shape = results[(clip_value, stride)]

            # Upscale to cell_size for comparison
            upscaled_img = output_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
            canvas.paste(upscaled_img.convert("RGB"), (x_pos, y_pos))

            # Add info label below the image
            info_text = f"{output_shape[0]}x{output_shape[1]}, μ={mask_mean:.2f}"
            draw.text((x_pos, y_pos + cell_size + 2), info_text, fill=(150, 150, 150), font=font_small)

            x_pos += cell_size + padding

    # # Add title
    # title = "Clip Value Ablation: clip_value (rows) vs stride (columns)"
    # # Draw title at top (we need to adjust canvas height for this)

    # Save
    output_path = output_dir / "clip_value_ablation.png"
    canvas.save(output_path)

    print("\n✅ Ablation complete!")
    print(f"   Saved to: {output_path}")
    print(f"   Grid: {len(clip_values)} clip_values x {len(strides)} strides")

    # Also print a summary table
    print("\n" + "=" * 70)
    print("Summary: Output sizes (HxW)")
    print("=" * 70)
    print(f"{'clip_value':>10} | " + " | ".join(f"s={s:.0f}x" for s in strides))
    print("-" * 70)
    for clip_value in clip_values:
        row = f"{clip_value:>10.2f} | "
        sizes = []
        for stride in strides:
            _, _, shape = results[(clip_value, stride)]
            sizes.append(f"{shape[0]:3d}x{shape[1]:<3d}")
        row += " | ".join(sizes)
        print(row)

    print("\n" + "=" * 70)
    print("Summary: Mask mean (brightness preservation)")
    print("=" * 70)
    print(f"{'clip_value':>10} | " + " | ".join(f"s={s:.0f}x" for s in strides))
    print("-" * 70)
    for clip_value in clip_values:
        row = f"{clip_value:>10.2f} | "
        means = []
        for stride in strides:
            _, mask_mean, _ = results[(clip_value, stride)]
            means.append(f"{mask_mean:7.4f}")
        row += " | ".join(means)
        print(row)


if __name__ == "__main__":
    run_clip_value_ablation()
