# TODO: Add license header here

"""Verification test: Apply spectral mask with stride < 1.0 (no filtering) to EMNIST samples.

This test verifies that when stride < 1.0, the spectral mask does NO filtering,
meaning the output should be identical to the input (within numerical precision).

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/test_no_filtering_verification.py
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from nvsubquadratic.modules.masks_nd import SpectralLinearMaskND
from nvsubquadratic.ops.spectral_masking import spectral_downsampling2d_bhl


def get_emnist_samples(num_samples: int = 2):
    """Load EMNIST spatial recall samples."""
    from experiments.datamodules.emnist import EMNISTDataModule
    from experiments.datamodules.spatial_recall_dataset import SpatialRecallDataModule
    from nvsubquadratic.lazy_config import LazyConfig

    print("Loading EMNIST spatial recall dataset...")

    base_datamodule_cfg = LazyConfig(EMNISTDataModule)(
        data_dir=".data/emnist",
        batch_size=8,
        data_type="image",
        num_workers=0,
        pin_memory=False,
        permuted=False,
        seed=42,
        normalize_input=True,
        split="byclass",
    )

    datamodule = SpatialRecallDataModule(
        base_datamodule_cfg=base_datamodule_cfg,
        target_size=16,
        canvas_size=64,
        data_type="image",
        placement="random",
        with_mask=False,
        use_colored_frames=True,
        num_items=4,
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    dataloader = datamodule.val_dataloader()
    batch = next(iter(dataloader))
    inputs, targets = batch

    inputs = inputs[:num_samples]

    # Ensure [B, C, H, W] format
    if inputs.shape[1] != 3:
        inputs = inputs.permute(0, 3, 1, 2).contiguous()

    print(f"  Loaded {inputs.shape[0]} samples, shape: {tuple(inputs.shape)}")
    return inputs


def apply_spectral_mask(x: torch.Tensor, stride: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply spectral mask with given stride."""
    B, C, H, W = x.shape

    mask_module = SpectralLinearMaskND(
        data_dim=2,
        transition_fraction=0.1,
        init_stride_value=stride,
        min_stride_value=0.5,  # Allow stride < 1.0
    )

    spectral_mask = mask_module((H, W))
    spectral_mask_expanded = spectral_mask.unsqueeze(1).expand(-1, C, -1, -1)

    output = spectral_downsampling2d_bhl(x, spectral_mask_expanded)

    return output, spectral_mask


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    arr = t.squeeze().detach().cpu().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))  # [C, H, W] -> [H, W, C]
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 3:
        return Image.fromarray(arr, mode="RGB")
    return Image.fromarray(arr, mode="L")


def test_no_filtering():
    """Test that stride < 1.0 results in no filtering."""
    print("\n" + "=" * 80)
    print("Verification: Spectral mask with stride < 1.0 should do NO filtering")
    print("=" * 80)

    output_dir = Path(__file__).parent

    # Load samples
    inputs = get_emnist_samples(num_samples=2)

    # Test different strides
    strides = [0.5, 0.75, 1.0, 2.0]

    print("\nApplying spectral Linear mask with different strides:")
    print("-" * 80)

    results = {}
    for stride in strides:
        output, mask = apply_spectral_mask(inputs, stride)

        # Check if output equals input (for stride < 1.0)
        if output.shape == inputs.shape:
            diff = (output - inputs).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
        else:
            max_diff = float("nan")
            mean_diff = float("nan")

        results[stride] = {
            "output": output,
            "mask": mask,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        }

        print(f"  stride={stride:.2f}:")
        print(f"    input shape:  {tuple(inputs.shape)}")
        print(f"    output shape: {tuple(output.shape)}")
        print(f"    mask shape:   {tuple(mask.shape)}")
        print(f"    mask min/max/mean: {mask.min().item():.4f} / {mask.max().item():.4f} / {mask.mean().item():.4f}")
        if not np.isnan(max_diff):
            print(f"    output-input diff: max={max_diff:.6f}, mean={mean_diff:.6f}")
            if max_diff < 1e-5:
                print("    ✅ NO FILTERING (output ≈ input)")
            else:
                print("    ⚠️  Some difference detected")
        else:
            print("    (shapes differ, cannot compare)")

    # Create visualization
    print("\nCreating visualization...")
    create_visualization(inputs, results, strides, output_dir)


def create_visualization(inputs: torch.Tensor, results: dict, strides: list, output_dir: Path):
    """Create visualization comparing input vs output vs difference for each stride."""
    num_samples = inputs.shape[0]
    # Only show strides where output has same shape as input (for difference comparison)
    valid_strides = [s for s in strides if results[s]["output"].shape == inputs.shape]
    num_strides = len(valid_strides)

    cell_size = 128
    padding = 10
    label_height = 25
    row_label_width = 80
    section_padding = 15

    # For each stride: 3 columns (Original | Output | Difference)
    cols_per_stride = 3
    num_cols = num_strides * cols_per_stride
    num_rows = num_samples

    total_width = row_label_width + num_cols * cell_size + (num_cols + 1) * padding
    total_height = label_height + num_rows * (cell_size + label_height) + (num_strides) * section_padding

    canvas = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_bold = font
        font_small = font

    # Draw header for each stride section
    for stride_idx, stride in enumerate(valid_strides):
        x_base = row_label_width + stride_idx * (cols_per_stride * (cell_size + padding) + section_padding)

        max_diff = results[stride]["max_diff"]
        mask_mean = results[stride]["mask"].mean().item()

        # Stride header
        if max_diff < 1e-5:
            header_color = (100, 255, 100)  # Green
            status = "✓ NO FILTERING"
        else:
            header_color = (255, 200, 100)  # Orange
            status = f"diff={max_diff:.4f}"

        draw.text(
            (x_base, 2), f"stride={stride:.2f} (mask={mask_mean:.2f}) {status}", fill=header_color, font=font_bold
        )

        # Column headers
        draw.text((x_base + padding, label_height - 10), "Original", fill=(180, 180, 180), font=font_small)
        draw.text(
            (x_base + cell_size + 2 * padding, label_height - 10), "Output", fill=(180, 180, 180), font=font_small
        )
        draw.text(
            (x_base + 2 * cell_size + 3 * padding, label_height - 10),
            "Difference",
            fill=(180, 180, 180),
            font=font_small,
        )

    for sample_idx in range(num_samples):
        y_offset = label_height + sample_idx * (cell_size + label_height) + padding

        # Row label
        draw.text((padding, y_offset + cell_size // 2), f"Sample {sample_idx + 1}", fill=(200, 200, 200), font=font)

        for stride_idx, stride in enumerate(valid_strides):
            x_base = row_label_width + stride_idx * (cols_per_stride * (cell_size + padding) + section_padding)

            output = results[stride]["output"][sample_idx]
            input_tensor = inputs[sample_idx]

            # Compute difference
            diff = (output - input_tensor).abs()

            # Original
            x_pos = x_base + padding
            input_img = tensor_to_image(input_tensor)
            input_img_display = input_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
            canvas.paste(input_img_display, (x_pos, y_offset))

            # Output
            x_pos = x_base + cell_size + 2 * padding
            output_img = tensor_to_image(output)
            output_img_display = output_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
            canvas.paste(output_img_display, (x_pos, y_offset))

            # Difference (amplified for visibility)
            x_pos = x_base + 2 * cell_size + 3 * padding
            # Amplify difference for visualization (scale to [0, 1])
            diff_max = diff.max().item()
            if diff_max > 0:
                diff_normalized = diff / diff_max  # Normalize to [0, 1]
            else:
                diff_normalized = diff
            diff_img = tensor_to_image(diff_normalized)
            diff_img_display = diff_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
            canvas.paste(diff_img_display, (x_pos, y_offset))

            # Add diff stats below the difference image
            draw.text((x_pos, y_offset + cell_size + 2), f"max={diff_max:.6f}", fill=(150, 150, 150), font=font_small)

    output_path = output_dir / "no_filtering_verification.png"
    canvas.save(output_path)
    print(f"\n✅ Visualization saved to: {output_path}")


if __name__ == "__main__":
    test_no_filtering()
