# TODO: Add license header here

"""Visual test for spectral downsampling on EMNIST spatial recall samples.

This test applies spectral downsampling with different strides to EMNIST 2D spatial recall
samples to diagnose if information is being erased during the downsampling process.

Applies strides: 1.0, 2.0, 4.0, 8.0, 16.0 to visualize quality degradation.

Usage:
    PYTHONPATH=. python nvsubquadratic/ops/test_spectral_downsampling_emnist_spatial_recall.py
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from experiments.datamodules.emnist import EMNISTDataModule
from experiments.datamodules.spatial_recall_dataset import SpatialRecallDataModule
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import (
    SpectralGaussianMaskND,
    SpectralLinearMaskND,
    SpectralSigmoidMaskND,
)
from nvsubquadratic.ops.spectral_masking import spectral_downsampling2d_bhl


# Dataset parameters
TARGET_SIZE = 16
CANVAS_SIZE = 64
BATCH_SIZE = 8
NUM_SAMPLES = 4  # Number of samples to visualize

# Spectral mask parameters
CLIP_VALUE = 0.5
STRIDES = [1.0, 2.0, 4.0, 8.0, 16.0]
CLIP_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]  # Different clip values to test

# Mask types for comparison (spectral methods)
SPECTRAL_MASK_TYPES = {
    "Gaussian (clip=0.5)": {
        "class": SpectralGaussianMaskND,
        "kwargs": {"clip_value": 0.5},
    },
    "Linear (frac=0.2)": {
        "class": SpectralLinearMaskND,
        "kwargs": {"transition_fraction": 0.2},
    },
    "Sigmoid (T=20)": {
        "class": SpectralSigmoidMaskND,
        "kwargs": {"temperature": 20.0},
    },
}

# Baseline interpolation methods (non-spectral)
BASELINE_METHODS = ["Nearest", "Bilinear"]

# All methods: baselines first, then spectral
ALL_METHODS = BASELINE_METHODS + list(SPECTRAL_MASK_TYPES.keys())


def get_spatial_recall_samples(num_samples: int = NUM_SAMPLES) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load EMNIST spatial recall samples.

    Returns:
        Tuple of (inputs, targets) where:
            inputs: [N, C, H, W] canvas images (RGB with colored frames)
            targets: [N, 1, target_size, target_size] target images to recall
    """
    print("Loading EMNIST spatial recall dataset...")

    # Create base EMNIST datamodule config
    base_datamodule_cfg = LazyConfig(EMNISTDataModule)(
        data_dir=".data/emnist",
        batch_size=BATCH_SIZE,
        data_type="image",
        num_workers=0,
        pin_memory=False,
        permuted=False,
        seed=42,
        normalize_input=True,
        split="byclass",
    )

    # Create spatial recall datamodule
    datamodule = SpatialRecallDataModule(
        base_datamodule_cfg=base_datamodule_cfg,
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        data_type="image",
        placement="random",
        with_mask=False,
        use_colored_frames=True,  # RGB with colored bounding boxes
        num_items=4,  # 1 target + 3 distractors
    )

    # Setup the datamodule
    datamodule.prepare_data()
    datamodule.setup("fit")

    # Get a batch of samples
    dataloader = datamodule.val_dataloader()
    batch = next(iter(dataloader))
    inputs, targets = batch

    # Datamodule returns [B, H, W, C] format for data_type="image"
    # Convert to [B, C, H, W] for convolution operations
    inputs = inputs[:num_samples]
    targets = targets[:num_samples]

    print(f"  Raw input shape: {tuple(inputs.shape)}")
    print(f"  Raw target shape: {tuple(targets.shape)}")

    # Check if already in [B, C, H, W] format (C < H typically)
    if inputs.shape[1] == 3 and inputs.shape[2] == CANVAS_SIZE:
        # Already [B, C, H, W]
        print("  Input already in [B, C, H, W] format")
    else:
        # Convert from [B, H, W, C] to [B, C, H, W]
        inputs = inputs.permute(0, 3, 1, 2).contiguous()
        targets = targets.permute(0, 3, 1, 2).contiguous()
        print("  Converted from [B, H, W, C] to [B, C, H, W]")

    print(f"  Loaded {inputs.shape[0]} samples")
    print(f"  Input shape: {tuple(inputs.shape)} (B, C, H, W)")
    print(f"  Target shape: {tuple(targets.shape)} (B, 1, H, W)")
    print(f"  Input range: [{inputs.min():.3f}, {inputs.max():.3f}]")
    print(f"  Target range: [{targets.min():.3f}, {targets.max():.3f}]")

    return inputs, targets


def apply_spectral_downsampling(
    x: torch.Tensor,
    stride: float,
    clip_value: float = CLIP_VALUE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply pure spectral masking (FFT -> mask -> crop -> IFFT) without convolution.

    This directly shows what information is preserved by the spectral mask.
    Uses the existing spectral_downsampling2d_bhl function.

    Args:
        x: Input tensor [B, C, H, W]
        stride: Downsampling stride
        clip_value: Gaussian mask clip value

    Returns:
        Tuple of (output, spectral_mask) where:
            output: Downsampled tensor [B, C, H', W']
            spectral_mask: The spectral mask used [1, sM_h, sM_w]
    """
    B, C, H, W = x.shape

    # Create spectral mask
    mask_module = SpectralGaussianMaskND(
        data_dim=2,
        clip_value=clip_value,
        init_stride_value=stride,
        min_stride_value=1.0,
        max_stride_value=None,
        parametrization="direct",
    )

    # Generate spectral mask: [1, sM_h, sM_w]
    spectral_mask = mask_module((H, W))

    # Get crop dimensions from mask
    sM_h, sM_w = spectral_mask.shape[1], spectral_mask.shape[2]
    target_H, target_W = sM_h, 2 * (sM_w - 1)

    print(f"    spectral_mask shape: {tuple(spectral_mask.shape)}, target output: ({target_H}, {target_W})")

    # Expand mask to [1, C, sM_h, sM_w] for spectral_downsampling2d_bhl
    spectral_mask_expanded = spectral_mask.unsqueeze(1).expand(-1, C, -1, -1)

    # Apply spectral downsampling using the existing function
    output = spectral_downsampling2d_bhl(x, spectral_mask_expanded)

    return output, spectral_mask


def apply_baseline_downsampling(
    x: torch.Tensor,
    stride: float,
    method: str = "Bilinear",
) -> torch.Tensor:
    """Apply baseline interpolation downsampling.

    Args:
        x: Input tensor [B, C, H, W]
        stride: Downsampling stride
        method: Interpolation method ("Nearest" or "Bilinear")

    Returns:
        Downsampled tensor [B, C, H', W']
    """
    B, C, H, W = x.shape
    target_H = int(H / stride)
    target_W = int(W / stride)

    # Map method names to PyTorch interpolate modes
    mode_map = {
        "Nearest": "nearest",
        "Bilinear": "bilinear",
    }
    mode = mode_map.get(method, "bilinear")

    # Use PyTorch's interpolate
    if mode == "nearest":
        output = torch.nn.functional.interpolate(
            x,
            size=(target_H, target_W),
            mode=mode,
        )
    else:
        output = torch.nn.functional.interpolate(
            x,
            size=(target_H, target_W),
            mode=mode,
            align_corners=False,
        )

    return output


def apply_spectral_downsampling_with_mask_type(
    x: torch.Tensor,
    stride: float,
    mask_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply spectral downsampling with a specific mask type.

    Args:
        x: Input tensor [B, C, H, W]
        stride: Downsampling stride
        mask_type: Key from SPECTRAL_MASK_TYPES dict

    Returns:
        Tuple of (output, spectral_mask)
    """
    B, C, H, W = x.shape

    # Get mask class and kwargs from SPECTRAL_MASK_TYPES
    mask_config = SPECTRAL_MASK_TYPES[mask_type]
    mask_class = mask_config["class"]
    mask_kwargs = mask_config["kwargs"].copy()

    # Create spectral mask module
    mask_module = mask_class(
        data_dim=2,
        init_stride_value=stride,
        min_stride_value=1.0,
        max_stride_value=None,
        parametrization="direct",
        **mask_kwargs,
    )

    # Generate spectral mask: [1, sM_h, sM_w]
    spectral_mask = mask_module((H, W))

    # Get crop dimensions from mask
    _sM_h, _sM_w = spectral_mask.shape[1], spectral_mask.shape[2]

    # Expand mask to [1, C, sM_h, sM_w] for spectral_downsampling2d_bhl
    spectral_mask_expanded = spectral_mask.unsqueeze(1).expand(-1, C, -1, -1)

    # Apply spectral downsampling using the existing function
    output = spectral_downsampling2d_bhl(x, spectral_mask_expanded)

    return output, spectral_mask


def apply_downsampling(
    x: torch.Tensor,
    stride: float,
    method: str,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Apply downsampling with any method (baseline or spectral).

    Args:
        x: Input tensor [B, C, H, W]
        stride: Downsampling stride
        method: Method name from ALL_METHODS

    Returns:
        Tuple of (output, mask) where mask is None for baseline methods
    """
    if method in BASELINE_METHODS:
        output = apply_baseline_downsampling(x, stride, method)
        return output, None
    else:
        return apply_spectral_downsampling_with_mask_type(x, stride, method)


def tensor_to_image(t: torch.Tensor, is_rgb: bool = False) -> Image.Image:
    """Convert tensor to PIL Image.

    Args:
        t: Tensor of shape [C, H, W] or [H, W]
        is_rgb: Whether the tensor is RGB (3 channels)
    """
    arr = t.squeeze().detach().cpu().numpy()

    if is_rgb and arr.ndim == 3:
        # [C, H, W] -> [H, W, C]
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    else:
        if arr.ndim == 3:
            arr = arr[0]  # Take first channel
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L")


def test_spectral_downsampling_on_spatial_recall():
    """Test spectral downsampling on EMNIST spatial recall samples."""
    print("\n" + "=" * 80)
    print("Test: Spectral downsampling on EMNIST spatial recall samples")
    print("=" * 80)

    output_dir = Path(__file__).parent

    # Load samples
    inputs, targets = get_spatial_recall_samples()

    # Apply spectral downsampling with different strides
    print(f"\nApplying spectral downsampling with strides: {STRIDES}")

    results = {}
    for stride in STRIDES:
        output, mask = apply_spectral_downsampling(inputs, stride)
        results[stride] = {
            "output": output,
            "mask": mask,
            "shape": tuple(output.shape),
        }
        print(f"  stride={stride:5.1f}: input {tuple(inputs.shape)} -> output {tuple(output.shape)}")
        print(
            f"              output range: [{output.min():.3f}, {output.max():.3f}], "
            f"mask shape: {tuple(mask.shape)}, mask mean: {mask.mean():.3f}"
        )

    # Create visualization
    print("\nCreating visualization...")
    create_visualization(inputs, targets, results, output_dir)


def create_visualization(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    results: dict,
    output_dir: Path,
):
    """Create a visualization of the spectral downsampling results.

    Layout:
    - Rows: Different samples
    - Columns: Original input | Target | Stride 1.0 | Stride 2.0 | Stride 4.0 | Stride 8.0 | Stride 16.0

    Each downsampled result is shown both at native size and upscaled.
    """
    num_samples = inputs.shape[0]
    num_strides = len(STRIDES)

    # Layout parameters
    cell_size = CANVAS_SIZE * 2  # Display size for each cell
    padding = 10
    label_height = 25

    # Two rows per sample: native size + upscaled
    # Columns: Input | Target | Stride outputs...
    num_cols = 2 + num_strides  # Input, Target, + one per stride
    num_rows = num_samples * 2  # Native + upscaled for each sample

    total_width = num_cols * cell_size + (num_cols + 1) * padding
    total_height = num_rows * (cell_size + label_height) + (num_rows // 2 + 1) * padding * 2

    canvas = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for sample_idx in range(num_samples):
        # Row offset for this sample (2 rows per sample)
        base_row = sample_idx * 2
        y_offset_native = base_row * (cell_size + label_height) + (sample_idx + 1) * padding * 2
        y_offset_upscaled = (base_row + 1) * (cell_size + label_height) + (sample_idx + 1) * padding * 2

        # Sample separator
        if sample_idx > 0:
            y_sep = y_offset_native - padding
            draw.line([(0, y_sep), (total_width, y_sep)], fill=(80, 80, 80), width=2)

        # Column 0: Original input
        x_pos = padding
        input_img = tensor_to_image(inputs[sample_idx], is_rgb=True)
        input_img_display = input_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

        draw.text((x_pos, y_offset_native), f"Input ({CANVAS_SIZE}x{CANVAS_SIZE})", fill=(255, 255, 255), font=font)
        canvas.paste(input_img_display, (x_pos, y_offset_native + label_height))

        draw.text((x_pos, y_offset_upscaled), "Sample " + str(sample_idx + 1), fill=(150, 150, 150), font=font)
        canvas.paste(input_img_display, (x_pos, y_offset_upscaled + label_height))

        # Column 1: Target
        x_pos = padding + cell_size + padding
        target_img = tensor_to_image(targets[sample_idx], is_rgb=False)
        target_img_display = target_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

        draw.text((x_pos, y_offset_native), f"Target ({TARGET_SIZE}x{TARGET_SIZE})", fill=(255, 255, 255), font=font)
        canvas.paste(target_img_display.convert("RGB"), (x_pos, y_offset_native + label_height))

        draw.text((x_pos, y_offset_upscaled), "(upscaled)", fill=(150, 150, 150), font=font)
        canvas.paste(target_img_display.convert("RGB"), (x_pos, y_offset_upscaled + label_height))

        # Columns 2+: Downsampled outputs
        for stride_idx, stride in enumerate(STRIDES):
            x_pos = padding + (2 + stride_idx) * (cell_size + padding)

            output = results[stride]["output"][sample_idx]
            output_shape = output.shape[-2:]  # H', W'
            output_img = tensor_to_image(output, is_rgb=True)

            # Native size row
            # Center the native image in the cell
            native_display_size = min(cell_size, max(output_img.width * 4, 32))  # Scale up small images
            output_native = output_img.resize((native_display_size, native_display_size), Image.Resampling.NEAREST)
            native_offset_x = (cell_size - native_display_size) // 2
            native_offset_y = (cell_size - native_display_size) // 2

            label = f"s={stride:.0f} ({output_shape[0]}x{output_shape[1]})"
            draw.text((x_pos, y_offset_native), label, fill=(255, 255, 255), font=font)

            # Create a cell with centered image
            native_cell = Image.new("RGB", (cell_size, cell_size), color=(40, 40, 40))
            native_cell.paste(output_native, (native_offset_x, native_offset_y))
            canvas.paste(native_cell, (x_pos, y_offset_native + label_height))

            # Upscaled row
            output_upscaled = output_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
            draw.text((x_pos, y_offset_upscaled), "(upscaled)", fill=(150, 150, 150), font=font)
            canvas.paste(output_upscaled, (x_pos, y_offset_upscaled + label_height))

    # Save
    output_path = output_dir / "spectral_downsampling_emnist_spatial_recall.png"
    canvas.save(output_path)
    print(f"\n✅ Visualization saved to: {output_path}")

    # Also create a separate mask visualization
    create_mask_visualization(results, output_dir)


def create_mask_visualization(results: dict, output_dir: Path):
    """Create a visualization of the spectral masks."""
    print("\nCreating mask visualization...")

    num_strides = len(STRIDES)
    cell_size = 128
    padding = 10
    label_height = 25

    total_width = num_strides * cell_size + (num_strides + 1) * padding
    total_height = cell_size + label_height + 2 * padding

    canvas = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for stride_idx, stride in enumerate(STRIDES):
        x_pos = padding + stride_idx * (cell_size + padding)
        y_pos = padding

        mask = results[stride]["mask"]  # [1, sM_h, sM_w]
        mask_np = mask[0].detach().cpu().numpy()

        # Normalize mask to [0, 1] for visualization
        mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
        mask_img_display = mask_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

        label = f"Stride {stride:.0f} ({mask.shape[1]}x{mask.shape[2]})"
        draw.text((x_pos, y_pos), label, fill=(255, 255, 255), font=font)
        canvas.paste(mask_img_display.convert("RGB"), (x_pos, y_pos + label_height))

    output_path = output_dir / "spectral_masks_emnist_spatial_recall.png"
    canvas.save(output_path)
    print(f"✅ Mask visualization saved to: {output_path}")


def test_information_preservation():
    """Quantitative test for information preservation during downsampling."""
    print("\n" + "=" * 80)
    print("Test: Information preservation analysis")
    print("=" * 80)

    # Load samples
    inputs, targets = get_spatial_recall_samples(num_samples=16)

    print("\nAnalyzing information preservation for different strides:")
    print("-" * 80)
    print(f"{'Stride':>8} | {'Out Shape':>12} | {'Mean':>8} | {'Std':>8} | {'Min':>8} | {'Max':>8} | {'Energy':>10}")
    print("-" * 80)

    # Original stats
    energy_orig = (inputs**2).mean().item()
    print(
        f"{'Orig':>8} | {str(tuple(inputs.shape[-2:])):>12} | "
        f"{inputs.mean().item():>8.4f} | {inputs.std().item():>8.4f} | "
        f"{inputs.min().item():>8.4f} | {inputs.max().item():>8.4f} | "
        f"{energy_orig:>10.4f}"
    )

    for stride in STRIDES:
        output, mask = apply_spectral_downsampling(inputs, stride)

        energy = (output**2).mean().item()
        energy_ratio = energy / energy_orig

        print(
            f"{stride:>8.1f} | {str(tuple(output.shape[-2:])):>12} | "
            f"{output.mean().item():>8.4f} | {output.std().item():>8.4f} | "
            f"{output.min().item():>8.4f} | {output.max().item():>8.4f} | "
            f"{energy:>10.4f} ({energy_ratio:.1%})"
        )

    print("-" * 80)


def test_clip_values():
    """Test spectral downsampling with different clip values."""
    print("\n" + "=" * 80)
    print("Test: Spectral downsampling with different clip values")
    print("=" * 80)

    output_dir = Path(__file__).parent

    # Load samples (just use 1 sample for this visualization)
    inputs, targets = get_spatial_recall_samples(num_samples=1)

    # Test a fixed stride with different clip values
    test_stride = 4.0  # Use stride 4.0 as a representative example

    print(f"\nApplying spectral downsampling with stride={test_stride} and clip_values: {CLIP_VALUES}")

    results_by_clip = {}
    for clip_value in CLIP_VALUES:
        output, mask = apply_spectral_downsampling(inputs, test_stride, clip_value)
        results_by_clip[clip_value] = {
            "output": output,
            "mask": mask,
            "shape": tuple(output.shape),
        }
        print(f"  clip={clip_value:.1f}: input {tuple(inputs.shape)} -> output {tuple(output.shape)}")
        print(
            f"              output range: [{output.min():.3f}, {output.max():.3f}], "
            f"mask shape: {tuple(mask.shape)}, mask mean: {mask.mean():.3f}"
        )

    # Create visualization for different clip values
    create_clip_value_visualization(inputs, targets, results_by_clip, test_stride, output_dir)

    # Also create a comprehensive grid: clip values x strides
    print("\nCreating comprehensive clip x stride visualization...")
    create_comprehensive_clip_stride_visualization(inputs, output_dir)


def create_clip_value_visualization(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    results_by_clip: dict,
    stride: float,
    output_dir: Path,
):
    """Create a visualization comparing different clip values.

    Layout:
    - Row 1: Original input | Mask clip=0.1 | Mask clip=0.3 | Mask clip=0.5 | Mask clip=0.7 | Mask clip=0.9
    - Row 2: Target | Output clip=0.1 | Output clip=0.3 | Output clip=0.5 | Output clip=0.7 | Output clip=0.9
    """
    num_clips = len(CLIP_VALUES)

    # Layout parameters
    cell_size = CANVAS_SIZE * 2  # Display size for each cell
    padding = 10
    label_height = 30

    num_cols = 1 + num_clips  # Input/Target + one per clip value
    num_rows = 2  # Masks row + Outputs row

    total_width = num_cols * cell_size + (num_cols + 1) * padding
    total_height = num_rows * (cell_size + label_height) + 3 * padding

    canvas = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Row 0: Input + Masks
    y_offset = padding

    # Column 0: Original input
    x_pos = padding
    input_img = tensor_to_image(inputs[0], is_rgb=True)
    input_img_display = input_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
    draw.text((x_pos, y_offset), f"Input ({CANVAS_SIZE}x{CANVAS_SIZE})", fill=(255, 255, 255), font=font)
    canvas.paste(input_img_display, (x_pos, y_offset + label_height))

    # Columns 1+: Masks for each clip value
    for clip_idx, clip_value in enumerate(CLIP_VALUES):
        x_pos = padding + (1 + clip_idx) * (cell_size + padding)

        mask = results_by_clip[clip_value]["mask"]  # [1, sM_h, sM_w]
        mask_np = mask[0].detach().cpu().numpy()

        # Normalize mask for visualization
        mask_np_viz = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
        mask_img = Image.fromarray((mask_np_viz * 255).astype(np.uint8), mode="L")
        mask_img_display = mask_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

        label = f"Mask clip={clip_value:.1f} (mean={mask.mean():.2f})"
        draw.text((x_pos, y_offset), label, fill=(255, 255, 255), font=font)
        canvas.paste(mask_img_display.convert("RGB"), (x_pos, y_offset + label_height))

    # Row 1: Target + Outputs
    y_offset = padding + cell_size + label_height + padding

    # Column 0: Target
    x_pos = padding
    target_img = tensor_to_image(targets[0], is_rgb=False)
    target_img_display = target_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
    draw.text((x_pos, y_offset), f"Target ({TARGET_SIZE}x{TARGET_SIZE})", fill=(255, 255, 255), font=font)
    canvas.paste(target_img_display.convert("RGB"), (x_pos, y_offset + label_height))

    # Columns 1+: Outputs for each clip value
    for clip_idx, clip_value in enumerate(CLIP_VALUES):
        x_pos = padding + (1 + clip_idx) * (cell_size + padding)

        output = results_by_clip[clip_value]["output"][0]
        output_shape = output.shape[-2:]
        output_img = tensor_to_image(output, is_rgb=True)
        output_img_display = output_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

        label = f"Output clip={clip_value:.1f} ({output_shape[0]}x{output_shape[1]})"
        draw.text((x_pos, y_offset), label, fill=(255, 255, 255), font=font)
        canvas.paste(output_img_display, (x_pos, y_offset + label_height))

    # Save
    output_path = output_dir / f"spectral_downsampling_clip_values_stride{stride:.0f}.png"
    canvas.save(output_path)
    print(f"\n✅ Clip value visualization saved to: {output_path}")


def create_comprehensive_clip_stride_visualization(inputs: torch.Tensor, output_dir: Path):
    """Create a comprehensive grid visualization: rows = clip values, cols = strides.

    Layout:
    - First column: Original input (repeated)
    - Columns 2+: Different strides
    - Rows: Different clip values
    """
    num_clips = len(CLIP_VALUES)
    num_strides = len(STRIDES)

    # Layout parameters
    cell_size = 80  # Smaller cells for the comprehensive view
    padding = 5
    label_height = 20
    row_label_width = 80

    total_width = row_label_width + (1 + num_strides) * (cell_size + padding) + padding
    total_height = label_height + num_clips * (cell_size + padding) + padding

    canvas = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Header row: column labels
    y_offset = 0
    x_pos = row_label_width + padding
    draw.text((x_pos, y_offset), "Input", fill=(255, 255, 255), font=font)

    for stride_idx, stride in enumerate(STRIDES):
        x_pos = row_label_width + (1 + stride_idx) * (cell_size + padding) + padding
        draw.text((x_pos, y_offset), f"s={stride:.0f}", fill=(255, 255, 255), font=font)

    # Prepare input image
    input_img = tensor_to_image(inputs[0], is_rgb=True)
    input_img_display = input_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

    # Generate all outputs
    for clip_idx, clip_value in enumerate(CLIP_VALUES):
        y_offset = label_height + clip_idx * (cell_size + padding) + padding

        # Row label
        draw.text((padding, y_offset + cell_size // 3), f"clip={clip_value:.1f}", fill=(200, 200, 200), font=font)

        # Column 0: Input
        x_pos = row_label_width + padding
        canvas.paste(input_img_display, (x_pos, y_offset))

        # Columns 1+: Outputs for each stride
        for stride_idx, stride in enumerate(STRIDES):
            x_pos = row_label_width + (1 + stride_idx) * (cell_size + padding) + padding

            output, _ = apply_spectral_downsampling(inputs, stride, clip_value)
            output_img = tensor_to_image(output[0], is_rgb=True)
            output_img_display = output_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

            canvas.paste(output_img_display, (x_pos, y_offset))

    # Save
    output_path = output_dir / "spectral_downsampling_clip_stride_grid.png"
    canvas.save(output_path)
    print(f"✅ Comprehensive clip x stride visualization saved to: {output_path}")


def test_mask_type_comparison():
    """Compare different methods: Bilinear baseline vs spectral masks (Gaussian, Linear, Sigmoid)."""
    print("\n" + "=" * 80)
    print("Test: Comparing downsampling methods (Bilinear, Gaussian, Linear, Sigmoid)")
    print("=" * 80)

    output_dir = Path(__file__).parent

    # Load samples
    inputs, targets = get_spatial_recall_samples(num_samples=2)

    # Test with different strides (including 1.5)
    test_strides = [1.5, 2.0, 4.0, 8.0]

    print(f"\nComparing methods: {ALL_METHODS}")
    print(f"Strides: {test_strides}")

    # Collect results: {method: {stride: {"output": ..., "mask": ...}}}
    all_results = {}

    for method in ALL_METHODS:
        all_results[method] = {}
        print(f"\n{method}:")

        for stride in test_strides:
            output, mask = apply_downsampling(inputs, stride, method)
            all_results[method][stride] = {
                "output": output,
                "mask": mask,
            }
            if mask is not None:
                print(
                    f"  stride={stride:.1f}: output shape {tuple(output.shape[-2:])}, "
                    f"mask shape {tuple(mask.shape[1:])}, "
                    f"mask mean={mask.mean():.3f}, mask max={mask.max():.3f}"
                )
            else:
                print(f"  stride={stride:.1f}: output shape {tuple(output.shape[-2:])} (no mask)")

    # Create visualization
    create_mask_type_comparison_visualization(inputs, targets, all_results, test_strides, output_dir)


def create_mask_type_comparison_visualization(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    all_results: dict,
    strides: List[float],
    output_dir: Path,
):
    """Create visualization comparing different downsampling methods.

    Layout:
    - Row label column on the left
    - Columns: Method | Input | Stride 1.5 | Stride 2.0 | Stride 4.0 | Stride 8.0 | Mask
    - Rows per sample: one row per method (Bilinear, Gaussian, Linear, Sigmoid)
    """
    num_samples = inputs.shape[0]
    num_strides = len(strides)
    method_names = ALL_METHODS
    num_methods = len(method_names)

    # Layout parameters
    cell_size = CANVAS_SIZE * 2
    padding = 8
    label_height = 20
    row_label_width = 120  # Width for method labels on left
    section_padding = 25

    # Columns: RowLabel | Input | stride outputs... | Mask
    num_data_cols = 1 + num_strides + 1  # Input + strides + Mask
    rows_per_sample = num_methods

    total_width = row_label_width + num_data_cols * cell_size + (num_data_cols + 1) * padding
    total_height = num_samples * (rows_per_sample * (cell_size + label_height) + section_padding) + section_padding

    canvas = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_bold = font
        font_small = font

    # Color coding for methods
    method_colors = {
        # Baselines (red tones)
        "Nearest": (255, 100, 100),  # Light red
        "Bilinear": (255, 150, 150),  # Lighter red
        # Spectral methods (other colors)
        "Gaussian (clip=0.5)": (120, 180, 255),  # Light blue
        "Linear (frac=0.2)": (120, 255, 120),  # Light green
        "Sigmoid (T=20)": (255, 200, 120),  # Light orange
    }

    # Short names for display
    method_short_names = {
        "Nearest": "Nearest",
        "Bilinear": "Bilinear",
        "Gaussian (clip=0.5)": "Gaussian",
        "Linear (frac=0.2)": "Linear",
        "Sigmoid (T=20)": "Sigmoid",
    }

    # Use a representative stride for mask visualization
    mask_viz_stride = 4.0 if 4.0 in strides else strides[len(strides) // 2]

    for sample_idx in range(num_samples):
        # Calculate y offset for this sample section
        sample_y_start = (
            sample_idx * (rows_per_sample * (cell_size + label_height) + section_padding) + section_padding
        )

        # Draw sample separator
        if sample_idx > 0:
            y_sep = sample_y_start - section_padding // 2
            draw.line([(0, y_sep), (total_width, y_sep)], fill=(100, 100, 100), width=2)

        # Draw sample header
        draw.text((padding, sample_y_start - 18), f"Sample {sample_idx + 1}", fill=(220, 220, 220), font=font_bold)

        # Draw column headers (only for first sample)
        if sample_idx == 0:
            header_y = sample_y_start - 2
            x_base = row_label_width + padding

            draw.text((padding, header_y), "Method", fill=(180, 180, 180), font=font_small)
            draw.text((x_base, header_y), "Input", fill=(180, 180, 180), font=font_small)

            for stride_idx, stride in enumerate(strides):
                x_pos = x_base + (1 + stride_idx) * (cell_size + padding)
                stride_label = f"s={stride:.1f}" if stride != int(stride) else f"s={stride:.0f}"
                draw.text((x_pos, header_y), stride_label, fill=(180, 180, 180), font=font_small)

            # Mask column header
            x_pos = x_base + (1 + num_strides) * (cell_size + padding)
            draw.text((x_pos, header_y), f"Mask (s={mask_viz_stride:.0f})", fill=(180, 180, 180), font=font_small)

        # For each method, draw a row of outputs
        for method_idx, method in enumerate(method_names):
            y_offset = sample_y_start + method_idx * (cell_size + label_height)
            color = method_colors.get(method, (255, 255, 255))
            short_name = method_short_names.get(method, method)

            # Draw method label on left (row label)
            label_y = y_offset + label_height + cell_size // 2 - 6
            draw.text((padding, label_y), short_name, fill=color, font=font_bold)

            # Draw colored indicator bar
            bar_y = y_offset + label_height
            draw.rectangle([(padding, bar_y), (padding + 4, bar_y + cell_size)], fill=color)

            # Column 0: Input (same for all methods, but show for each row)
            x_pos = row_label_width + padding
            input_img = tensor_to_image(inputs[sample_idx], is_rgb=True)
            input_img_display = input_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)
            canvas.paste(input_img_display, (x_pos, y_offset + label_height))

            # Columns 1 to num_strides: Outputs for each stride
            for stride_idx, stride in enumerate(strides):
                x_pos = row_label_width + padding + (1 + stride_idx) * (cell_size + padding)

                output = all_results[method][stride]["output"][sample_idx]
                output_img = tensor_to_image(output, is_rgb=True)
                output_img_display = output_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

                canvas.paste(output_img_display, (x_pos, y_offset + label_height))

            # Last column: Mask visualization (or N/A for baseline methods)
            x_pos = row_label_width + padding + (1 + num_strides) * (cell_size + padding)

            if method in BASELINE_METHODS:
                # Draw "N/A" for baseline methods (no spectral mask)
                draw.text(
                    (x_pos + cell_size // 3, y_offset + label_height + cell_size // 3),
                    "N/A",
                    fill=(100, 100, 100),
                    font=font_bold,
                )
                # Draw empty cell border
                draw.rectangle(
                    [(x_pos, y_offset + label_height), (x_pos + cell_size, y_offset + label_height + cell_size)],
                    outline=(60, 60, 60),
                    width=1,
                )
            else:
                # Draw spectral mask
                mask = all_results[method][mask_viz_stride]["mask"]  # [1, sM_h, sM_w]
                mask_np = mask[0].detach().cpu().numpy()

                # Normalize mask for visualization
                mask_np_viz = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
                mask_img = Image.fromarray((mask_np_viz * 255).astype(np.uint8), mode="L")
                mask_img_display = mask_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

                canvas.paste(mask_img_display.convert("RGB"), (x_pos, y_offset + label_height))

    # Save
    output_path = output_dir / "spectral_mask_type_comparison.png"
    canvas.save(output_path)
    print(f"\n✅ Mask type comparison saved to: {output_path}")

    # Create separate mask visualization
    create_mask_visualization_grid(all_results, strides, output_dir)

    # Also create a 1D cross-section comparison
    create_1d_mask_comparison(strides[2] if len(strides) > 2 else strides[-1], output_dir)


def create_mask_visualization_grid(all_results: dict, strides: List[float], output_dir: Path):
    """Create a separate visualization showing only the spectral masks."""
    print("\nCreating mask visualization grid...")

    spectral_methods = list(SPECTRAL_MASK_TYPES.keys())
    num_methods = len(spectral_methods)
    num_strides = len(strides)

    cell_size = 100
    padding = 10
    label_height = 25
    row_label_width = 100

    total_width = row_label_width + num_strides * (cell_size + padding) + padding
    total_height = label_height + num_methods * (cell_size + padding) + padding

    canvas = Image.new("RGB", (total_width, total_height), color=(40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()

    method_colors = {
        "Gaussian (clip=0.5)": (120, 180, 255),
        "Linear (frac=0.2)": (120, 255, 120),
        "Sigmoid (T=20)": (255, 200, 120),
    }

    # Draw header row with stride labels
    for stride_idx, stride in enumerate(strides):
        x_pos = row_label_width + stride_idx * (cell_size + padding) + padding
        draw.text((x_pos, 5), f"Stride {stride:.0f}", fill=(200, 200, 200), font=font)

    # Draw masks for each method
    for method_idx, method in enumerate(spectral_methods):
        y_pos = label_height + method_idx * (cell_size + padding) + padding
        color = method_colors.get(method, (255, 255, 255))

        # Row label
        short_name = method.split()[0]
        draw.text((padding, y_pos + cell_size // 3), short_name, fill=color, font=font)

        for stride_idx, stride in enumerate(strides):
            x_pos = row_label_width + stride_idx * (cell_size + padding) + padding

            mask = all_results[method][stride]["mask"]  # [1, sM_h, sM_w]
            mask_np = mask[0].detach().cpu().numpy()

            # Normalize mask for visualization
            mask_np_viz = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min() + 1e-8)
            mask_img = Image.fromarray((mask_np_viz * 255).astype(np.uint8), mode="L")
            mask_img_display = mask_img.resize((cell_size, cell_size), Image.Resampling.NEAREST)

            canvas.paste(mask_img_display.convert("RGB"), (x_pos, y_pos))

    output_path = output_dir / "spectral_masks_grid.png"
    canvas.save(output_path)
    print(f"✅ Mask grid saved to: {output_path}")


def create_1d_mask_comparison(stride: float, output_dir: Path):
    """Create 1D cross-section comparison of the different mask types."""
    import matplotlib.pyplot as plt

    print("\nCreating 1D mask cross-section comparison...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"1D Cross-Section of Spectral Masks vs Bilinear (stride={stride:.1f})")

    x_1d = torch.linspace(-1, 1, 200)

    # Colors for each method
    colors = {
        "Bilinear": "red",
        "Gaussian (clip=0.5)": "blue",
        "Linear (frac=0.2)": "green",
        "Sigmoid (T=20)": "orange",
    }

    cutoff = 1.0 / stride

    # Bilinear interpolation equivalent frequency response (sinc-like)
    # Bilinear is approximately a tent/triangular function in spatial domain
    # Its frequency response is sinc^2. We approximate it here.
    # For visualization, we show the ideal low-pass (box in freq domain)
    bilinear_response = (x_1d.abs() <= cutoff).float()
    ax.plot(
        x_1d.numpy(),
        bilinear_response.numpy(),
        label="Bilinear (ideal cutoff)",
        linewidth=2,
        color=colors["Bilinear"],
        linestyle="--",
        alpha=0.7,
    )

    for mask_type, config in SPECTRAL_MASK_TYPES.items():
        mask_class = config["class"]
        mask_kwargs = config["kwargs"]

        # Create mask module
        mask_module = mask_class(
            data_dim=1,
            init_stride_value=stride,
            **mask_kwargs,
        )

        # Compute mask values analytically for visualization
        mask_cutoff = mask_module._compute_cutoff().detach()

        if mask_class == SpectralGaussianMaskND:
            std = mask_cutoff / mask_module._gaussian_cutoff_factor
            y = torch.exp(-0.5 * (x_1d / std).pow(2))
        elif mask_class == SpectralLinearMaskND:
            transition_width = mask_module._get_transition_width(mask_cutoff).detach()
            y = ((mask_cutoff - x_1d.abs()) / transition_width).clamp(min=0.0, max=1.0)
        elif mask_class == SpectralSigmoidMaskND:
            dist_from_cutoff = x_1d.abs() - mask_cutoff
            y = torch.sigmoid(-dist_from_cutoff * mask_module.temperature)

        ax.plot(x_1d.numpy(), y.detach().numpy(), label=mask_type, linewidth=2, color=colors.get(mask_type, "gray"))

    # Draw cutoff line
    ax.axvline(-cutoff, color="gray", linestyle=":", alpha=0.7, label=f"cutoff (±{cutoff:.2f})")
    ax.axvline(cutoff, color="gray", linestyle=":", alpha=0.7)
    ax.axhline(0.5, color="lightgray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Mask Value / Frequency Response")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add note about bilinear
    ax.text(
        0.02,
        0.02,
        "Note: Bilinear shown as ideal cutoff (actual response is sinc²)",
        transform=ax.transAxes,
        fontsize=8,
        color="gray",
        style="italic",
    )

    plt.tight_layout()
    output_path = output_dir / "spectral_mask_type_1d_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 1D comparison saved to: {output_path}")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Spectral Downsampling Tests on EMNIST Spatial Recall")
    print("=" * 80)

    test_spectral_downsampling_on_spatial_recall()
    test_mask_type_comparison()
    test_clip_values()
    test_information_preservation()

    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
