# TODO: Add license header here

"""Tests for spatial modulation kernel size initialization.

Tests GaussianModulationND, TrapezoidModulationND, and SigmoidModulationND:
1. Kernel size initialization and get_kernel_size() accuracy
2. Boundary behavior (mask value at kernel_size radius)
3. get_kernel_size_pixels() conversion
4. Visual comparison of all three mask types

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_gaussian_modulation_kernel_size.py

Or with pytest:
    PYTHONPATH=. pytest nvsubquadratic/modules/test_gaussian_modulation_kernel_size.py -v
"""

import importlib.util
import math
from pathlib import Path

import torch

from nvsubquadratic.modules.masks_nd import (
    GaussianModulationND,
    SigmoidModulationND,
    TrapezoidModulationND,
)


HAS_PYTEST = importlib.util.find_spec("pytest") is not None


def create_grid(grid_size: int, data_dim: int = 2) -> torch.Tensor:
    """Create a normalized grid for testing."""
    linspace = torch.linspace(-1, 1, grid_size)
    if data_dim == 1:
        grid = linspace.unsqueeze(-1)  # [grid_size, 1]
    else:
        grid = torch.stack(torch.meshgrid(*[linspace] * data_dim, indexing="ij"), dim=-1)  # [grid_size, ..., data_dim]
    return grid.unsqueeze(0)  # [1, grid_size, ..., data_dim]


def test_gaussian_kernel_size_boundary():
    """Test that Gaussian mask = clip_value at the kernel_size boundary."""
    print("\n" + "=" * 70)
    print("Test: Gaussian mask = clip_value at kernel_size boundary")
    print("=" * 70)

    grid_size = 65
    kernel_size = 0.25
    clip_value = 0.1

    grid = create_grid(grid_size, data_dim=2)
    linspace = torch.linspace(-1, 1, grid_size)

    modulator = GaussianModulationND(
        data_dim=2,
        num_channels=1,
        init_kernel_size_low=kernel_size,
        init_kernel_size_high=kernel_size,
        clip_value=clip_value,
    )

    mask = modulator(grid, None)[0, :, :, 0].detach()

    # Find boundary index
    boundary_idx = (linspace - kernel_size).abs().argmin().item()
    center = grid_size // 2

    center_value = mask[center, center].item()
    boundary_value = mask[center, boundary_idx].item()

    print(f"  kernel_size: {kernel_size}, clip_value: {clip_value}")
    print(f"  Center value: {center_value:.4f} (expected 1.0)")
    print(f"  Value at x={kernel_size}: {boundary_value:.4f} (expected ~{clip_value})")

    assert abs(center_value - 1.0) < 1e-4, f"Center should be 1.0, got {center_value}"
    assert abs(boundary_value - clip_value) < 0.02, f"Boundary should be ~{clip_value}, got {boundary_value}"

    print("  ✅ Gaussian boundary test passed!")


def test_trapezoid_kernel_size_boundary():
    """Test that Trapezoid mask = 0 at the kernel_size boundary."""
    print("\n" + "=" * 70)
    print("Test: Trapezoid mask = 0 at kernel_size boundary")
    print("=" * 70)

    grid_size = 65
    kernel_size = 0.25
    transition_fraction = 0.5

    grid = create_grid(grid_size, data_dim=2)
    linspace = torch.linspace(-1, 1, grid_size)

    modulator = TrapezoidModulationND(
        data_dim=2,
        num_channels=1,
        init_kernel_size_low=kernel_size,
        init_kernel_size_high=kernel_size,
        transition_fraction=transition_fraction,
    )

    mask = modulator(grid, None)[0, :, :, 0].detach()

    # Find boundary index
    boundary_idx = (linspace - kernel_size).abs().argmin().item()
    center = grid_size // 2

    # Inner edge (where mask = 1)
    inner_edge = kernel_size * (1 - transition_fraction)
    inner_idx = (linspace - inner_edge).abs().argmin().item()

    center_value = mask[center, center].item()
    inner_value = mask[center, inner_idx].item()
    boundary_value = mask[center, boundary_idx].item()

    print(f"  kernel_size: {kernel_size}, transition_fraction: {transition_fraction}")
    print(f"  Inner edge (mask=1) at x={inner_edge:.3f}")
    print(f"  Center value: {center_value:.4f} (expected 1.0)")
    print(f"  Value at inner edge x={inner_edge:.3f}: {inner_value:.4f} (expected ~1.0)")
    print(f"  Value at outer edge x={kernel_size}: {boundary_value:.4f} (expected 0.0)")

    assert abs(center_value - 1.0) < 1e-4, f"Center should be 1.0, got {center_value}"
    assert abs(boundary_value - 0.0) < 0.02, f"Boundary should be 0.0, got {boundary_value}"

    print("  ✅ Trapezoid boundary test passed!")


def test_sigmoid_kernel_size_boundary():
    """Test that Sigmoid mask = 0.5 at the kernel_size boundary."""
    print("\n" + "=" * 70)
    print("Test: Sigmoid mask = 0.5 at kernel_size boundary")
    print("=" * 70)

    grid_size = 65
    kernel_size = 0.25
    temperature = 20.0  # Higher temperature for sharper transition

    grid = create_grid(grid_size, data_dim=2)
    linspace = torch.linspace(-1, 1, grid_size)

    modulator = SigmoidModulationND(
        data_dim=2,
        num_channels=1,
        init_kernel_size_low=kernel_size,
        init_kernel_size_high=kernel_size,
        temperature=temperature,
    )

    mask = modulator(grid, None)[0, :, :, 0].detach()

    # Find boundary index
    boundary_idx = (linspace - kernel_size).abs().argmin().item()
    center = grid_size // 2

    center_value = mask[center, center].item()
    boundary_value = mask[center, boundary_idx].item()

    # For 2D, boundary_value is product of two sigmoids, each at 0.5 boundary
    # At (0, kernel_size): one sigmoid at 0.5, one at ~1.0
    # So we expect ~0.5 * 1.0 = 0.5 along the axis
    expected_boundary = 0.5

    print(f"  kernel_size: {kernel_size}, temperature: {temperature}")
    print(f"  Center value: {center_value:.4f} (expected ~1.0)")
    print(f"  Value at x={kernel_size}: {boundary_value:.4f} (expected ~{expected_boundary})")

    # For high temperature, center should be close to 1
    assert center_value > 0.95, f"Center should be close to 1.0, got {center_value}"
    # Boundary should be close to 0.5
    assert abs(boundary_value - expected_boundary) < 0.1, (
        f"Boundary should be ~{expected_boundary}, got {boundary_value}"
    )

    print("  ✅ Sigmoid boundary test passed!")


def test_get_kernel_size_consistency():
    """Test that get_kernel_size() returns the initialized kernel size."""
    print("\n" + "=" * 70)
    print("Test: get_kernel_size() returns initialized values")
    print("=" * 70)

    init_kernel_size_low = 0.1
    init_kernel_size_high = 0.5
    num_channels = 4

    # Expected kernel sizes (log-spaced)
    expected = torch.logspace(math.log10(init_kernel_size_low), math.log10(init_kernel_size_high), num_channels)

    print(f"  Expected kernel sizes: {expected.tolist()}")

    for name, modulator in [
        (
            "Gaussian",
            GaussianModulationND(
                data_dim=2,
                num_channels=num_channels,
                init_kernel_size_low=init_kernel_size_low,
                init_kernel_size_high=init_kernel_size_high,
                clip_value=0.1,
            ),
        ),
        (
            "Trapezoid",
            TrapezoidModulationND(
                data_dim=2,
                num_channels=num_channels,
                init_kernel_size_low=init_kernel_size_low,
                init_kernel_size_high=init_kernel_size_high,
            ),
        ),
        (
            "Sigmoid",
            SigmoidModulationND(
                data_dim=2,
                num_channels=num_channels,
                init_kernel_size_low=init_kernel_size_low,
                init_kernel_size_high=init_kernel_size_high,
            ),
        ),
    ]:
        actual = modulator.get_kernel_size()[0]  # First dimension
        print(f"  {name}: {actual.tolist()}")

        assert torch.allclose(actual, expected, rtol=1e-4), f"{name} kernel sizes don't match: {actual} vs {expected}"

    print("  ✅ get_kernel_size() consistency test passed!")


def test_get_kernel_size_pixels():
    """Test that get_kernel_size_pixels() correctly converts to pixel units."""
    print("\n" + "=" * 70)
    print("Test: get_kernel_size_pixels() conversion")
    print("=" * 70)

    kernel_size = 0.25  # 25% of half-grid = boundary at 0.25
    grid_sizes = [65, 129, 33]

    for name, modulator in [
        (
            "Gaussian",
            GaussianModulationND(
                data_dim=2,
                num_channels=1,
                init_kernel_size_low=kernel_size,
                init_kernel_size_high=kernel_size,
                clip_value=0.1,
            ),
        ),
        (
            "Trapezoid",
            TrapezoidModulationND(
                data_dim=2,
                num_channels=1,
                init_kernel_size_low=kernel_size,
                init_kernel_size_high=kernel_size,
            ),
        ),
        (
            "Sigmoid",
            SigmoidModulationND(
                data_dim=2,
                num_channels=1,
                init_kernel_size_low=kernel_size,
                init_kernel_size_high=kernel_size,
            ),
        ),
    ]:
        print(f"  {name}:")
        for gs in grid_sizes:
            pixel_size = modulator.get_kernel_size_pixels(gs)[0, 0].item()
            # For grid [-1, 1] with gs points, spacing = 2/(gs-1)
            # kernel_size in pixels = kernel_size * (gs - 1)
            expected_pixels = kernel_size * (gs - 1)
            print(f"    Grid {gs}: {pixel_size:.2f} pixels (expected {expected_pixels:.1f})")
            assert abs(pixel_size - expected_pixels) < 0.1, f"Pixel size mismatch: {pixel_size} vs {expected_pixels}"

    print("  ✅ get_kernel_size_pixels() test passed!")


def test_min_max_kernel_size_clamping():
    """Test that min/max kernel size bounds are enforced."""
    print("\n" + "=" * 70)
    print("Test: min/max kernel_size clamping")
    print("=" * 70)

    grid = create_grid(65, data_dim=2)

    for name, modulator in [
        (
            "Gaussian",
            GaussianModulationND(
                data_dim=2,
                num_channels=1,
                init_kernel_size_low=0.001,  # Very small, should be clamped
                init_kernel_size_high=0.001,
                clip_value=0.1,
                min_kernel_size=0.1,  # Minimum is 0.1
            ),
        ),
        (
            "Trapezoid",
            TrapezoidModulationND(
                data_dim=2,
                num_channels=1,
                init_kernel_size_low=0.001,
                init_kernel_size_high=0.001,
                min_kernel_size=0.1,
            ),
        ),
        (
            "Sigmoid",
            SigmoidModulationND(
                data_dim=2,
                num_channels=1,
                init_kernel_size_low=0.001,
                init_kernel_size_high=0.001,
                min_kernel_size=0.1,
            ),
        ),
    ]:
        # Force forward to trigger clamping
        _ = modulator(grid, None)
        actual_ks = modulator.get_kernel_size()[0, 0].item()
        print(f"  {name}: init=0.001, min=0.1 -> actual={actual_ks:.4f}")
        assert actual_ks >= 0.1 - 1e-6, f"{name} should clamp to min, got {actual_ks}"

    print("  ✅ min/max clamping test passed!")


def test_visual_comparison():
    """Create visual comparison of all three mask types."""
    print("\n" + "=" * 70)
    print("Test: Visual comparison of all mask types")
    print("=" * 70)

    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ matplotlib not available, skipping visual test")
        return

    output_dir = Path(__file__).parent
    grid_size = 65
    kernel_sizes = [0.125, 0.25, 0.5, 0.75]
    clip_value = 0.1

    grid = create_grid(grid_size, data_dim=2)
    linspace = torch.linspace(-1, 1, grid_size)

    # 2D masks plot
    fig, axes = plt.subplots(3, len(kernel_sizes), figsize=(4 * len(kernel_sizes), 12))
    fig.suptitle(
        "Spatial Modulation Masks: Gaussian vs Trapezoid vs Sigmoid\n"
        "kernel_size = boundary radius in normalized coords [-1, 1]",
        fontsize=14,
        fontweight="bold",
    )

    mask_configs = [
        ("Gaussian", lambda ks: GaussianModulationND(2, 1, ks, ks, clip_value=clip_value)),
        ("Trapezoid", lambda ks: TrapezoidModulationND(2, 1, ks, ks, transition_fraction=0.5)),
        ("Sigmoid", lambda ks: SigmoidModulationND(2, 1, ks, ks, temperature=20.0)),
    ]

    for row, (name, factory) in enumerate(mask_configs):
        for col, ks in enumerate(kernel_sizes):
            modulator = factory(ks)
            mask = modulator(grid, None)[0, :, :, 0].detach().cpu().numpy()

            ax = axes[row, col]
            im = ax.imshow(mask, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis", vmin=0, vmax=1)

            # Draw boundary circle
            circle = patches.Circle((0, 0), ks, fill=False, edgecolor="red", linestyle="--", linewidth=2)
            ax.add_patch(circle)

            ax.set_title(f"{name}, ks={ks}")
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect("equal")

            if col == 0:
                ax.set_ylabel(name)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = output_dir / "spatial_modulation_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved 2D comparison to: {output_path}")

    # 1D cross-section plot
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle("1D Cross-sections of Spatial Modulation Masks (through center)", fontsize=14, fontweight="bold")

    x_coords = linspace.numpy()

    for ax, (name, factory) in zip(axes2, mask_configs):
        for ks in kernel_sizes:
            modulator = factory(ks)
            mask = modulator(grid, None)[0, grid_size // 2, :, 0].detach().cpu().numpy()
            ax.plot(x_coords, mask, label=f"ks={ks}", linewidth=2)

            # Mark boundary
            ax.axvline(x=ks, color="gray", linestyle=":", alpha=0.5)
            ax.axvline(x=-ks, color="gray", linestyle=":", alpha=0.5)

        ax.set_title(name)
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("mask value")
        ax.legend(loc="upper right")
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # Add reference lines for boundary conditions
        if name == "Gaussian":
            ax.axhline(y=clip_value, color="red", linestyle="--", alpha=0.7, label=f"clip={clip_value}")
        elif name == "Trapezoid":
            ax.axhline(y=0.0, color="red", linestyle="--", alpha=0.7)
        else:  # Sigmoid
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="0.5")

    plt.tight_layout()
    output_path2 = output_dir / "spatial_modulation_cross_section.png"
    fig2.savefig(output_path2, dpi=150, bbox_inches="tight")
    print(f"  Saved cross-sections to: {output_path2}")

    plt.close("all")
    print("  ✅ Visual comparison test complete!")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("Spatial Modulation Kernel Size Tests")
    print("(GaussianModulationND, TrapezoidModulationND, SigmoidModulationND)")
    print("=" * 70)

    test_gaussian_kernel_size_boundary()
    test_trapezoid_kernel_size_boundary()
    test_sigmoid_kernel_size_boundary()
    test_get_kernel_size_consistency()
    test_get_kernel_size_pixels()
    test_min_max_kernel_size_clamping()
    test_visual_comparison()

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
