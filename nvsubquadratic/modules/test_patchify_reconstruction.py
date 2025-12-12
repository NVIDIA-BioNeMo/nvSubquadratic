"""Test patchify/unpatchify reconstruction capability (Conv-based).

This test verifies that conv-based downsampling doesn't lose critical information
by training a simple network to reconstruct its input through:
    Input → Patchify (stride=4) → Unpatchify → Output

This is a baseline comparison for the spectral patchify/unpatchify test.

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_patchify_reconstruction.py
    PYTHONPATH=. python nvsubquadratic/modules/test_patchify_reconstruction.py --num_steps 2000
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.test_spectral_patchify_reconstruction import (
    TEST_CONFIG,
    get_test_target,
    init_flatten_weights,
    train_reconstruction,
)


class PatchifyReconstructionNet(nn.Module):
    """Network that patchifies then unpatchifies to test reconstruction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        patch_size: int = 4,
        init_flatten: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size

        # Patchify with Conv-based downsampling (non-overlapping patches)
        self.patchify = Patchify(
            in_features=in_channels,
            out_features=hidden_channels,
            data_dim=2,
            patch_size=patch_size,
            stride=patch_size,  # Non-overlapping
        )

        # Unpatchify with ConvTranspose-based upsampling
        self.unpatchify = Unpatchify(
            in_features=hidden_channels,
            out_features=in_channels,
            data_dim=2,
            patch_size=patch_size,
            stride=patch_size,
        )

        # Initialize conv weights to flatten/unflatten if requested
        if init_flatten:
            init_flatten_weights(self.patchify.conv, is_transpose=False)
            init_flatten_weights(self.unpatchify.deconv, is_transpose=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass with intermediate outputs for debugging.

        Args:
            x: Input tensor [B, C, H, W] (BHL format)

        Returns:
            Tuple of (output, intermediates_dict)
        """
        intermediates = {"input": x}
        B, C, H, W = x.shape

        # Convert from BHL (B, C, H, W) to channels-last (B, H, W, C) for Patchify
        x_bhwc = rearrange(x, "b c h w -> b h w c")

        # Step 1: Patchify (conv-based downsampling)
        x_down = self.patchify(x_bhwc)
        intermediates["after_patchify"] = rearrange(x_down, "b h w c -> b c h w")

        # Step 2: Unpatchify (conv-transpose-based upsampling)
        x_up = self.unpatchify(x_down, output_spatial_shape=(H, W))

        # Convert back to BHL format
        x_out = rearrange(x_up, "b h w c -> b c h w")
        intermediates["after_unpatchify"] = x_out
        intermediates["output"] = x_out

        return x_out, intermediates


def main(num_steps: int | None = None, init_flatten: bool = False):
    print("=" * 70)
    print("Patchify/Unpatchify Reconstruction Test (Conv-based baseline)")
    print("=" * 70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Use shared configuration (same as spectral test)
    batch_size = TEST_CONFIG["batch_size"]
    in_channels = TEST_CONFIG["in_channels"]
    H, W = TEST_CONFIG["H"], TEST_CONFIG["W"]
    patch_size = TEST_CONFIG["stride"]  # Use stride as patch_size for equivalence
    # Use enough hidden channels to avoid bottleneck: patch_size^2 * in_channels
    hidden_channels = patch_size * patch_size * in_channels  # 4*4*3 = 48
    if num_steps is None:
        num_steps = TEST_CONFIG["num_steps"]

    print("\nConfiguration (shared with spectral test):")
    print(f"  Input shape: [{batch_size}, {in_channels}, {H}, {W}]")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Patch size (stride): {patch_size}")
    print(f"  Downsampled shape: [{batch_size}, {hidden_channels}, {int(H / patch_size)}, {int(W / patch_size)}]")
    print(f"  Training steps: {num_steps}")
    print(f"  Init flatten: {init_flatten}")

    # Create random target using shared function (same data as spectral test)
    target = get_test_target(device=device)
    print(f"\nTarget tensor: shape={target.shape}, mean={target.mean():.4f}, std={target.std():.4f}")

    # Create model
    model = PatchifyReconstructionNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        patch_size=patch_size,
        init_flatten=init_flatten,
    ).to(device)

    # Print model info
    print("\nModel:")
    print(f"  Patchify: {model.patchify}")
    print(f"  Unpatchify: {model.unpatchify}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")

    # Initial forward pass to check shapes
    print("\n" + "-" * 70)
    print("Initial forward pass:")
    print("-" * 70)
    with torch.no_grad():
        output, intermediates = model(target)
        for name, tensor in intermediates.items():
            print(f"  {name:25s}: {tuple(tensor.shape)}")

        initial_loss = F.mse_loss(output, target).item()
        print(f"\nInitial MSE loss: {initial_loss:.6f}")

    # Train
    print("\n" + "-" * 70)
    print("Training:")
    print("-" * 70)
    losses = train_reconstruction(model, target, num_steps=num_steps, lr=1e-3, print_every=50)

    # Final evaluation
    print("\n" + "-" * 70)
    print("Final evaluation:")
    print("-" * 70)
    with torch.no_grad():
        output, intermediates = model(target)
        final_loss = F.mse_loss(output, target).item()

        # Per-pixel metrics
        abs_error = (output - target).abs()
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()

        # Correlation
        target_flat = target.flatten()
        output_flat = output.flatten()
        correlation = torch.corrcoef(torch.stack([target_flat, output_flat]))[0, 1].item()

        print(f"Final MSE loss: {final_loss:.6f}")
        print(f"Mean absolute error: {mean_error:.6f}")
        print(f"Max absolute error: {max_error:.6f}")
        print(f"Correlation: {correlation:.6f}")
        print(f"Loss reduction: {initial_loss / final_loss:.1f}x")

    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Patchify/Unpatchify Reconstruction (patch_size={patch_size})", fontsize=14)

        # Row 1: Input -> Downsampled -> Upsampled -> Output
        sample_idx = 0
        channel_idx = 0

        axes[0, 0].imshow(target[sample_idx, channel_idx].cpu().numpy(), cmap="viridis")
        axes[0, 0].set_title(f"Target (ch={channel_idx})")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(intermediates["after_patchify"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[0, 1].set_title(f"After patchify ({int(H / patch_size)}x{int(W / patch_size)})")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(intermediates["after_unpatchify"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[0, 2].set_title(f"After unpatchify ({H}x{W})")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(output[sample_idx, channel_idx].cpu().numpy(), cmap="viridis")
        axes[0, 3].set_title(f"Output (ch={channel_idx})")
        axes[0, 3].axis("off")

        # Row 2: Error map, loss curve, placeholder, histogram
        error_map = (output - target)[sample_idx, channel_idx].abs().cpu().numpy()
        im = axes[1, 0].imshow(error_map, cmap="hot")
        axes[1, 0].set_title(f"Absolute error (max={max_error:.4f})")
        axes[1, 0].axis("off")
        plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

        axes[1, 1].plot(losses)
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("MSE Loss")
        axes[1, 1].set_title("Training curve")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        # Placeholder for spectral mask (not used in conv-based)
        axes[1, 2].text(0.5, 0.5, "N/A\n(Conv-based)", ha="center", va="center", fontsize=12)
        axes[1, 2].set_title("Spectral mask (N/A)")
        axes[1, 2].axis("off")

        axes[1, 3].hist(target.cpu().numpy().flatten(), bins=50, alpha=0.5, label="Target", density=True)
        axes[1, 3].hist(output.cpu().numpy().flatten(), bins=50, alpha=0.5, label="Output", density=True)
        axes[1, 3].set_xlabel("Value")
        axes[1, 3].set_ylabel("Density")
        axes[1, 3].set_title("Value distribution")
        axes[1, 3].legend()

        plt.tight_layout()
        output_path = "nvsubquadratic/modules/patchify_reconstruction_test.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved visualization to: {output_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available, skipping visualization")

    # Summary
    print("\n" + "=" * 70)
    if final_loss < 0.01:
        print("✅ SUCCESS: Network can reconstruct input through conv-based downsampling!")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    elif final_loss < 0.1:
        print("⚠️ PARTIAL SUCCESS: Reconstruction is reasonable but not perfect")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    else:
        print("❌ FAILED: Conv-based downsampling loses too much information")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    print("=" * 70)

    return final_loss, correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test conv-based patchify/unpatchify reconstruction")
    parser.add_argument(
        "--num_steps", type=int, default=None, help=f"Number of training steps (default: {TEST_CONFIG['num_steps']})"
    )
    parser.add_argument(
        "--init_flatten",
        action="store_true",
        help="Initialize conv weights to flatten/unflatten patches into channels",
    )
    args = parser.parse_args()
    main(num_steps=args.num_steps, init_flatten=args.init_flatten)
