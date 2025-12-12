"""Test hybrid patchify/unpatchify reconstruction capability.

This test uses:
    - SpectralPatchify for downsampling (spectral masking)
    - Unpatchify (ConvTranspose) for upsampling

This helps isolate whether reconstruction issues come from:
    - Spectral downsampling, or
    - Bilinear upsampling

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_hybrid_patchify_reconstruction.py
    PYTHONPATH=. python nvsubquadratic/modules/test_hybrid_patchify_reconstruction.py --num_steps 2000
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import SpectralLinearMaskND
from nvsubquadratic.modules.patchify import SpectralPatchify, Unpatchify
from nvsubquadratic.modules.test_spectral_patchify_reconstruction import (
    TEST_CONFIG,
    get_test_target,
    init_flatten_weights,
    train_reconstruction,
)


class HybridReconstructionNet(nn.Module):
    """Network with spectral patchify + conv-transpose unpatchify."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 4,
        freeze_masks: bool = True,
        init_flatten: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.stride = stride

        # SpectralPatchify with a simple Conv2d pre-processing
        self.patchify = SpectralPatchify(
            in_features=in_channels,
            out_features=hidden_channels,
            data_dim=2,
            spectral_mask_cfg=LazyConfig(SpectralLinearMaskND)(
                data_dim=2,
                transition_fraction=0.1,
                init_stride_value=float(stride),
            ),
            conv_cfg=LazyConfig(nn.Conv2d)(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=8,
                padding="same",
            ),
        )

        # Conv-based Unpatchify (ConvTranspose)
        self.unpatchify = Unpatchify(
            in_features=hidden_channels,
            out_features=in_channels,
            data_dim=2,
            patch_size=stride,
            stride=stride,
        )

        # Freeze spectral masks if requested
        if freeze_masks:
            for param in self.patchify.spectral_mask.parameters():
                param.requires_grad = False

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

        # Step 1: SpectralPatchify (conv + spectral downsampling)
        # Input is BHL format [B, C, H, W]
        x_down, patchify_intermediates = self.patchify(x, is_bhl_input=True, return_intermediates=True)
        intermediates["after_patchify_conv"] = patchify_intermediates.get(
            "after_conv", patchify_intermediates["input"]
        )
        intermediates["spectral_mask"] = patchify_intermediates["spectral_mask"]
        intermediates["after_spectral_down"] = x_down

        # Step 2: Conv-based Unpatchify (ConvTranspose)
        # Unpatchify expects channels-last [B, H, W, C], so convert
        x_down_bhwc = rearrange(x_down, "b c h w -> b h w c")
        x_up = self.unpatchify(x_down_bhwc, output_spatial_shape=(H, W))

        # Convert back to BHL format
        x_out = rearrange(x_up, "b h w c -> b c h w")
        intermediates["after_unpatchify"] = x_out
        intermediates["output"] = x_out

        return x_out, intermediates


def main(num_steps: int | None = None, init_flatten: bool = False):
    print("=" * 70)
    print("Hybrid Reconstruction Test (Spectral Patchify + Conv Unpatchify)")
    print("=" * 70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Use shared configuration
    batch_size = TEST_CONFIG["batch_size"]
    in_channels = TEST_CONFIG["in_channels"]
    hidden_channels = TEST_CONFIG["hidden_channels"]
    H, W = TEST_CONFIG["H"], TEST_CONFIG["W"]
    stride = TEST_CONFIG["stride"]
    if num_steps is None:
        num_steps = TEST_CONFIG["num_steps"]

    print("\nConfiguration (shared with other tests):")
    print(f"  Input shape: [{batch_size}, {in_channels}, {H}, {W}]")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Stride: {stride}")
    print(f"  Downsampled shape: [{batch_size}, {hidden_channels}, {int(H / stride)}, {int(W / stride)}]")
    print(f"  Training steps: {num_steps}")
    print(f"  Init flatten: {init_flatten}")
    print("\n  Patchify: SpectralPatchify (spectral downsampling)")
    print("  Unpatchify: Unpatchify (ConvTranspose upsampling)")

    # Create random target using shared function
    target = get_test_target(device=device)
    print(f"\nTarget tensor: shape={target.shape}, mean={target.mean():.4f}, std={target.std():.4f}")

    # Create model
    model = HybridReconstructionNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        stride=stride,
        freeze_masks=True,
        init_flatten=init_flatten,
    ).to(device)

    # Print model info
    print("\nModel:")
    print(f"  Patchify: {model.patchify}")
    print(f"  Unpatchify: {model.unpatchify}")

    # Check mask is frozen
    mask_params = list(model.patchify.spectral_mask.parameters())
    print(f"\nSpectral mask frozen: {not any(p.requires_grad for p in mask_params)}")
    print(f"  Mask stride: {model.patchify.spectral_mask.get_stride().tolist()}")

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
        fig.suptitle(f"Hybrid: Spectral Patchify + Conv Unpatchify (stride={stride})", fontsize=14)

        # Row 1: Input -> Downsampled -> Upsampled -> Output
        sample_idx = 0
        channel_idx = 0

        axes[0, 0].imshow(target[sample_idx, channel_idx].cpu().numpy(), cmap="viridis")
        axes[0, 0].set_title(f"Target (ch={channel_idx})")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(intermediates["after_spectral_down"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[0, 1].set_title(f"After spectral down ({int(H / stride)}x{int(W / stride)})")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(intermediates["after_unpatchify"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[0, 2].set_title(f"After conv unpatchify ({H}x{W})")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(output[sample_idx, channel_idx].cpu().numpy(), cmap="viridis")
        axes[0, 3].set_title(f"Output (ch={channel_idx})")
        axes[0, 3].axis("off")

        # Row 2: Error map, loss curve, spectral mask, histogram
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

        mask = intermediates["spectral_mask"][0, 0].cpu().numpy()
        axes[1, 2].imshow(mask, cmap="viridis", aspect="auto")
        axes[1, 2].set_title(f"Spectral mask ({mask.shape[0]}x{mask.shape[1]})")
        axes[1, 2].axis("off")

        axes[1, 3].hist(target.cpu().numpy().flatten(), bins=50, alpha=0.5, label="Target", density=True)
        axes[1, 3].hist(output.cpu().numpy().flatten(), bins=50, alpha=0.5, label="Output", density=True)
        axes[1, 3].set_xlabel("Value")
        axes[1, 3].set_ylabel("Density")
        axes[1, 3].set_title("Value distribution")
        axes[1, 3].legend()

        plt.tight_layout()
        output_path = "nvsubquadratic/modules/hybrid_reconstruction_test.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved visualization to: {output_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available, skipping visualization")

    # Summary
    print("\n" + "=" * 70)
    if final_loss < 0.01:
        print("✅ SUCCESS: Network can reconstruct input through hybrid pipeline!")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    elif final_loss < 0.1:
        print("⚠️ PARTIAL SUCCESS: Reconstruction is reasonable but not perfect")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    else:
        print("❌ FAILED: Hybrid pipeline loses too much information")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    print("=" * 70)

    return final_loss, correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hybrid patchify/unpatchify reconstruction")
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
