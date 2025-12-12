"""Test dual-path patchify/unpatchify reconstruction capability.

This test combines:
    - Spectral path: SpectralPatchify (learnable stride, anti-aliased low-freq)
    - Spatial path: Strided Conv (uses spectral stride, aliased high-freq)

The spatial path reads the stride from the spectral path and uses it for
strided convolution, providing the aliased high-frequency content that
spectral downsampling loses.

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_reconstruction.py
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_reconstruction.py --num_steps 2000
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import SpectralLinearMaskND
from nvsubquadratic.modules.patchify import SpectralPatchify, SpectralUnpatchify
from nvsubquadratic.modules.test_spectral_patchify_reconstruction import (
    TEST_CONFIG,
    get_test_target,
)


class DualPathReconstructionNet(nn.Module):
    """Network with spectral + spatial dual-path patchify/unpatchify."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 4,
        max_stride: int = 16,  # Maximum stride for architecture definition
        freeze_masks: bool = True,
        init_flatten: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.stride = stride
        self.max_stride = max_stride

        # Both paths use full hidden channels (will be summed, not concatenated)
        self.spectral_channels = hidden_channels
        self.spatial_channels = hidden_channels

        # === SPECTRAL PATH ===
        # SpectralPatchify with learnable stride
        self.spectral_patchify = SpectralPatchify(
            in_features=in_channels,
            out_features=self.spectral_channels,
            data_dim=2,
            spectral_mask_cfg=LazyConfig(SpectralLinearMaskND)(
                data_dim=2,
                transition_fraction=0.1,
                init_stride_value=float(stride),
                max_stride_value=float(max_stride),  # Limit max stride
            ),
            conv_cfg=LazyConfig(nn.Conv2d)(
                in_channels=in_channels,
                out_channels=self.spectral_channels,
                kernel_size=6,
                padding="same",
            ),
        )

        # SpectralUnpatchify for spectral path
        self.spectral_unpatchify = SpectralUnpatchify(
            in_features=self.spectral_channels,
            out_features=in_channels,
            data_dim=2,
            output_proj_cfg=LazyConfig(nn.Conv2d)(
                in_channels=self.spectral_channels,
                out_channels=in_channels,
                kernel_size=6,
                padding="same",
            ),
            interpolation_mode="bilinear",
        )

        # === SPATIAL PATH ===
        # Normal conv (stride=1, padding=0) for patchify, followed by explicit subsampling
        # Use max_stride for kernel size to handle any stride up to max_stride
        self.spatial_patchify_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.spatial_channels,
            kernel_size=max_stride,  # Use max_stride as kernel size
            stride=1,
            padding=0,  # No padding - aligns with strided conv behavior
        )

        # PixelShuffle-based spatial unpatchify:
        # Use max_stride for PixelShuffle to handle any stride up to max_stride
        # 1. Conv to expand channels: hidden -> in_channels * max_stride^2
        # 2. PixelShuffle to rearrange channels to spatial dimensions
        self.spatial_unpatchify_expand = nn.Conv2d(
            in_channels=self.spatial_channels,
            out_channels=in_channels * max_stride * max_stride,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.spatial_unpatchify_shuffle = nn.PixelShuffle(max_stride)

        # Freeze spectral masks if requested
        if freeze_masks:
            for param in self.spectral_patchify.spectral_mask.parameters():
                param.requires_grad = False

    def get_stride(self) -> torch.Tensor:
        """Get the current stride from the spectral path."""
        return self.spectral_patchify.spectral_mask.get_stride()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass with dual-path processing.

        Args:
            x: Input tensor [B, C, H, W] (BHL format)

        Returns:
            Tuple of (output, intermediates_dict)
        """
        intermediates = {"input": x}
        B, C, H, W = x.shape

        # Get current stride from spectral path
        current_stride = self.get_stride()
        intermediates["stride"] = current_stride

        # === SPECTRAL PATH (forward) ===
        x_spectral, spectral_intermediates = self.spectral_patchify(x, is_bhl_input=True, return_intermediates=True)
        intermediates["spectral_down"] = x_spectral
        intermediates["spectral_mask"] = spectral_intermediates["spectral_mask"]

        # === SPATIAL PATH (forward) ===
        # Step 1: Apply conv (stride=1, same padding)
        x_spatial_conv = self.spatial_patchify_conv(x)
        intermediates["spatial_after_conv"] = x_spatial_conv

        # Step 2: Subsample by selecting every Nth value (strided indexing)
        # Use per-dimension strides from spectral path
        stride_h = max(1, round(current_stride[0].item()))
        stride_w = max(1, round(current_stride[1].item()))
        x_spatial = x_spatial_conv[:, :, ::stride_h, ::stride_w]
        intermediates["spatial_down"] = x_spatial

        # Ensure spatial output matches spectral output size
        if x_spatial.shape[2:] != x_spectral.shape[2:]:
            x_spatial = F.interpolate(x_spatial, size=x_spectral.shape[2:], mode="nearest")
            intermediates["spatial_down_resized"] = x_spatial

        # Combine paths (SUM instead of concatenation)
        x_combined = x_spectral + x_spatial
        intermediates["combined"] = x_combined

        # === UNPATCHIFY ===
        # Spectral unpatchify on the combined representation
        x_spectral_up = self.spectral_unpatchify(x_combined, target_shape=(H, W), is_bhl_input=True)
        intermediates["spectral_up"] = x_spectral_up

        # Spatial unpatchify: PixelShuffle-based upsampling
        # Expand channels and rearrange to spatial dimensions
        x_spatial_up = self.spatial_unpatchify_expand(x_combined)
        x_spatial_up = self.spatial_unpatchify_shuffle(x_spatial_up)
        # Final resize if needed (handles stride mismatch)
        if x_spatial_up.shape[2:] != (H, W):
            x_spatial_up = F.interpolate(x_spatial_up, size=(H, W), mode="nearest")
        intermediates["spatial_up"] = x_spatial_up

        # Combine reconstructions (simple addition)
        x_out = x_spectral_up + x_spatial_up
        intermediates["output"] = x_out

        return x_out, intermediates


def main(num_steps: int | None = None, init_flatten: bool = False, learn_stride: bool = False):
    print("=" * 70)
    print("Dual-Path Reconstruction Test (Spectral + Spatial)")
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
    print(f"  Hidden channels: {hidden_channels} (both paths use full channels, summed)")
    print(f"  Stride: {stride}")
    print(f"  Downsampled shape: [{batch_size}, {hidden_channels}, {int(H / stride)}, {int(W / stride)}]")
    print(f"  Training steps: {num_steps}")
    print(f"  Init flatten (spatial path): {init_flatten}")
    print(f"  Learn stride: {learn_stride}")

    # Create random target using shared function
    target = get_test_target(device=device)
    print(f"\nTarget tensor: shape={target.shape}, mean={target.mean():.4f}, std={target.std():.4f}")

    # Create model
    max_stride = 16  # Maximum stride the architecture can handle
    model = DualPathReconstructionNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        stride=stride,
        max_stride=max_stride,
        freeze_masks=not learn_stride,  # Unfreeze if learning stride
        init_flatten=init_flatten,
    ).to(device)

    # Print model info
    print("\nModel:")
    print(f"  Spectral patchify: {model.spectral_patchify}")
    print(f"  Spatial patchify: {model.spatial_patchify_conv}")
    print(f"  Spectral unpatchify: {model.spectral_unpatchify}")
    print(f"  Spatial unpatchify: {model.spatial_unpatchify_expand} + PixelShuffle({max_stride})")

    # Check mask is frozen and record initial stride
    mask_params = list(model.spectral_patchify.spectral_mask.parameters())
    print(f"\nSpectral mask frozen: {not any(p.requires_grad for p in mask_params)}")
    initial_stride = model.get_stride().clone().detach()
    print(f"  Initial stride: {initial_stride.tolist()}")

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
            if isinstance(tensor, torch.Tensor):
                print(f"  {name:25s}: {tuple(tensor.shape)}")

        initial_loss = F.mse_loss(output, target).item()
        print(f"\nInitial MSE loss: {initial_loss:.6f}")

    # Train with stride gradient tracking
    print("\n" + "-" * 70)
    print("Training:")
    print("-" * 70)

    # Custom training loop to track stride gradients
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    stride_grads = []

    for step in range(num_steps):
        optimizer.zero_grad()

        output, _ = model(target)
        loss = F.mse_loss(output, target)

        loss.backward()

        # Track stride gradients
        if learn_stride:
            stride_param = next(iter(model.spectral_patchify.spectral_mask.parameters()))
            if stride_param.grad is not None:
                stride_grads.append(stride_param.grad.clone().detach().mean().item())
            else:
                stride_grads.append(0.0)

        optimizer.step()

        losses.append(loss.item())

        if step % 50 == 0 or step == num_steps - 1:
            stride_info = ""
            if learn_stride and stride_grads:
                stride_info = f", stride_grad={stride_grads[-1]:.6f}"
            print(f"Step {step:4d}: Loss = {loss.item():.6f}{stride_info}")

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

        # Print stride change if learning stride
        final_stride = model.get_stride()
        print(f"\nStride: {initial_stride.tolist()} -> {final_stride.tolist()}")
        stride_change = (final_stride - initial_stride).abs().mean().item()
        print(f"Stride change (mean abs): {stride_change:.6f}")

    # Visualize if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Dual-Path Reconstruction (Spectral + Spatial, stride={stride})", fontsize=14)

        # Row 1: Target, Spectral down, Spatial down, Combined, Output
        sample_idx = 0
        channel_idx = 0

        axes[0, 0].imshow(target[sample_idx, channel_idx].cpu().numpy(), cmap="viridis")
        axes[0, 0].set_title(f"Target (ch={channel_idx})")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(intermediates["spectral_down"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[0, 1].set_title(
            f"Spectral down ({intermediates['spectral_down'].shape[2]}x{intermediates['spectral_down'].shape[3]})"
        )
        axes[0, 1].axis("off")

        axes[0, 2].imshow(intermediates["spatial_down"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[0, 2].set_title(
            f"Spatial down ({intermediates['spatial_down'].shape[2]}x{intermediates['spatial_down'].shape[3]})"
        )
        axes[0, 2].axis("off")

        axes[0, 3].imshow(intermediates["spectral_up"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[0, 3].set_title("Spectral up")
        axes[0, 3].axis("off")

        axes[0, 4].imshow(output[sample_idx, channel_idx].cpu().numpy(), cmap="viridis")
        axes[0, 4].set_title(f"Output (ch={channel_idx})")
        axes[0, 4].axis("off")

        # Row 2: Error map, loss curve, spectral mask, spatial up, histogram
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

        axes[1, 3].imshow(intermediates["spatial_up"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[1, 3].set_title("Spatial up")
        axes[1, 3].axis("off")

        axes[1, 4].hist(target.cpu().numpy().flatten(), bins=50, alpha=0.5, label="Target", density=True)
        axes[1, 4].hist(output.cpu().numpy().flatten(), bins=50, alpha=0.5, label="Output", density=True)
        axes[1, 4].set_xlabel("Value")
        axes[1, 4].set_ylabel("Density")
        axes[1, 4].set_title("Value distribution")
        axes[1, 4].legend()

        plt.tight_layout()
        output_path = "nvsubquadratic/modules/dual_path_reconstruction_test.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved visualization to: {output_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available, skipping visualization")

    # Summary
    print("\n" + "=" * 70)
    if final_loss < 0.01:
        print("✅ SUCCESS: Dual-path network can reconstruct input!")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    elif final_loss < 0.1:
        print("⚠️ PARTIAL SUCCESS: Reconstruction is reasonable but not perfect")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    else:
        print("❌ FAILED: Dual-path pipeline loses too much information")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    print("=" * 70)

    return final_loss, correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test dual-path patchify/unpatchify reconstruction")
    parser.add_argument(
        "--num_steps", type=int, default=None, help=f"Number of training steps (default: {TEST_CONFIG['num_steps']})"
    )
    parser.add_argument(
        "--init_flatten",
        action="store_true",
        help="Initialize spatial path conv weights to flatten patches into channels",
    )
    parser.add_argument(
        "--learn_stride",
        action="store_true",
        help="Make the spectral stride learnable (unfreeze spectral mask parameters)",
    )
    args = parser.parse_args()
    main(num_steps=args.num_steps, init_flatten=args.init_flatten, learn_stride=args.learn_stride)
