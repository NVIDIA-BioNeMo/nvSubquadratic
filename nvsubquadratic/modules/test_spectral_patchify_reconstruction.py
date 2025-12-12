"""Test spectral patchify/unpatchify reconstruction capability.

This test verifies that spectral downsampling doesn't lose critical information
by training a simple network to reconstruct its input through:
    Input → SpectralPatchify (stride=4) → SpectralUnpatchify → Output

The spectral masks are frozen (non-learnable) and only the convolutions are trained.

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_spectral_patchify_reconstruction.py
    PYTHONPATH=. python nvsubquadratic/modules/test_spectral_patchify_reconstruction.py --num_steps 2000
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import SpectralLinearMaskND
from nvsubquadratic.modules.patchify import SpectralPatchify, SpectralUnpatchify


class SpectralReconstructionNet(nn.Module):
    """Network that patchifies then unpatchifies to test reconstruction."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: float = 4.0,
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
                transition_fraction=0.01,
                init_stride_value=stride,
            ),
            conv_cfg=LazyConfig(nn.Conv2d)(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=4,
                padding="same",
            ),
        )

        # SpectralUnpatchify with a simple Conv2d output projection
        self.unpatchify = SpectralUnpatchify(
            in_features=hidden_channels,
            out_features=in_channels,
            data_dim=2,
            output_proj_cfg=LazyConfig(nn.Conv2d)(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=4,
                padding="same",
            ),
            interpolation_mode="bilinear",
        )

        # Freeze spectral masks if requested
        if freeze_masks:
            for param in self.patchify.spectral_mask.parameters():
                param.requires_grad = False

        # Initialize conv weights to flatten/unflatten if requested
        if init_flatten:
            init_flatten_weights(self.patchify.conv, is_transpose=False)
            init_flatten_weights(self.unpatchify.output_proj, is_transpose=True)

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
        x_down, patchify_intermediates = self.patchify(x, is_bhl_input=True, return_intermediates=True)
        intermediates["after_patchify_conv"] = patchify_intermediates.get(
            "after_conv", patchify_intermediates["input"]
        )
        intermediates["spectral_mask"] = patchify_intermediates["spectral_mask"]
        intermediates["after_spectral_down"] = x_down

        # Step 2: SpectralUnpatchify (bilinear upsampling + conv)
        x_out = self.unpatchify(x_down, target_shape=(H, W), is_bhl_input=True)
        intermediates["after_spectral_up"] = x_out  # After interpolation, before/after conv
        intermediates["output"] = x_out

        return x_out, intermediates


# Shared test configuration
TEST_CONFIG = {
    "batch_size": 2,
    "in_channels": 3,
    "H": 64,
    "W": 64,
    "stride": 4,  # Use int for consistency with patch_size
    "num_steps": 1000,
    "seed": 42,
}
# Compute hidden_channels to avoid bottleneck: stride^2 * in_channels
TEST_CONFIG["hidden_channels"] = TEST_CONFIG["stride"] ** 2 * TEST_CONFIG["in_channels"]  # 4*4*3 = 48


def get_test_target(device: str = "cpu") -> torch.Tensor:
    """Generate the shared test target tensor.

    Uses fixed seed for reproducibility across tests.

    Args:
        device: Device to create tensor on.

    Returns:
        Target tensor of shape [batch_size, in_channels, H, W].
    """
    torch.manual_seed(TEST_CONFIG["seed"])
    target = torch.randn(
        TEST_CONFIG["batch_size"],
        TEST_CONFIG["in_channels"],
        TEST_CONFIG["H"],
        TEST_CONFIG["W"],
        device=device,
    )
    return target


def init_flatten_weights(
    conv: nn.Conv2d | nn.ConvTranspose2d,
    is_transpose: bool = False,
) -> None:
    """Initialize conv weights to flatten/unflatten patches into channels.

    For Conv2d (patchify): Each output channel samples one (in_channel, y, x)
    position from the patch. This is like a learned version of:
        rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=k, p2=k)

    For ConvTranspose2d (unpatchify): The inverse mapping.

    Args:
        conv: The conv layer to initialize.
        is_transpose: Whether this is a transpose conv (for unpatchify).
    """
    with torch.no_grad():
        if is_transpose:
            # ConvTranspose2d: weight shape is [in_channels, out_channels, kH, kW]
            in_ch, out_ch, kH, kW = conv.weight.shape
            conv.weight.zero_()
            if conv.bias is not None:
                conv.bias.zero_()

            # Map each input channel to a (out_channel, y, x) position
            idx = 0
            for c in range(out_ch):
                for ky in range(kH):
                    for kx in range(kW):
                        if idx < in_ch:
                            conv.weight[idx, c, ky, kx] = 1.0
                            idx += 1
        else:
            # Conv2d: weight shape is [out_channels, in_channels, kH, kW]
            out_ch, in_ch, kH, kW = conv.weight.shape
            conv.weight.zero_()
            if conv.bias is not None:
                conv.bias.zero_()

            # Map each (in_channel, y, x) to one output channel
            idx = 0
            for c in range(in_ch):
                for ky in range(kH):
                    for kx in range(kW):
                        if idx < out_ch:
                            conv.weight[idx, c, ky, kx] = 1.0
                            idx += 1


def train_reconstruction(
    model: nn.Module,
    target: torch.Tensor,
    num_steps: int = 1000,
    lr: float = 1e-3,
    print_every: int = 100,
) -> list[float]:
    """Train the model to reconstruct the target from random input.

    Args:
        model: The reconstruction network
        target: Target tensor to reconstruct [B, C, H, W]
        num_steps: Number of training steps
        lr: Learning rate
        print_every: Print loss every N steps

    Returns:
        List of losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        output, _ = model(target)
        loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % print_every == 0 or step == num_steps - 1:
            print(f"Step {step:4d}: Loss = {loss.item():.6f}")

    return losses


def main(num_steps: int | None = None, init_flatten: bool = False):
    print("=" * 70)
    print("Spectral Patchify/Unpatchify Reconstruction Test")
    print("=" * 70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Use shared configuration
    batch_size = TEST_CONFIG["batch_size"]
    in_channels = TEST_CONFIG["in_channels"]
    hidden_channels = TEST_CONFIG["hidden_channels"]
    H, W = TEST_CONFIG["H"], TEST_CONFIG["W"]
    stride = float(TEST_CONFIG["stride"])  # SpectralPatchify uses float stride
    if num_steps is None:
        num_steps = TEST_CONFIG["num_steps"]

    print("\nConfiguration:")
    print(f"  Input shape: [{batch_size}, {in_channels}, {H}, {W}]")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Stride: {stride}")
    print(f"  Downsampled shape: [{batch_size}, {hidden_channels}, {int(H / stride)}, {int(W / stride)}]")
    print(f"  Training steps: {num_steps}")
    print(f"  Init flatten: {init_flatten}")

    # Create random target using shared function
    target = get_test_target(device=device)
    print(f"\nTarget tensor: shape={target.shape}, mean={target.mean():.4f}, std={target.std():.4f}")

    # Create model
    model = SpectralReconstructionNet(
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
        fig.suptitle(f"Spectral Patchify/Unpatchify Reconstruction (stride={stride})", fontsize=14)

        # Row 1: Input -> Downsampled -> Upsampled -> Output
        sample_idx = 0
        channel_idx = 0

        axes[0, 0].imshow(target[sample_idx, channel_idx].cpu().numpy(), cmap="viridis")
        axes[0, 0].set_title(f"Target (ch={channel_idx})")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(intermediates["after_spectral_down"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[0, 1].set_title(f"After spectral down ({int(H / stride)}x{int(W / stride)})")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(intermediates["after_spectral_up"][sample_idx, 0].cpu().numpy(), cmap="viridis")
        axes[0, 2].set_title(f"After spectral up ({H}x{W})")
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
        output_path = "nvsubquadratic/modules/spectral_reconstruction_test.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved visualization to: {output_path}")
        plt.close()

    except ImportError:
        print("\nmatplotlib not available, skipping visualization")

    # Summary
    print("\n" + "=" * 70)
    if final_loss < 0.01:
        print("✅ SUCCESS: Network can reconstruct input through spectral downsampling!")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    elif final_loss < 0.1:
        print("⚠️ PARTIAL SUCCESS: Reconstruction is reasonable but not perfect")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    else:
        print("❌ FAILED: Spectral downsampling loses too much information")
        print(f"   Final loss: {final_loss:.6f}, Correlation: {correlation:.4f}")
    print("=" * 70)

    return final_loss, correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test spectral patchify/unpatchify reconstruction")
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
