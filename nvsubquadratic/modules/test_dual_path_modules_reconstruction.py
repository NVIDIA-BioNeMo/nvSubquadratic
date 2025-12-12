"""Test reconstruction using the production DualPathPatchify/DualPathUnpatchify modules.

This test uses the established modules from patchify.py:
    - DualPathPatchify: Combines spectral + spatial downsampling
    - DualPathUnpatchify: Combines spectral + spatial upsampling

The configuration matches examples/spatial_recall_2d/emnist_regression_colored/ccnn_4_160_attn_dual_path_patchify.py

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_modules_reconstruction.py
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_modules_reconstruction.py --num_steps 2000
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_modules_reconstruction.py --learn_stride
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND, SpectralLinearMaskND
from nvsubquadratic.modules.patchify import DualPathPatchify, DualPathUnpatchify, SpectralPatchify, SpectralUnpatchify


# Configuration matching ccnn_4_160_attn_dual_path_patchify.py
CONFIG = {
    "batch_size": 2,
    "in_channels": 3,  # RGB input
    "hidden_channels": 160,  # NUM_HIDDEN_CHANNELS from config
    "H": 64,  # CANVAS_SIZE
    "W": 64,
    "init_stride": 4,  # INIT_STRIDE
    "max_stride": 16,  # MAX_STRIDE
    "num_steps": 1000,
    "seed": 42,
}


class DualPathModulesReconstructionNet(nn.Module):
    """Network using production DualPathPatchify/DualPathUnpatchify modules.

    Configuration matches ccnn_4_160_attn_dual_path_patchify.py
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        init_stride: int = 4,
        max_stride: int = 16,
        freeze_spectral_mask: bool = True,
        mask_type: str = "gaussian",  # Match config: SpectralGaussianMaskND
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.init_stride = init_stride
        self.max_stride = max_stride

        # Choose spectral mask type (config uses SpectralGaussianMaskND)
        if mask_type == "linear":
            spectral_mask_cfg = LazyConfig(SpectralLinearMaskND)(
                data_dim=2,
                transition_fraction=0.1,
                init_stride_value=float(init_stride),
                min_stride_value=1.0,
                max_stride_value=float(max_stride),
                parametrization="direct",
            )
        elif mask_type == "gaussian":
            # Match config: SpectralGaussianMaskND with clip_value=0.5
            spectral_mask_cfg = LazyConfig(SpectralGaussianMaskND)(
                data_dim=2,
                clip_value=0.5,
                init_stride_value=float(init_stride),
                min_stride_value=1.0,
                max_stride_value=float(max_stride),
                parametrization="direct",
            )
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        # Build spectral patchify config (match config: kernel_size=MAX_STRIDE)
        spectral_patchify_cfg = LazyConfig(SpectralPatchify)(
            in_features=in_channels,
            out_features=hidden_channels,
            data_dim=2,
            spectral_mask_cfg=spectral_mask_cfg,
            conv_cfg=LazyConfig(nn.Conv2d)(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=max_stride,  # Match config: kernel_size=MAX_STRIDE
                padding="same",
            ),
        )

        # === PATCHIFY ===
        self.patchify = DualPathPatchify(
            in_features=in_channels,
            out_features=hidden_channels,
            data_dim=2,
            spectral_patchify_cfg=spectral_patchify_cfg,
            max_stride=max_stride,
            freeze_spectral_mask=freeze_spectral_mask,
        )

        # Build spectral unpatchify config (match config: kernel_size=MAX_STRIDE)
        spectral_unpatchify_cfg = LazyConfig(SpectralUnpatchify)(
            in_features=hidden_channels,
            out_features=in_channels,
            data_dim=2,
            output_proj_cfg=LazyConfig(nn.Conv2d)(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=max_stride,  # Match config: kernel_size=MAX_STRIDE
                padding="same",
            ),
            interpolation_mode="bilinear",
        )

        # === UNPATCHIFY ===
        self.unpatchify = DualPathUnpatchify(
            in_features=hidden_channels,
            out_features=in_channels,
            data_dim=2,
            spectral_unpatchify_cfg=spectral_unpatchify_cfg,
            max_stride=max_stride,
            interpolation_mode="bilinear",
        )

    def get_stride(self) -> torch.Tensor:
        """Get the current stride from the patchify module."""
        return self.patchify.get_stride()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass: patchify -> unpatchify.

        Args:
            x: Input tensor [B, C, H, W] (BHL format)

        Returns:
            Tuple of (output, intermediates_dict)
        """
        intermediates = {"input": x}
        B, C, H, W = x.shape

        # Get current stride
        current_stride = self.get_stride()
        intermediates["stride"] = current_stride

        # === PATCHIFY ===
        x_down = self.patchify(x)
        intermediates["patchify_output"] = x_down

        # === UNPATCHIFY ===
        x_up = self.unpatchify(x_down, target_shape=(H, W))
        intermediates["unpatchify_output"] = x_up

        return x_up, intermediates


def get_test_target(device: str = "cpu") -> torch.Tensor:
    """Generate random test target matching config dimensions."""
    torch.manual_seed(CONFIG["seed"])
    return torch.randn(
        CONFIG["batch_size"],
        CONFIG["in_channels"],
        CONFIG["H"],
        CONFIG["W"],
        device=device,
    )


def main(num_steps: int | None = None, learn_stride: bool = False, mask_type: str = "gaussian"):
    print("=" * 70)
    print("Dual-Path Modules Reconstruction Test")
    print("(Configuration matches ccnn_4_160_attn_dual_path_patchify.py)")
    print("=" * 70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Use config
    batch_size = CONFIG["batch_size"]
    in_channels = CONFIG["in_channels"]
    hidden_channels = CONFIG["hidden_channels"]
    H, W = CONFIG["H"], CONFIG["W"]
    init_stride = CONFIG["init_stride"]
    max_stride = CONFIG["max_stride"]
    if num_steps is None:
        num_steps = CONFIG["num_steps"]

    print("\nConfiguration (matching ccnn_4_160_attn_dual_path_patchify.py):")
    print(f"  Input shape: [{batch_size}, {in_channels}, {H}, {W}]")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Init stride: {init_stride}")
    print(f"  Max stride: {max_stride}")
    print(f"  Downsampled shape: [{batch_size}, {hidden_channels}, {int(H / init_stride)}, {int(W / init_stride)}]")
    print(f"  Training steps: {num_steps}")
    print(f"  Learn stride: {learn_stride}")
    print(f"  Mask type: {mask_type}")

    # Create random target
    target = get_test_target(device=device)
    print(f"\nTarget tensor: shape={target.shape}, mean={target.mean():.4f}, std={target.std():.4f}")

    # Create model
    model = DualPathModulesReconstructionNet(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        init_stride=init_stride,
        max_stride=max_stride,
        freeze_spectral_mask=not learn_stride,
        mask_type=mask_type,
    ).to(device)

    # Print model info
    print("\nModel:")
    print(f"  Patchify: {model.patchify}")
    print(f"  Unpatchify: {model.unpatchify}")

    # Check mask is frozen and record initial stride
    initial_stride = model.get_stride().clone().detach()
    print(f"\nSpectral mask frozen: {not learn_stride}")
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    stride_history = []

    for step in range(num_steps):
        optimizer.zero_grad()

        output, _ = model(target)
        loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Track stride
        current_stride = model.get_stride().detach().cpu().tolist()
        stride_history.append(current_stride)

        # Log progress
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  Step {step + 1:4d}: Loss = {loss.item():.6f}, Stride = {current_stride}")

    # Final evaluation
    print("\n" + "-" * 70)
    print("Final Evaluation:")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        output, intermediates = model(target)
        final_loss = F.mse_loss(output, target).item()
        correlation = torch.corrcoef(torch.stack([output.flatten(), target.flatten()]))[0, 1].item()

        print(f"  Final MSE loss: {final_loss:.6f}")
        print(f"  Improvement: {initial_loss / max(final_loss, 1e-10):.1f}x")
        print(f"  Correlation: {correlation:.6f}")

        # Stride change
        final_stride = model.get_stride().detach().cpu().tolist()
        print(f"\n  Initial stride: {initial_stride.tolist()}")
        print(f"  Final stride:   {final_stride}")

        if learn_stride:
            stride_change = [f - i for f, i in zip(final_stride, initial_stride.tolist())]
            print(f"  Stride change:  {stride_change}")

    # Summary
    print("\n" + "=" * 70)
    if final_loss < 0.01:
        print("✓ EXCELLENT: Near-perfect reconstruction (loss < 0.01)")
    elif final_loss < 0.1:
        print("✓ GOOD: Good reconstruction (loss < 0.1)")
    elif final_loss < 0.5:
        print("⚠ MODERATE: Moderate reconstruction (loss < 0.5)")
    else:
        print("✗ POOR: Poor reconstruction (loss >= 0.5)")
    print("=" * 70)

    return {
        "final_loss": final_loss,
        "initial_loss": initial_loss,
        "correlation": correlation,
        "initial_stride": initial_stride.tolist(),
        "final_stride": final_stride,
        "losses": losses,
        "stride_history": stride_history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DualPathPatchify/DualPathUnpatchify modules")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--learn_stride", action="store_true", help="Allow stride to be learned")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="gaussian",
        choices=["linear", "gaussian"],
        help="Type of spectral mask to use (default: gaussian to match config)",
    )

    args = parser.parse_args()
    main(num_steps=args.num_steps, learn_stride=args.learn_stride, mask_type=args.mask_type)
