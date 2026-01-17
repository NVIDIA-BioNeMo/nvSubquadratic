"""Test reconstruction using DualPathPatchify/DualPathUnpatchify modules.

Goal: Verify whether the dual-path architecture can achieve absolute zero loss
like the conventional patchify (Conv2d + ConvTranspose2d).

Uses shared TEST_CONFIG for fair comparison with conventional patchify test.

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_modules_reconstruction.py
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_modules_reconstruction.py --num_steps 10000
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_modules_reconstruction.py --clip_value 0.99
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_modules_reconstruction.py --no_stride_dependent_mask

    # Test paths in isolation:
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_modules_reconstruction.py --spectral_only
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_modules_reconstruction.py --spatial_only
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND, SpectralLinearMaskND
from nvsubquadratic.modules.patchify import DualPathPatchify, DualPathUnpatchify, SpectralPatchify, SpectralUnpatchify
from nvsubquadratic.modules.test_spectral_patchify_reconstruction import (
    TEST_CONFIG,
    get_test_target,
)


class SpectralOnlyReconstructionNet(nn.Module):
    """Spectral path only: low-pass filter + subsample + bilinear upsample + conv."""

    def __init__(self, in_channels: int, hidden_channels: int, stride: int = 4, clip_value: float = 0.5):
        super().__init__()
        self.stride = stride

        spectral_mask_cfg = LazyConfig(SpectralGaussianMaskND)(
            data_dim=2,
            clip_value=clip_value,
            init_stride_value=float(stride),
            min_stride_value=1.0,
            max_stride_value=float(stride),
            parametrization="direct",
        )

        self.patchify = SpectralPatchify(
            in_features=in_channels,
            out_features=hidden_channels,
            data_dim=2,
            spectral_mask_cfg=spectral_mask_cfg,
            conv_cfg=LazyConfig(nn.Conv2d)(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=stride,
                padding="same",
            ),
        )

        self.unpatchify = SpectralUnpatchify(
            in_features=hidden_channels,
            out_features=in_channels,
            data_dim=2,
            output_proj_cfg=LazyConfig(nn.Conv2d)(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=stride,
                padding="same",
            ),
            interpolation_mode="bilinear",
        )

    def get_stride(self) -> torch.Tensor:
        return self.patchify.spectral_mask.get_stride()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        # Patchify expects channels-last for spectral path
        x_cl = rearrange(x, "b c h w -> b h w c")
        x_down = self.patchify(x_cl)  # Returns channels-last
        # Unpatchify expects channels-first (BHL)
        x_down_bhl = rearrange(x_down, "b h w c -> b c h w")
        x_up = self.unpatchify(x_down_bhl, target_shape=(H, W), is_bhl_input=True)
        return x_up, {"patchify_output": x_down_bhl}


class SpatialOnlyReconstructionNet(nn.Module):
    """Spatial path only: strided conv + PixelShuffle (or ConvTranspose2d)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 4,
        use_conv_transpose: bool = False,  # If True, use ConvTranspose2d like conventional patchify
    ):
        super().__init__()
        self.stride = stride
        self.use_conv_transpose = use_conv_transpose

        # Patchify: strided conv (match conventional patchify: kernel_size=stride, padding=0)
        self.patchify_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=stride,
            stride=stride,
            padding=0,  # Match conventional patchify
        )

        if use_conv_transpose:
            # Unpatchify: ConvTranspose2d (like conventional patchify)
            self.unpatchify = nn.ConvTranspose2d(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=stride,
                stride=stride,
            )
        else:
            # Unpatchify: conv + PixelShuffle
            self.unpatchify_expand = nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=in_channels * stride * stride,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.unpatchify_shuffle = nn.PixelShuffle(stride)

    def get_stride(self) -> torch.Tensor:
        return torch.tensor([float(self.stride), float(self.stride)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        x_down = self.patchify_conv(x)

        if self.use_conv_transpose:
            x_up = self.unpatchify(x_down)
        else:
            x_up = self.unpatchify_expand(x_down)
            x_up = self.unpatchify_shuffle(x_up)

        # Resize if needed (for non-divisible sizes)
        if x_up.shape[2:] != (H, W):
            x_up = F.interpolate(x_up, size=(H, W), mode="bilinear")
        return x_up, {"patchify_output": x_down}


class DualPathModulesReconstructionNet(nn.Module):
    """Network using DualPathPatchify/DualPathUnpatchify modules for reconstruction test."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 4,
        freeze_spectral_mask: bool = True,
        mask_type: str = "gaussian",
        clip_value: float = 0.5,
        use_stride_dependent_mask: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.stride = stride
        self.clip_value = clip_value
        self.use_stride_dependent_mask = use_stride_dependent_mask

        # Choose spectral mask type
        if mask_type == "linear":
            spectral_mask_cfg = LazyConfig(SpectralLinearMaskND)(
                data_dim=2,
                transition_fraction=0.1,
                init_stride_value=float(stride),
                min_stride_value=1.0,
                max_stride_value=float(stride),
                parametrization="direct",
            )
        elif mask_type == "gaussian":
            spectral_mask_cfg = LazyConfig(SpectralGaussianMaskND)(
                data_dim=2,
                clip_value=clip_value,
                init_stride_value=float(stride),
                min_stride_value=1.0,
                max_stride_value=float(stride),
                parametrization="direct",
            )
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        # Build spectral patchify config
        spectral_patchify_cfg = LazyConfig(SpectralPatchify)(
            in_features=in_channels,
            out_features=hidden_channels,
            data_dim=2,
            spectral_mask_cfg=spectral_mask_cfg,
            conv_cfg=LazyConfig(nn.Conv2d)(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=stride,
                padding="same",
            ),
        )

        # === PATCHIFY ===
        self.patchify = DualPathPatchify(
            in_features=in_channels,
            out_features=hidden_channels,
            data_dim=2,
            spectral_patchify_cfg=spectral_patchify_cfg,
            max_stride=stride,
            freeze_spectral_mask=freeze_spectral_mask,
            use_stride_dependent_mask=use_stride_dependent_mask,
        )

        # Build spectral unpatchify config
        spectral_unpatchify_cfg = LazyConfig(SpectralUnpatchify)(
            in_features=hidden_channels,
            out_features=in_channels,
            data_dim=2,
            output_proj_cfg=LazyConfig(nn.Conv2d)(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=stride,
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
            max_stride=stride,
            interpolation_mode="bilinear",
        )

    def get_stride(self) -> torch.Tensor:
        """Get the current stride from the patchify module."""
        return self.patchify.get_stride()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Forward pass: patchify -> unpatchify."""
        intermediates = {"input": x}
        B, C, H, W = x.shape

        current_stride = self.get_stride()
        intermediates["stride"] = current_stride

        x_down = self.patchify(x)
        intermediates["patchify_output"] = x_down

        x_up = self.unpatchify(x_down, target_shape=(H, W))
        intermediates["unpatchify_output"] = x_up

        return x_up, intermediates


def main(
    num_steps: int | None = None,
    learn_stride: bool = False,
    mask_type: str = "gaussian",
    clip_value: float = 0.5,
    use_stride_dependent_mask: bool = True,
    spectral_only: bool = False,
    spatial_only: bool = False,
    use_conv_transpose: bool = False,
):
    # Determine mode
    if spectral_only and spatial_only:
        raise ValueError("Cannot use both --spectral_only and --spatial_only")

    if spectral_only:
        mode = "SPECTRAL ONLY"
    elif spatial_only:
        mode = "SPATIAL ONLY"
    else:
        mode = "DUAL PATH"

    print("=" * 70)
    print(f"Reconstruction Test: {mode}")
    print("Goal: Can this architecture reach ABSOLUTE ZERO loss?")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Use shared TEST_CONFIG (same as conventional patchify)
    batch_size = TEST_CONFIG["batch_size"]
    in_channels = TEST_CONFIG["in_channels"]
    hidden_channels = TEST_CONFIG["hidden_channels"]  # 48 = 4*4*3
    H, W = TEST_CONFIG["H"], TEST_CONFIG["W"]
    stride = TEST_CONFIG["stride"]  # 4
    if num_steps is None:
        num_steps = TEST_CONFIG["num_steps"]

    print("\nConfiguration (shared with conventional patchify test):")
    print(f"  Mode: {mode}")
    print(f"  Input shape: [{batch_size}, {in_channels}, {H}, {W}]")
    print(f"  Hidden channels: {hidden_channels}")
    print(f"  Stride: {stride}")
    print(f"  Downsampled shape: [{batch_size}, {hidden_channels}, {H // stride}, {W // stride}]")
    print(f"  Training steps: {num_steps}")
    if not spatial_only:
        print(f"  Clip value: {clip_value}")
    if not spectral_only and not spatial_only:
        print(f"  Learn stride: {learn_stride}")
        print(f"  Mask type: {mask_type}")
        print(f"  Use stride-dependent mask: {use_stride_dependent_mask}")

    # Create shared test target
    target = get_test_target(device=device)
    print(f"\nTarget tensor: shape={target.shape}, mean={target.mean():.4f}, std={target.std():.4f}")

    # Create model based on mode
    if spectral_only:
        model = SpectralOnlyReconstructionNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            clip_value=clip_value,
        ).to(device)
    elif spatial_only:
        model = SpatialOnlyReconstructionNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            use_conv_transpose=use_conv_transpose,
        ).to(device)
    else:
        model = DualPathModulesReconstructionNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            freeze_spectral_mask=not learn_stride,
            mask_type=mask_type,
            clip_value=clip_value,
            use_stride_dependent_mask=use_stride_dependent_mask,
        ).to(device)

    print("\nModel:")
    print(f"  {model}")

    initial_stride = model.get_stride().clone().detach()
    print(f"\n  Initial stride: {initial_stride.tolist()}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable_params:,} / {total_params:,}")

    # Initial forward pass
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

    # Training
    print("\n" + "-" * 70)
    print("Training:")
    print("-" * 70)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()
        output, _ = model(target)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 100 == 0 or step == 0:
            _ = model.get_stride().detach().cpu().tolist()  # current stride value
            print(f"  Step {step + 1:5d}: Loss = {loss.item():.10f}")

    # Final evaluation
    print("\n" + "-" * 70)
    print("Final Evaluation:")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        output, _ = model(target)
        final_loss = F.mse_loss(output, target).item()
        correlation = torch.corrcoef(torch.stack([output.flatten(), target.flatten()]))[0, 1].item()

        print(f"  Final MSE loss: {final_loss:.10f}")
        print(f"  Improvement: {initial_loss / max(final_loss, 1e-10):.1f}x")
        print(f"  Correlation: {correlation:.6f}")

        final_stride = model.get_stride().detach().cpu().tolist()
        print(f"\n  Initial stride: {initial_stride.tolist()}")
        print(f"  Final stride:   {final_stride}")

    # Summary
    print("\n" + "=" * 70)
    if final_loss < 1e-6:
        print("✅ ABSOLUTE ZERO: Loss < 1e-6 (matches conventional patchify)")
    elif final_loss < 1e-4:
        print("⚠️  VERY CLOSE: Loss < 1e-4 (nearly zero)")
    elif final_loss < 0.01:
        print("⚠️  GOOD: Loss < 0.01 (but not absolute zero)")
    else:
        print("❌ NOT ZERO: Loss >= 0.01 (architecture cannot achieve zero loss)")
    print("=" * 70)

    return {"final_loss": final_loss, "correlation": correlation, "losses": losses}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test if DualPath can reach absolute zero loss")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of training steps")
    parser.add_argument("--learn_stride", action="store_true", help="Allow stride to be learned")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="gaussian",
        choices=["linear", "gaussian"],
        help="Type of spectral mask (default: gaussian)",
    )
    parser.add_argument(
        "--clip_value",
        type=float,
        default=0.5,
        help="Spectral mask clip value (default: 0.5, use ~0.99 for minimal clipping)",
    )
    parser.add_argument(
        "--no_stride_dependent_mask",
        action="store_true",
        help="Disable stride-dependent Gaussian mask on spatial conv kernel",
    )
    # Path isolation options
    parser.add_argument(
        "--spectral_only",
        action="store_true",
        help="Test spectral path only (low-pass + bilinear upsample)",
    )
    parser.add_argument(
        "--spatial_only",
        action="store_true",
        help="Test spatial path only (strided conv + PixelShuffle)",
    )
    parser.add_argument(
        "--use_conv_transpose",
        action="store_true",
        help="Use ConvTranspose2d instead of PixelShuffle for spatial upsampling",
    )

    args = parser.parse_args()
    main(
        num_steps=args.num_steps,
        learn_stride=args.learn_stride,
        mask_type=args.mask_type,
        clip_value=args.clip_value,
        use_stride_dependent_mask=not args.no_stride_dependent_mask,
        spectral_only=args.spectral_only,
        spatial_only=args.spatial_only,
        use_conv_transpose=args.use_conv_transpose,
    )
