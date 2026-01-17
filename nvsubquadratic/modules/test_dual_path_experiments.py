"""Experimental test file for DualPathPatchify/DualPathUnpatchify reconstruction.

Goal: Find modifications that allow the dual-path architecture to reach absolute zero loss
like the conventional patchify (Conv2d + ConvTranspose2d).

This file contains local copies of the DualPath classes that can be modified without
affecting the original patchify.py.

Usage:
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_experiments.py --experiment baseline
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_experiments.py --experiment conv_transpose
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_experiments.py --experiment spectral_only_conv_transpose
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_experiments.py --experiment spatial_only_conv_transpose
    PYTHONPATH=. python nvsubquadratic/modules/test_dual_path_experiments.py --experiment all
"""

import argparse
import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND
from nvsubquadratic.modules.patchify import SpectralPatchify
from nvsubquadratic.modules.test_spectral_patchify_reconstruction import (
    TEST_CONFIG,
    get_test_target,
)


# =============================================================================
# LOCAL COPIES OF DUALPATH CLASSES (can be modified for experiments)
# =============================================================================


class ExperimentalSpectralUnpatchify(nn.Module):
    """Modified SpectralUnpatchify with configurable upsampling method."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        output_proj_cfg: LazyConfig,
        upsample_mode: Literal["bilinear", "conv_transpose", "nearest"] = "bilinear",
        stride: int = 4,
    ):
        super().__init__()
        assert data_dim == 2
        self.upsample_mode = upsample_mode
        self.stride = stride

        self.output_proj = instantiate(output_proj_cfg)

        if upsample_mode == "conv_transpose":
            # Learnable upsampling via ConvTranspose2d
            self.upsample = nn.ConvTranspose2d(
                in_channels=in_features,
                out_channels=in_features,
                kernel_size=stride,
                stride=stride,
            )

    def forward(self, x: torch.Tensor, target_shape: tuple[int, int], is_bhl_input: bool = False) -> torch.Tensor:
        if not is_bhl_input:
            x = rearrange(x, "b h w c -> b c h w")

        # Upsample based on mode
        if self.upsample_mode == "conv_transpose":
            x = self.upsample(x)
            # Resize to exact target if needed
            if x.shape[2:] != target_shape:
                x = F.interpolate(x, size=target_shape, mode="bilinear", align_corners=False)
        elif self.upsample_mode == "bilinear":
            if x.shape[2:] != target_shape:
                x = F.interpolate(x, size=target_shape, mode="bilinear", align_corners=False)
        elif self.upsample_mode == "nearest":
            if x.shape[2:] != target_shape:
                x = F.interpolate(x, size=target_shape, mode="nearest")

        # Apply output projection
        x = self.output_proj(x)

        if not is_bhl_input:
            x = rearrange(x, "b c h w -> b h w c")
        return x


class ExperimentalDualPathUnpatchify(nn.Module):
    """Experimental DualPathUnpatchify with configurable upsampling methods.

    Modifications available:
    - spatial_upsample_mode: "pixel_shuffle" (original) or "conv_transpose"
    - spectral_upsample_mode: "bilinear" (original), "conv_transpose", or "nearest"
    - combine_mode: "sum" (original), "concat" (concatenate then project), or "learnable_blend"
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        max_stride: int = 4,
        spatial_upsample_mode: Literal["pixel_shuffle", "conv_transpose"] = "pixel_shuffle",
        spectral_upsample_mode: Literal["bilinear", "conv_transpose", "nearest"] = "bilinear",
        combine_mode: Literal["sum", "concat", "learnable_blend"] = "sum",
        clip_value: float = 0.5,
    ):
        super().__init__()
        assert data_dim == 2

        self.in_features = in_features
        self.out_features = out_features
        self.max_stride = max_stride
        self.spatial_upsample_mode = spatial_upsample_mode
        self.spectral_upsample_mode = spectral_upsample_mode
        self.combine_mode = combine_mode

        # For concat mode, each path outputs out_features, then we combine
        path_out_features = out_features

        # === SPECTRAL PATH ===
        self.spectral_unpatchify = ExperimentalSpectralUnpatchify(
            in_features=in_features,
            out_features=path_out_features,
            data_dim=2,
            output_proj_cfg=LazyConfig(nn.Conv2d)(
                in_channels=in_features,
                out_channels=path_out_features,
                kernel_size=max_stride,
                padding="same",
            ),
            upsample_mode=spectral_upsample_mode,
            stride=max_stride,
        )

        # === SPATIAL PATH ===
        if spatial_upsample_mode == "conv_transpose":
            # ConvTranspose2d (like conventional patchify) - CAN reach zero!
            self.spatial_upsample = nn.ConvTranspose2d(
                in_channels=in_features,
                out_channels=path_out_features,
                kernel_size=max_stride,
                stride=max_stride,
            )
        else:
            # PixelShuffle (original) - CANNOT reach zero
            self.spatial_expand = nn.Conv2d(
                in_channels=in_features,
                out_channels=path_out_features * max_stride * max_stride,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            self.spatial_shuffle = nn.PixelShuffle(max_stride)

        # === COMBINE PROJECTION (for concat mode) ===
        if combine_mode == "concat":
            self.combine_proj = nn.Conv2d(
                in_channels=path_out_features * 2,  # Both paths concatenated
                out_channels=out_features,
                kernel_size=1,
            )
        elif combine_mode == "learnable_blend":
            # Learnable alpha: output = (1-alpha) * spatial + alpha * spectral
            # Initialize alpha to 0.5 (equal weighting), use sigmoid to keep in [0, 1]
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def forward(self, x: torch.Tensor, target_shape: tuple[int, int]) -> torch.Tensor:
        # === SPECTRAL PATH ===
        x_spectral = self.spectral_unpatchify(x, target_shape=target_shape, is_bhl_input=True)

        # === SPATIAL PATH ===
        if self.spatial_upsample_mode == "conv_transpose":
            x_spatial = self.spatial_upsample(x)
        else:
            x_spatial = self.spatial_expand(x)
            x_spatial = self.spatial_shuffle(x_spatial)

        # Resize spatial to target shape if needed
        if x_spatial.shape[2:] != target_shape:
            x_spatial = F.interpolate(x_spatial, size=target_shape, mode="bilinear", align_corners=False)

        # Combine paths
        if self.combine_mode == "concat":
            output = torch.cat([x_spectral, x_spatial], dim=1)
            output = self.combine_proj(output)
        elif self.combine_mode == "learnable_blend":
            alpha = torch.sigmoid(self.alpha_logit)
            output = (1 - alpha) * x_spatial + alpha * x_spectral
        else:
            output = x_spectral + x_spatial

        return output

    def get_alpha(self) -> float:
        """Get current alpha value (for learnable_blend mode)."""
        if self.combine_mode == "learnable_blend":
            return torch.sigmoid(self.alpha_logit).item()
        return None


class ExperimentalDualPathPatchify(nn.Module):
    """Experimental DualPathPatchify - local copy for modifications."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        max_stride: int = 4,
        clip_value: float = 0.5,
        use_stride_dependent_mask: bool = False,
        combine_mode: Literal["sum", "learnable_blend"] = "sum",
    ):
        super().__init__()
        assert data_dim == 2

        self.in_features = in_features
        self.out_features = out_features
        self.max_stride = max_stride
        self.use_stride_dependent_mask = use_stride_dependent_mask
        self.combine_mode = combine_mode
        self._cutoff_factor = math.sqrt(-2.0 * math.log(clip_value)) if use_stride_dependent_mask else None

        # === SPECTRAL PATH ===
        spectral_mask_cfg = LazyConfig(SpectralGaussianMaskND)(
            data_dim=2,
            clip_value=clip_value,
            init_stride_value=float(max_stride),
            min_stride_value=1.0,
            max_stride_value=float(max_stride),
            parametrization="direct",
        )
        self.spectral_patchify = SpectralPatchify(
            in_features=in_features,
            out_features=out_features,
            data_dim=2,
            spectral_mask_cfg=spectral_mask_cfg,
            conv_cfg=LazyConfig(nn.Conv2d)(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=max_stride,
                padding="same",
            ),
        )

        # Freeze spectral mask
        for param in self.spectral_patchify.spectral_mask.parameters():
            param.requires_grad = False

        # === SPATIAL PATH ===
        # Match conventional patchify: kernel_size=stride, stride=stride, padding=0
        self.spatial_conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=max_stride,
            stride=max_stride,
            padding=0,
        )

        # === LEARNABLE BLEND ===
        if combine_mode == "learnable_blend":
            # Learnable alpha: output = (1-alpha) * spatial + alpha * spectral
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

    def get_stride(self) -> torch.Tensor:
        return self.spectral_patchify.get_stride()

    def get_alpha(self) -> float:
        """Get current alpha value (for learnable_blend mode)."""
        if self.combine_mode == "learnable_blend":
            return torch.sigmoid(self.alpha_logit).item()
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === SPECTRAL PATH ===
        x_spectral = self.spectral_patchify(x, is_bhl_input=True)

        # === SPATIAL PATH ===
        x_spatial = self.spatial_conv(x)

        # Resize spatial to match spectral if needed
        if x_spatial.shape[2:] != x_spectral.shape[2:]:
            x_spatial = F.interpolate(x_spatial, size=x_spectral.shape[2:], mode="bilinear", align_corners=False)

        # Combine paths
        if self.combine_mode == "learnable_blend":
            alpha = torch.sigmoid(self.alpha_logit)
            output = (1 - alpha) * x_spatial + alpha * x_spectral
        else:
            output = x_spectral + x_spatial

        return output


class ExperimentalDualPathNet(nn.Module):
    """Experimental network using configurable DualPath modules."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 4,
        clip_value: float = 0.5,
        spatial_upsample_mode: str = "pixel_shuffle",
        spectral_upsample_mode: str = "bilinear",
        combine_mode: str = "sum",
    ):
        super().__init__()
        self.stride = stride
        self.combine_mode = combine_mode

        # For patchify, only "sum" or "learnable_blend" are valid
        patchify_combine = combine_mode if combine_mode in ["sum", "learnable_blend"] else "sum"

        self.patchify = ExperimentalDualPathPatchify(
            in_features=in_channels,
            out_features=hidden_channels,
            data_dim=2,
            max_stride=stride,
            clip_value=clip_value,
            use_stride_dependent_mask=False,
            combine_mode=patchify_combine,
        )

        self.unpatchify = ExperimentalDualPathUnpatchify(
            in_features=hidden_channels,
            out_features=in_channels,
            data_dim=2,
            max_stride=stride,
            spatial_upsample_mode=spatial_upsample_mode,
            spectral_upsample_mode=spectral_upsample_mode,
            combine_mode=combine_mode,
            clip_value=clip_value,
        )

    def get_stride(self) -> torch.Tensor:
        return self.patchify.get_stride()

    def get_alphas(self) -> dict:
        """Get alpha values from patchify and unpatchify (for learnable_blend mode)."""
        return {
            "patchify_alpha": self.patchify.get_alpha(),
            "unpatchify_alpha": self.unpatchify.get_alpha(),
        }

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        x_down = self.patchify(x)
        x_up = self.unpatchify(x_down, target_shape=(H, W))
        return x_up, {"patchify_output": x_down}


# =============================================================================
# ISOLATED PATH NETWORKS FOR ABLATION
# =============================================================================


class SpectralOnlyNet(nn.Module):
    """Spectral path only with configurable upsampling."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 4,
        clip_value: float = 0.5,
        upsample_mode: str = "bilinear",
    ):
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

        self.unpatchify = ExperimentalSpectralUnpatchify(
            in_features=hidden_channels,
            out_features=in_channels,
            data_dim=2,
            output_proj_cfg=LazyConfig(nn.Conv2d)(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=stride,
                padding="same",
            ),
            upsample_mode=upsample_mode,
            stride=stride,
        )

    def get_stride(self) -> torch.Tensor:
        return self.patchify.spectral_mask.get_stride()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        x_cl = rearrange(x, "b c h w -> b h w c")
        x_down = self.patchify(x_cl)
        x_down_bhl = rearrange(x_down, "b h w c -> b c h w")
        x_up = self.unpatchify(x_down_bhl, target_shape=(H, W), is_bhl_input=True)
        return x_up, {"patchify_output": x_down_bhl}


class NoMaskSpectralNet(nn.Module):
    """Spectral-style path but WITHOUT the spectral mask (just conv + subsample + upsample + conv)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 4,
        upsample_mode: str = "conv_transpose",
    ):
        super().__init__()
        self.stride = stride
        self.upsample_mode = upsample_mode

        # Patchify: Conv with padding=same, then subsample (no spectral mask!)
        self.patchify_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=stride,
            stride=1,
            padding="same",  # Use proper same padding
        )

        # Unpatchify: Upsample then conv
        if upsample_mode == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=stride,
                stride=stride,
            )

        self.unpatchify_conv = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=stride,
            stride=1,
            padding="same",
        )

    def get_stride(self) -> torch.Tensor:
        return torch.tensor([float(self.stride), float(self.stride)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape

        # Patchify: conv then subsample
        x_down = self.patchify_conv(x)
        x_down = x_down[:, :, :: self.stride, :: self.stride]  # Subsample

        # Unpatchify: upsample then conv
        if self.upsample_mode == "conv_transpose":
            x_up = self.upsample(x_down)
        else:
            x_up = F.interpolate(x_down, size=(H, W), mode="bilinear", align_corners=False)

        # Resize to exact target shape
        if x_up.shape[2:] != (H, W):
            x_up = F.interpolate(x_up, size=(H, W), mode="bilinear", align_corners=False)

        x_up = self.unpatchify_conv(x_up)

        return x_up, {"patchify_output": x_down}


class SpatialOnlyNet(nn.Module):
    """Spatial path only with configurable upsampling."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        stride: int = 4,
        upsample_mode: str = "conv_transpose",
    ):
        super().__init__()
        self.stride = stride
        self.upsample_mode = upsample_mode

        # Patchify: strided conv (match conventional: kernel=stride, stride=stride)
        self.patchify_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=stride,
            stride=stride,
            padding=0,
        )

        if upsample_mode == "conv_transpose":
            self.unpatchify = nn.ConvTranspose2d(
                in_channels=hidden_channels,
                out_channels=in_channels,
                kernel_size=stride,
                stride=stride,
            )
        else:
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

        if self.upsample_mode == "conv_transpose":
            x_up = self.unpatchify(x_down)
        else:
            x_up = self.unpatchify_expand(x_down)
            x_up = self.unpatchify_shuffle(x_up)

        if x_up.shape[2:] != (H, W):
            x_up = F.interpolate(x_up, size=(H, W), mode="bilinear", align_corners=False)

        return x_up, {"patchify_output": x_down}


class LearnableStrideConvTransposeNet(nn.Module):
    """ConvTranspose with learnable stride via bilinear resize bottleneck.

    Architecture:
        Input [HxW]
            │
            ▼ Conv(stride=max_stride)
        [H/max_stride x W/max_stride]  (e.g., 4x4 for 64x64 with max_stride=16)
            │
            ▼ Bilinear↑ to learned stride resolution
        [H/stride x W/stride]  ← BOTTLENECK (e.g., 16x16 for stride=4)
            │
            ▼ Bilinear↓ back to max_stride resolution
        [H/max_stride x W/max_stride]
            │
            ▼ ConvTranspose(stride=max_stride)
        Output [HxW]

    Key properties:
        - Conv and ConvTranspose always operate at max_stride resolution, preserving
          their ability to be perfect inverses (achieves zero reconstruction loss)
        - Learned stride controls the intermediate bottleneck resolution
        - Stride is parameterized as: stride = max_stride * sigmoid(stride_logit)
        - Stride learning comes from downstream tasks, not reconstruction loss

    Args:
        in_channels: Number of input/output channels
        hidden_channels: Number of channels in the latent representation
        init_stride: Initial stride value (must be <= max_stride)
        max_stride: Maximum stride (determines Conv/ConvTranspose kernel and stride)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        init_stride: int = 4,
        max_stride: int = 16,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.init_stride = init_stride
        self.max_stride = max_stride

        # Learnable stride parameter (sigmoid parameterization for range [0, max_stride])
        init_ratio = init_stride / max_stride
        init_logit = math.log(init_ratio / (1 - init_ratio))  # inverse sigmoid
        self.stride_logit = nn.Parameter(torch.tensor([init_logit, init_logit]))

        # Patchify: Conv with max_stride
        self.patchify_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=max_stride,
            stride=max_stride,
            padding=0,
        )

        # Unpatchify: ConvTranspose with max_stride
        self.unpatchify_conv = nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=in_channels,
            kernel_size=max_stride,
            stride=max_stride,
        )

    def get_stride(self) -> torch.Tensor:
        """Get current learned stride as tensor of shape [2] for (H, W)."""
        return self.max_stride * torch.sigmoid(self.stride_logit)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        current_stride = self.get_stride()

        # Compute resolutions
        stride_res = (int(H / current_stride[0].item()), int(W / current_stride[1].item()))
        max_res = (H // self.max_stride, W // self.max_stride)

        # === PATCHIFY ===
        x_max_down = self.patchify_conv(x)  # [H, W] → [H/max_stride, W/max_stride]
        x_down = F.interpolate(
            x_max_down, size=stride_res, mode="bilinear", align_corners=False
        )  # → [H/stride, W/stride]

        # === UNPATCHIFY ===
        x_back_to_max = F.interpolate(
            x_down, size=max_res, mode="bilinear", align_corners=False
        )  # → [H/max_stride, W/max_stride]
        x_up = self.unpatchify_conv(x_back_to_max)  # → [H, W]

        # Ensure output matches input size (handles edge cases)
        if x_up.shape[2:] != (H, W):
            x_up = F.interpolate(x_up, size=(H, W), mode="bilinear", align_corners=False)

        return x_up, {"patchify_output": x_down, "stride": current_stride}


# =============================================================================
# CKCONV-BASED PATCHIFY/UNPATCHIFY (Continuous Kernels)
# =============================================================================


class SimpleContinuousKernel(nn.Module):
    """Simple MLP-based continuous kernel generator.

    Maps normalized coordinates in [-1, 1]^d to kernel values.
    The same kernel function can be sampled at any resolution.
    """

    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.out_channels = out_channels

        # Simple MLP: coordinates → kernel values
        layers = [nn.Linear(data_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, out_channels))
        self.mlp = nn.Sequential(*layers)

        # Initialize output layer with small values (for stable conv)
        with torch.no_grad():
            self.mlp[-1].weight.data *= 0.01
            self.mlp[-1].bias.data.zero_()

    def make_grid(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Create normalized coordinate grid in [-1, 1]^d."""
        grids = [torch.linspace(-1, 1, s, device=device) for s in shape]
        mesh = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1)
        return mesh  # [*shape, data_dim]

    def forward(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate kernel at given resolution.

        Args:
            shape: Spatial shape to sample at, e.g., (64, 64)
            device: Device to create tensors on

        Returns:
            Kernel of shape [out_channels, *shape]
        """
        grid = self.make_grid(shape, device)  # [*shape, data_dim]
        kernel = self.mlp(grid)  # [*shape, out_channels]
        # Rearrange to [out_channels, *shape]
        kernel = rearrange(kernel, "... c -> c ...")
        return kernel


class SIRENKernelWrapper(nn.Module):
    """Wrapper around SIRENKernelND for easy sampling at any resolution.

    Uses SIREN (sinusoidal representation networks) which are better at
    representing high-frequency continuous functions.

    Note: SIRENKernelND returns kernel of size (2*seq_len - 1) for full convolution.
    This wrapper extracts the center portion to get the requested size.
    """

    def __init__(
        self,
        data_dim: int,
        out_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        L_cache: int = 64,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.data_dim = data_dim

        # Import and use the actual SIREN kernel
        from nvsubquadratic.modules.kernels_nd import SIRENKernelND

        self.kernel = SIRENKernelND(
            out_dim=out_channels,
            data_dim=data_dim,
            mlp_hidden_dim=hidden_dim,
            num_layers=num_layers,
            embedding_dim=hidden_dim,
            omega_0=omega_0,
            L_cache=L_cache,
            use_bias=True,
            hidden_omega_0=hidden_omega_0,
        )

    def forward(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate kernel at given resolution.

        Args:
            shape: Spatial shape to sample at, e.g., (4, 4)
            device: Device (for compatibility, kernel handles this internally)

        Returns:
            Kernel of shape [out_channels, *shape]
        """
        # SIRENKernelND returns (2*seq_len - 1) size, so we request ceil((shape+1)/2)
        # and then extract the center portion
        request_shape = tuple((s + 1) // 2 + 1 for s in shape)
        kernel, _ = self.kernel(request_shape)  # [1, 2*h-1, 2*w-1, out_channels]
        kernel = kernel.squeeze(0)  # [2*h-1, 2*w-1, out_channels]

        # Extract center portion of requested size
        center_slices = []
        for i, s in enumerate(shape):
            full_size = kernel.shape[i]
            start = (full_size - s) // 2
            center_slices.append(slice(start, start + s))

        kernel = kernel[tuple(center_slices)]  # [shape[0], shape[1], out_channels]
        kernel = rearrange(kernel, "... c -> c ...")  # [out_channels, *shape]
        return kernel


class CKConvPatchifyUnpatchifyNet(nn.Module):
    """Patchify/Unpatchify using continuous kernels with standard conv2d.

    Key insight: The kernel is a continuous function K(x,y) = MLP([x,y]).
    We sample it at a fixed small size (e.g., 7x7) but the coordinates
    are always normalized to [-1, 1], making it resolution-agnostic.

    Uses F.conv2d and F.conv_transpose2d for proper forward/inverse relationship.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        init_stride: int = 4,
        kernel_size: int = 7,
        kernel_hidden_dim: int = 64,
        kernel_num_layers: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.init_stride = init_stride
        self.kernel_size = kernel_size

        # Learnable stride
        self.log_stride = nn.Parameter(torch.tensor([math.log(init_stride)] * 2))

        # Continuous kernel for patchify (in_channels -> hidden_channels)
        self.patchify_kernel = SimpleContinuousKernel(
            data_dim=2,
            out_channels=hidden_channels * in_channels,  # [out, in] flattened
            hidden_dim=kernel_hidden_dim,
            num_layers=kernel_num_layers,
        )

        # Continuous kernel for unpatchify (hidden_channels -> in_channels)
        self.unpatchify_kernel = SimpleContinuousKernel(
            data_dim=2,
            out_channels=in_channels * hidden_channels,  # [out, in] flattened
            hidden_dim=kernel_hidden_dim,
            num_layers=kernel_num_layers,
        )

    def get_stride(self) -> torch.Tensor:
        return torch.exp(self.log_stride)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        device = x.device
        current_stride = self.get_stride()

        stride_h = int(current_stride[0].item())
        stride_w = int(current_stride[1].item())
        down_h, down_w = H // stride_h, W // stride_w
        ks = self.kernel_size
        pad = ks // 2

        # === PATCHIFY ===
        # Generate kernel at fixed size (coordinates normalized to [-1, 1])
        patchify_k = self.patchify_kernel((ks, ks), device)  # [hidden*in, ks, ks]
        patchify_k = patchify_k.view(self.hidden_channels, self.in_channels, ks, ks)

        x_conv = F.conv2d(x, patchify_k, padding=pad)
        x_down = F.interpolate(x_conv, size=(down_h, down_w), mode="bilinear", align_corners=False)

        # === UNPATCHIFY ===
        x_up = F.interpolate(x_down, size=(H, W), mode="bilinear", align_corners=False)

        unpatchify_k = self.unpatchify_kernel((ks, ks), device)  # [in*hidden, ks, ks]
        unpatchify_k = unpatchify_k.view(self.in_channels, self.hidden_channels, ks, ks)

        output = F.conv2d(x_up, unpatchify_k, padding=pad)

        return output, {"patchify_output": x_down, "stride": current_stride}


class CKConvSharedKernelNet(nn.Module):
    """CKConv patchify/unpatchify with SHARED kernel using BILINEAR resize.

    Uses the SAME kernel for both directions:
    - Patchify: F.conv2d (stride=1) + bilinear downsample
    - Unpatchify: bilinear upsample + F.conv_transpose2d

    This uses bilinear interpolation for resizing, which we've shown
    loses information and cannot reach zero reconstruction loss.

    Compare with CKConvStridedNet which uses strided conv instead.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        init_stride: int = 4,
        kernel_size: int | None = None,  # If None, uses stride as kernel size
        kernel_hidden_dim: int = 64,
        kernel_num_layers: int = 3,
        use_siren: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.init_stride = init_stride
        self.kernel_size = kernel_size if kernel_size is not None else init_stride
        self.use_siren = use_siren

        # Learnable stride
        self.log_stride = nn.Parameter(torch.tensor([math.log(init_stride)] * 2))

        # SHARED continuous kernel
        if use_siren:
            self.shared_kernel = SIRENKernelWrapper(
                data_dim=2,
                out_channels=hidden_channels * in_channels,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
                L_cache=max(16, self.kernel_size),
            )
        else:
            self.shared_kernel = SimpleContinuousKernel(
                data_dim=2,
                out_channels=hidden_channels * in_channels,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
            )

    def get_stride(self) -> torch.Tensor:
        return torch.exp(self.log_stride)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        device = x.device
        current_stride = self.get_stride()

        stride_h = int(current_stride[0].item())
        stride_w = int(current_stride[1].item())
        down_h, down_w = H // stride_h, W // stride_w
        ks = self.kernel_size

        # Generate SHARED kernel
        kernel = self.shared_kernel((ks, ks), device)  # [hidden*in, ks, ks]
        kernel = kernel.view(self.hidden_channels, self.in_channels, ks, ks)

        # === PATCHIFY ===
        # Conv with stride=1, then bilinear downsample
        x_conv = F.conv2d(x, kernel, padding="same")
        x_down = F.interpolate(x_conv, size=(down_h, down_w), mode="bilinear", align_corners=False)

        # === UNPATCHIFY ===
        # Bilinear upsample, then conv_transpose
        x_up = F.interpolate(x_down, size=(H, W), mode="bilinear", align_corners=False)
        output = F.conv_transpose2d(x_up, kernel, padding=ks // 2)

        # Ensure output matches input size
        if output.shape[2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=False)

        return output, {"patchify_output": x_down, "stride": current_stride}


class CKConvStridedNet(nn.Module):
    """CKConv with STRIDED conv and conv_transpose (no bilinear!).

    This is the continuous-kernel version of the working Conv/ConvTranspose pattern:
    - Patchify: F.conv2d with stride (kernel sampled at kernel_size)
    - Unpatchify: F.conv_transpose2d with stride (SAME kernel)

    Key parameters:
    - kernel_size: Size of the sampled kernel (can be >= stride)
    - If kernel_size > stride, uses padding to maintain correct output sizes

    Since kernel is implicitly parameterized by MLP, larger kernels don't add params!
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        init_stride: int = 4,
        max_stride: int = 16,
        kernel_size: int | None = None,  # If None, uses stride as kernel size
        kernel_hidden_dim: int = 64,
        kernel_num_layers: int = 3,
        use_siren: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.init_stride = init_stride
        self.max_stride = max_stride
        self.kernel_size = kernel_size  # None means use stride
        self.use_siren = use_siren

        # Learnable stride (for future use - currently fixed)
        self.log_stride = nn.Parameter(torch.tensor([math.log(init_stride)] * 2))

        # SHARED continuous kernel
        max_ks = kernel_size if kernel_size is not None else max_stride
        if use_siren:
            self.shared_kernel = SIRENKernelWrapper(
                data_dim=2,
                out_channels=hidden_channels * in_channels,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
                omega_0=30.0,
                hidden_omega_0=30.0,
                L_cache=max_ks,
            )
        else:
            self.shared_kernel = SimpleContinuousKernel(
                data_dim=2,
                out_channels=hidden_channels * in_channels,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
            )

    def get_stride(self) -> torch.Tensor:
        return torch.exp(self.log_stride)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        device = x.device
        current_stride = self.get_stride()

        # Use integer stride for conv operations
        stride_int = int(current_stride[0].item())

        # Kernel size: use provided value or default to stride
        ks = self.kernel_size if self.kernel_size is not None else stride_int

        # Sample kernel at specified size
        kernel = self.shared_kernel((ks, ks), device)
        kernel = kernel.view(self.hidden_channels, self.in_channels, ks, ks)

        # Padding for larger kernels (to maintain output size relationship)
        pad = (ks - stride_int) // 2

        # === PATCHIFY ===
        # Strided conv: downsample by stride factor
        x_down = F.conv2d(x, kernel, stride=stride_int, padding=pad)

        # === UNPATCHIFY ===
        # Strided conv_transpose: upsample by stride factor
        output = F.conv_transpose2d(x_down, kernel, stride=stride_int, padding=pad)

        return output, {"patchify_output": x_down, "stride": current_stride}


class CKConvGaussianMaskNet(nn.Module):
    """CKConv with Gaussian mask for learnable stride.

    Strategy:
    1. Always sample kernel at max_stride size
    2. Apply Gaussian mask controlled by current_stride (differentiable)
    3. Use F.conv2d with integer stride (rounded from continuous)

    The Gaussian mask makes the kernel "act like" a smaller kernel,
    allowing gradients to flow through the stride parameter.

    NOTE: This architecture CANNOT reach zero loss due to padding issues.
    But it demonstrates that gradients CAN flow through stride.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        init_stride: int = 4,
        max_stride: int = 16,
        kernel_hidden_dim: int = 64,
        kernel_num_layers: int = 3,
        use_siren: bool = True,
        clip_value: float = 0.01,  # Gaussian cutoff threshold
        use_gaussian_mask: bool = True,  # Can disable for ablation
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.init_stride = init_stride
        self.max_stride = max_stride
        self.use_siren = use_siren
        self.clip_value = clip_value
        self.use_gaussian_mask = use_gaussian_mask

        # Learnable stride
        self.log_stride = nn.Parameter(torch.tensor([math.log(init_stride)] * 2))

        # Precompute cutoff factor
        self._cutoff_factor = math.sqrt(-2 * math.log(clip_value))

        # Create normalized kernel grid [-1, 1] for max_stride size
        coords = torch.linspace(-1, 1, max_stride)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("_kernel_grid_y", grid_y)
        self.register_buffer("_kernel_grid_x", grid_x)

        # SHARED continuous kernel (always sampled at max_stride)
        if use_siren:
            self.shared_kernel = SIRENKernelWrapper(
                data_dim=2,
                out_channels=hidden_channels * in_channels,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
                omega_0=30.0,
                hidden_omega_0=30.0,
                L_cache=max_stride,
            )
        else:
            self.shared_kernel = SimpleContinuousKernel(
                data_dim=2,
                out_channels=hidden_channels * in_channels,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
            )

    def get_stride(self) -> torch.Tensor:
        return torch.exp(self.log_stride)

    def _compute_gaussian_mask(self, stride: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian mask based on current stride."""
        kernel_frac_y = stride[0] / self.max_stride
        kernel_frac_x = stride[1] / self.max_stride
        sigma_y = kernel_frac_y / self._cutoff_factor
        sigma_x = kernel_frac_x / self._cutoff_factor
        exponent = -0.5 * ((self._kernel_grid_y / sigma_y) ** 2 + (self._kernel_grid_x / sigma_x) ** 2)
        mask = torch.exp(exponent)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        device = x.device
        current_stride = self.get_stride()

        stride_int = int(torch.round(current_stride[0]).item())
        stride_int = max(1, min(stride_int, self.max_stride))

        kernel = self.shared_kernel((self.max_stride, self.max_stride), device)
        kernel = kernel.view(self.hidden_channels, self.in_channels, self.max_stride, self.max_stride)

        if self.use_gaussian_mask:
            mask = self._compute_gaussian_mask(current_stride)
            kernel = kernel * mask

        pad = (self.max_stride - stride_int) // 2
        x_down = F.conv2d(x, kernel, stride=stride_int, padding=pad)
        output = F.conv_transpose2d(x_down, kernel, stride=stride_int, padding=pad)

        if output.shape[2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=False)

        return output, {"patchify_output": x_down, "stride": current_stride, "stride_int": stride_int}


class CKConvLearnableStrideNet(nn.Module):
    """CKConv with LEARNABLE stride via blending between adjacent integer strides.

    Strategy for learnable stride while maintaining near-zero reconstruction:
    1. Compute floor and ceil of continuous stride
    2. Run patchify/unpatchify at BOTH strides
    3. Blend outputs based on fractional part (differentiable!)

    This provides smooth gradients through stride while each individual
    stride operation can reach zero reconstruction.

    Example: stride=5.3
    - stride_lo=5, stride_hi=6, alpha=0.3
    - output = (1-0.3)*output_at_5 + 0.3*output_at_6
    - Gradient flows through alpha back to stride
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        init_stride: int = 4,
        max_stride: int = 16,
        kernel_hidden_dim: int = 64,
        kernel_num_layers: int = 3,
        use_siren: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.init_stride = init_stride
        self.max_stride = max_stride
        self.use_siren = use_siren

        # Learnable stride (continuous)
        self.log_stride = nn.Parameter(torch.tensor([math.log(init_stride)] * 2))

        # SHARED continuous kernel
        if use_siren:
            self.shared_kernel = SIRENKernelWrapper(
                data_dim=2,
                out_channels=hidden_channels * in_channels,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
                omega_0=30.0,
                hidden_omega_0=1.0,
                L_cache=max_stride,
            )
        else:
            self.shared_kernel = SimpleContinuousKernel(
                data_dim=2,
                out_channels=hidden_channels * in_channels,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
            )

    def get_stride(self) -> torch.Tensor:
        return torch.exp(self.log_stride)

    def _patchify_unpatchify(self, x: torch.Tensor, stride_int: int, target_size: tuple, device) -> torch.Tensor:
        """Run patchify/unpatchify at a specific integer stride."""
        kernel = self.shared_kernel((stride_int, stride_int), device)
        kernel = kernel.view(self.hidden_channels, self.in_channels, stride_int, stride_int)

        x_down = F.conv2d(x, kernel, stride=stride_int, padding=0)
        output = F.conv_transpose2d(x_down, kernel, stride=stride_int, padding=0)

        # Ensure output matches target size
        if output.shape[2:] != target_size:
            output = F.interpolate(output, size=target_size, mode="bilinear", align_corners=False)

        return output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = x.shape
        device = x.device
        current_stride = self.get_stride()

        # Use mean of x and y strides for simplicity
        # exp(log_stride) is already strictly positive, so stride >= 0 naturally
        stride_val = current_stride.mean()

        # Note: No clamping here to preserve gradients
        # The floor/ceil below will handle out-of-range values gracefully

        # Get floor and ceil strides (stride=1 is valid, means no downsampling)
        stride_lo = int(torch.floor(stride_val).item())
        stride_hi = int(torch.ceil(stride_val).item())
        stride_lo = max(1, min(stride_lo, self.max_stride))
        stride_hi = max(1, min(stride_hi, self.max_stride))

        # Compute blend factor (differentiable!)
        alpha = stride_val - stride_lo  # This is differentiable!

        if stride_lo == stride_hi:
            # Exactly integer stride - no blending needed
            output = self._patchify_unpatchify(x, stride_lo, (H, W), device)
        else:
            # Blend between two strides
            output_lo = self._patchify_unpatchify(x, stride_lo, (H, W), device)
            output_hi = self._patchify_unpatchify(x, stride_hi, (H, W), device)

            # Blend (alpha provides gradient to stride)
            output = (1 - alpha) * output_lo + alpha * output_hi

        return output, {
            "stride": current_stride,
            "stride_lo": stride_lo,
            "stride_hi": stride_hi,
            "alpha": alpha if stride_lo != stride_hi else torch.tensor(0.0),
        }


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================


def train_and_evaluate(model, target, num_steps: int, experiment_name: str, lr: float = 0.001) -> dict:
    """Train model and return results."""
    device = target.device
    model = model.to(device)

    # Count params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Check if model has learnable alphas
    has_alphas = hasattr(model, "get_alphas")

    # Initial loss
    with torch.no_grad():
        output, _ = model(target)
        initial_loss = F.mse_loss(output, target).item()

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()
        output, _ = model(target)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (step + 1) % 200 == 0 or step == 0:
            extra_str = ""
            if has_alphas:
                alphas = model.get_alphas()
                if alphas.get("patchify_alpha") is not None:
                    extra_str = f", alpha_p={alphas['patchify_alpha']:.4f}, alpha_u={alphas['unpatchify_alpha']:.4f}"
            # Show learned stride if available
            if hasattr(model, "get_stride"):
                stride = model.get_stride()
                if isinstance(stride, torch.Tensor) and stride.requires_grad:
                    extra_str += f", stride=[{stride[0].item():.2f}, {stride[1].item():.2f}]"
            print(f"  [{experiment_name}] Step {step + 1:5d}: Loss = {loss.item():.10f}{extra_str}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        output, _ = model(target)
        final_loss = F.mse_loss(output, target).item()
        correlation = torch.corrcoef(torch.stack([output.flatten(), target.flatten()]))[0, 1].item()

    result = {
        "experiment": experiment_name,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "correlation": correlation,
        "trainable_params": trainable_params,
        "reached_zero": final_loss < 1e-6,
    }

    # Add final alphas if available
    if has_alphas:
        alphas = model.get_alphas()
        result["final_patchify_alpha"] = alphas.get("patchify_alpha")
        result["final_unpatchify_alpha"] = alphas.get("unpatchify_alpha")

    return result


def run_experiment(experiment_name: str, num_steps: int = 1000, lr: float = 0.001) -> dict:
    """Run a single experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target = get_test_target(device=device)

    in_channels = TEST_CONFIG["in_channels"]
    hidden_channels = TEST_CONFIG["hidden_channels"]
    stride = TEST_CONFIG["stride"]

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'=' * 70}")

    if experiment_name == "baseline":
        # Original dual path with PixelShuffle
        model = ExperimentalDualPathNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            spatial_upsample_mode="pixel_shuffle",
            spectral_upsample_mode="bilinear",
        )
    elif experiment_name == "conv_transpose_spatial":
        # Replace PixelShuffle with ConvTranspose2d in spatial path
        model = ExperimentalDualPathNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            spatial_upsample_mode="conv_transpose",
            spectral_upsample_mode="bilinear",
        )
    elif experiment_name == "conv_transpose_both":
        # Use ConvTranspose2d in both paths
        model = ExperimentalDualPathNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            spatial_upsample_mode="conv_transpose",
            spectral_upsample_mode="conv_transpose",
        )
    elif experiment_name == "spectral_only_bilinear":
        # Spectral path only with bilinear upsampling
        model = SpectralOnlyNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            upsample_mode="bilinear",
        )
    elif experiment_name == "spectral_only_conv_transpose":
        # Spectral path only with ConvTranspose2d upsampling
        model = SpectralOnlyNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            upsample_mode="conv_transpose",
        )
    elif experiment_name == "spatial_only_pixel_shuffle":
        # Spatial path only with PixelShuffle
        model = SpatialOnlyNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            upsample_mode="pixel_shuffle",
        )
    elif experiment_name == "spatial_only_conv_transpose":
        # Spatial path only with ConvTranspose2d
        model = SpatialOnlyNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            upsample_mode="conv_transpose",
        )
    elif experiment_name == "no_mask_spectral":
        # Like spectral path but WITHOUT the spectral mask (tests if mask is the problem)
        model = NoMaskSpectralNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            upsample_mode="conv_transpose",
        )
    elif experiment_name == "concat_conv_transpose":
        # Use concatenation instead of sum, with ConvTranspose in both paths
        model = ExperimentalDualPathNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            spatial_upsample_mode="conv_transpose",
            spectral_upsample_mode="conv_transpose",
            combine_mode="concat",
        )
    elif experiment_name == "learnable_blend":
        # Learnable alpha: (1-alpha)*spatial + alpha*spectral, with ConvTranspose
        model = ExperimentalDualPathNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            spatial_upsample_mode="conv_transpose",
            spectral_upsample_mode="conv_transpose",
            combine_mode="learnable_blend",
        )
    elif experiment_name == "learnable_blend_spatial_init":
        # Same as learnable_blend but initialize alpha near 0 (favor spatial)
        model = ExperimentalDualPathNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            spatial_upsample_mode="conv_transpose",
            spectral_upsample_mode="conv_transpose",
            combine_mode="learnable_blend",
        )
        # Reinitialize alpha_logit to -5 so sigmoid(-5) ≈ 0.0067 (almost pure spatial)
        if hasattr(model.patchify, "alpha_logit"):
            model.patchify.alpha_logit.data.fill_(-5.0)
        if hasattr(model.unpatchify, "alpha_logit"):
            model.unpatchify.alpha_logit.data.fill_(-5.0)
    elif experiment_name == "spatial_only_pure":
        # Spatial path ONLY with no spectral path at all (like conventional patchify)
        model = SpatialOnlyNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            stride=stride,
            upsample_mode="conv_transpose",
        )
    elif experiment_name == "learnable_stride_conv_transpose":
        # Learnable stride with ConvTranspose and bilinear bottleneck
        model = LearnableStrideConvTransposeNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            max_stride=16,
        )
    elif experiment_name == "ckconv_separate":
        # CKConv-based with separate patchify/unpatchify kernels
        model = CKConvPatchifyUnpatchifyNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
        )
    elif experiment_name == "ckconv_shared":
        # CKConv-based with SHARED kernel + bilinear (simple MLP)
        model = CKConvSharedKernelNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            kernel_size=stride,  # Same kernel size as stride
            use_siren=False,
        )
    elif experiment_name == "ckconv_shared_siren":
        # CKConv-based with SHARED kernel + bilinear (SIREN)
        model = CKConvSharedKernelNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            kernel_size=stride,
            use_siren=True,
        )
    elif experiment_name == "ckconv_strided":
        # CKConv with strided conv/conv_transpose (no bilinear!)
        model = CKConvStridedNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            max_stride=16,
            use_siren=False,
        )
    elif experiment_name == "ckconv_siren":
        # CKConv with SIREN kernel, kernel_size=stride (default)
        model = CKConvStridedNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            max_stride=16,
            kernel_size=None,  # Default: kernel_size = stride
            use_siren=True,
        )
    elif experiment_name == "ckconv_siren_ks8":
        # CKConv SIREN with larger kernel (8x8 for stride=4)
        model = CKConvStridedNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            max_stride=16,
            kernel_size=8,
            use_siren=True,
        )
    elif experiment_name == "ckconv_siren_ks16":
        # CKConv SIREN with even larger kernel (16x16 for stride=4)
        model = CKConvStridedNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            max_stride=16,
            kernel_size=16,
            use_siren=True,
        )
    elif experiment_name == "ckconv_gaussian":
        # CKConv with Gaussian mask (for learnable stride)
        model = CKConvGaussianMaskNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            max_stride=16,
            use_siren=True,
            use_gaussian_mask=True,
        )
    elif experiment_name == "ckconv_gaussian_no_mask":
        # CKConv Gaussian architecture but WITHOUT mask (ablation)
        model = CKConvGaussianMaskNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            max_stride=16,
            use_siren=True,
            use_gaussian_mask=False,
        )
    elif experiment_name == "ckconv_learnable_stride":
        # CKConv with learnable stride via blending
        model = CKConvLearnableStrideNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            init_stride=stride,
            max_stride=16,
            use_siren=True,
        )
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    print(f"Model: {model}")
    result = train_and_evaluate(model, target, num_steps, experiment_name, lr=lr)

    # Print result
    status = "✅ ZERO" if result["reached_zero"] else "❌ NOT ZERO"
    print(f"\n{status}: final_loss={result['final_loss']:.10f}, correlation={result['correlation']:.6f}")

    return result


def run_all_experiments(num_steps: int = 1000, lr: float = 0.001) -> list[dict]:
    """Run all experiments and return results."""
    experiments = [
        "baseline",
        "conv_transpose_spatial",
        "conv_transpose_both",
        "concat_conv_transpose",
        "learnable_blend",
        "learnable_blend_spatial_init",
        "spectral_only_bilinear",
        "spectral_only_conv_transpose",
        "spatial_only_pixel_shuffle",
        "spatial_only_conv_transpose",
        "no_mask_spectral",
    ]

    results = []
    for exp in experiments:
        result = run_experiment(exp, num_steps, lr=lr)
        results.append(result)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 90)
    print(f"{'Experiment':<35} {'Final Loss':<18} {'Correlation':<12} {'Status'}")
    print("-" * 90)
    for r in results:
        status = "✅ ZERO" if r["reached_zero"] else "❌ NOT ZERO"
        print(f"{r['experiment']:<35} {r['final_loss']:<18.10f} {r['correlation']:<12.6f} {status}")
    print("=" * 90)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run DualPath reconstruction experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=[
            "all",
            "baseline",
            "conv_transpose_spatial",
            "conv_transpose_both",
            "concat_conv_transpose",
            "learnable_blend",
            "learnable_blend_spatial_init",
            "spectral_only_bilinear",
            "spectral_only_conv_transpose",
            "spatial_only_pixel_shuffle",
            "spatial_only_conv_transpose",
            "no_mask_spectral",
            "learnable_stride_conv_transpose",
            "ckconv_separate",
            "ckconv_shared",
            "ckconv_shared_siren",
            "ckconv_strided",
            "ckconv_siren",
            "ckconv_siren_ks8",
            "ckconv_siren_ks16",
            "ckconv_gaussian",
            "ckconv_gaussian_no_mask",
            "ckconv_learnable_stride",
        ],
        help="Which experiment to run",
    )
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    if args.experiment == "all":
        run_all_experiments(args.num_steps, lr=args.lr)
    else:
        run_experiment(args.experiment, args.num_steps, lr=args.lr)


if __name__ == "__main__":
    main()
