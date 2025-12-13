# David W. Romero, 2025-09-09

"""Patchify and Unpatchify layers as ConvND and ConvTransposeND layers.

Also includes SpectralUnpatchify for bilinear upsampling + output projection.

Usage test:
    PYTHONPATH=. python nvsubquadratic/modules/patchify.py
"""

from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate


# Mapping from data_dim to Conv and ConvTranspose classes
_CONV_CLASSES = {
    1: torch.nn.Conv1d,
    2: torch.nn.Conv2d,
    3: torch.nn.Conv3d,
}

_CONV_TRANSPOSE_CLASSES = {
    1: torch.nn.ConvTranspose1d,
    2: torch.nn.ConvTranspose2d,
    3: torch.nn.ConvTranspose3d,
}


class Patchify(torch.nn.Module):
    """Conv-based image patchification (channels-last input).

    This mirrors the ViT/timm approach where a Conv with kernel_size=stride=patch_size
    produces one embedding per patch location (non-overlapping patches).

    Input shape:  [B, *spatial_dims, in_features] (channels-last, e.g., BHWC)
    Output shape: [B, *spatial_dims // patch_size, out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        patch_size: int,
        stride: int | None = None,
    ):
        """Initialize the Patchify layer.

        Args:
            in_features: The number of input channels.
            out_features: The number of output channels (embedding dimension).
            data_dim: The spatial dimensionality (1, 2, or 3).
            patch_size: The size of each patch (kernel_size for the conv).
            stride: The stride for the conv. Defaults to patch_size (non-overlapping).
        """
        super().__init__()
        if data_dim not in _CONV_CLASSES:
            raise ValueError(f"data_dim must be 1, 2, or 3, got {data_dim}")

        if stride is None:
            stride = patch_size  # Default: non-overlapping patches (ViT-style)

        self.data_dim = data_dim
        self.patch_size = patch_size
        self.stride = stride

        conv_class = _CONV_CLASSES[data_dim]
        self.conv = conv_class(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Patchify layer.

        Args:
            x: The input tensor of shape [B, *spatial_dims, in_features].

        Returns:
            The output tensor of shape [B, *spatial_dims // stride, out_features].
        """
        # Channels-last -> channels-first for ConvNd
        x = rearrange(x, "b ... c -> b c ...")

        # Apply conv
        y = self.conv(x)

        # Channels-first -> channels-last
        y = rearrange(y, "b c ... -> b ... c")
        return y


class Unpatchify(torch.nn.Module):
    """Inverse of Patchify for channels-last inputs (supports 1D/2D/3D).

    Uses ConvTranspose to upsample from patch resolution back to original resolution.

    Input shape:  [B, *spatial_dims, in_features] (channels-last)
    Output shape: [B, *spatial_dims * stride, out_features]

    If exact spatial size control is required, pass output_spatial_shape to forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        patch_size: int,
        stride: int | None = None,
    ):
        """Initialize the Unpatchify layer.

        Args:
            in_features: The number of input channels (embedding dimension).
            out_features: The number of output channels.
            data_dim: The spatial dimensionality (1, 2, or 3).
            patch_size: The size of each patch (kernel_size for the deconv).
            stride: The stride for the deconv. Defaults to patch_size (inverse of non-overlapping).
        """
        super().__init__()
        if data_dim not in _CONV_TRANSPOSE_CLASSES:
            raise ValueError(f"data_dim must be 1, 2, or 3, got {data_dim}")

        if stride is None:
            stride = patch_size  # Default: inverse of non-overlapping patches

        self.data_dim = data_dim
        self.patch_size = patch_size
        self.stride = stride

        deconv_class = _CONV_TRANSPOSE_CLASSES[data_dim]
        self.deconv = deconv_class(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor, output_spatial_shape: Tuple[int, ...] | None = None) -> torch.Tensor:
        """Forward pass of the Unpatchify layer.

        Args:
            x: The input tensor of shape [B, *spatial_dims, in_features].
            output_spatial_shape: The desired output spatial shape (optional).

        Returns:
            The output tensor of shape [B, *spatial_dims * stride, out_features].
        """
        expected_dims = self.data_dim + 2  # batch + spatial_dims + channels
        assert x.dim() == expected_dims, (
            f"Expected {expected_dims}D tensor for data_dim={self.data_dim}, got {x.dim()}D tensor with shape {tuple(x.shape)}"
        )

        # Channels-last -> channels-first for ConvTransposeNd
        x_bc = rearrange(x, "b ... c -> b c ...")

        # Apply deconvolution (optionally with target output size)
        if output_spatial_shape is None:
            y_bc = self.deconv(x_bc)
        else:
            y_bc = self.deconv(x_bc, output_size=tuple(int(v) for v in output_spatial_shape))

        # Channels-first -> channels-last
        y = rearrange(y_bc, "b c ... -> b ... c")
        return y


class SpectralUnpatchify(torch.nn.Module):
    """Spectral unpatchification via bilinear interpolation + output projection.

    Performs upsampling by:
    1. Bilinear/bicubic/nearest interpolation to target spatial size
    2. Output projection (e.g., Linear, CKConv) for channel mapping and refinement

    Input shape:  [B, H_small, W_small, C] (channels-last) or [B, C, H_small, W_small] (BHL)
    Output shape: [B, target_H, target_W, C_out] (channels-last) or [B, C_out, target_H, target_W] (BHL)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        output_proj_cfg: LazyConfig,
        interpolation_mode: Literal["bilinear", "bicubic", "nearest"] = "bilinear",
        align_corners: bool = False,
    ):
        """Initialize the SpectralUnpatchify layer.

        Args:
            in_features: The number of input channels.
            out_features: The number of output channels.
            data_dim: The spatial dimensionality (currently only 2 is supported).
            output_proj_cfg: LazyConfig for the output projection layer. Should be fully
                configured (e.g., include hidden_dim for CKConv, or in_features/out_features
                for Linear).
            interpolation_mode: Interpolation mode for upsampling. One of:
                - "bilinear": Bilinear interpolation (default, good balance)
                - "bicubic": Bicubic interpolation (smoother, but slower)
                - "nearest": Nearest neighbor (fastest, but blocky)
            align_corners: If True, align corners for interpolation.
                Only used for bilinear/bicubic modes.
        """
        super().__init__()

        assert data_dim == 2, f"data_dim must be 2, got {data_dim}"
        if interpolation_mode == "nearest":
            assert not align_corners, "align_corners must be False for nearest interpolation."

        self.data_dim = data_dim
        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners if interpolation_mode != "nearest" else None

        self.output_proj = instantiate(output_proj_cfg)

    def forward(
        self,
        x: torch.Tensor,
        target_shape: Tuple[int, int],
        is_bhl_input: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the SpectralUnpatchify layer.

        Args:
            x: Input tensor of shape [B, H, W, C] (channels-last).
            target_shape: Target spatial shape (H_out, W_out) for the upsampled output.
            is_bhl_input: Whether the input is in BHL format, i.e., (B, C, H, W).
                Default is False.

        Returns:
            Upsampled tensor of shape [B, H_out, W_out, C_out].
        """
        if not is_bhl_input:
            x = rearrange(x, "b h w c -> b c h w")

        # Bilinear interpolation to target size (skip if already at target size)
        current_shape = x.shape[2:]
        if current_shape != target_shape:  # Skip interpolation if already at target size
            if self.interpolation_mode == "nearest":
                x = F.interpolate(x, size=target_shape, mode=self.interpolation_mode)
            else:
                x = F.interpolate(x, size=target_shape, mode=self.interpolation_mode, align_corners=self.align_corners)

        # If we have a linear output projection, we need to convert to channels-last format before applying the projection.
        if isinstance(self.output_proj, torch.nn.Linear):
            x = rearrange(x, "b c h w -> b h w c")
            x = self.output_proj(x)
            if is_bhl_input:
                x = rearrange(x, "b h w c -> b c h w")
            return x

        # Check if output_proj accepts is_bhl_input (e.g., CKConvND) or not (e.g., Conv2d)
        if hasattr(self.output_proj, "forward") and "is_bhl_input" in self.output_proj.forward.__code__.co_varnames:
            x = self.output_proj(x, is_bhl_input=True)
        else:
            x = self.output_proj(x)

        if not is_bhl_input:
            x = rearrange(x, "b c h w -> b h w c")

        return x

    def extra_repr(self) -> str:
        """Additional string when printing the module."""
        return (
            f"data_dim={self.data_dim}, "
            f"interpolation_mode={self.interpolation_mode}, "
            f"align_corners={self.align_corners}, "
        )


class SpectralPatchify(torch.nn.Module):
    """Spectral patchification with decoupled convolution and spectral masking.

    This module decouples the convolution (optional) from the spectral masking,
    allowing step-by-step validation:

    1. **Pre-conv** (optional): Apply a convolution before spectral downsampling.
       This can encode spatial information into channels before frequency cropping.
    2. **Spectral downsampling**: Apply FFT, spectral masking, and IFFT to
       downsample to the target resolution determined by the mask.

    The spectral mask controls the effective stride/downsampling factor and can
    be learnable (e.g., SpectralGaussianMaskND, SpectralLinearMaskND).

    Input shape:  [B, H, W, C] (channels-last) or [B, C, H, W] (BHL)
    Output shape: [B, H_out, W_out, C] or [B, C, H_out, W_out] depending on input format

    Args:
        in_features: Number of input channels.
        out_features: Number of output channels (only used if conv_cfg is provided).
        data_dim: Spatial dimensionality (currently only 2 is supported).
        spectral_mask_cfg: LazyConfig for the spectral mask module (e.g., SpectralGaussianMaskND).
            The mask determines the downsampling factor.
        conv_cfg: Optional LazyConfig for the pre-downsampling convolution.
            If None, no convolution is applied before spectral downsampling.
            If provided, should be a CKConvND or similar config.

    Example:
        >>> from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND
        >>> spectral_patch = SpectralPatchify(
        ...     in_features=32,
        ...     out_features=64,
        ...     data_dim=2,
        ...     spectral_mask_cfg=LazyConfig(SpectralGaussianMaskND)(
        ...         data_dim=2, clip_value=0.1, init_stride_value=2.0
        ...     ),
        ...     conv_cfg=LazyConfig(CKConvND)(...),  # Optional
        ... )
        >>> x = torch.randn(1, 64, 64, 32)  # [B, H, W, C]
        >>> y = spectral_patch(x, is_bhl_input=False)  # [B, 32, 32, 64]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        spectral_mask_cfg: LazyConfig,
        conv_cfg: LazyConfig | None = None,
    ):
        """Initialize the SpectralPatchify layer.

        Args:
            in_features: Number of input channels.
            out_features: Number of output channels.
            data_dim: Spatial dimensionality (currently only 2 is supported).
            spectral_mask_cfg: LazyConfig for the spectral mask module.
            conv_cfg: Optional LazyConfig for the pre-downsampling convolution.
        """
        super().__init__()

        assert data_dim == 2, f"SpectralPatchify only supports data_dim=2, got {data_dim}"

        self.data_dim = data_dim
        self.in_features = in_features
        self.out_features = out_features

        # Spectral mask for learnable stride
        self.spectral_mask = instantiate(spectral_mask_cfg)

        # Optional pre-downsampling convolution
        if conv_cfg is not None:
            # The conv_cfg should be fully configured (including hidden_dim for CKConv)
            # We just instantiate it directly
            self.conv = instantiate(conv_cfg)
            self._has_conv = True
        else:
            self.conv = None
            self._has_conv = False
            # Ensure in_features == out_features when no conv is used
            assert in_features == out_features, (
                f"Without conv, in_features ({in_features}) must equal out_features ({out_features})"
            )

    def get_stride(self) -> torch.Tensor:
        """Get the current effective stride from the spectral mask.

        Returns:
            torch.Tensor: Stride per dimension, shape [data_dim].
        """
        return self.spectral_mask.get_stride()

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute the output spatial shape for a given input shape.

        Args:
            input_shape: Input spatial shape (H, W) for 2D.

        Returns:
            Tuple[int, ...]: Output spatial shape after downsampling.
        """
        stride = self.get_stride()
        return tuple(int(s / stride[i].item()) for i, s in enumerate(input_shape))

    def forward(
        self,
        x: torch.Tensor,
        is_bhl_input: bool = False,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """Apply spectral downsampling (with optional pre-conv).

        Args:
            x: Input tensor of shape [B, H, W, C] (channels-last) or [B, C, H, W] (BHL).
            is_bhl_input: Whether the input is in BHL format (B, C, H, W). Default False.
            return_intermediates: If True, return a dict with intermediate results
                for debugging/validation. Default False.

        Returns:
            Downsampled tensor. If return_intermediates=True, also returns a dict
            with key 'spectral_mask'.
        """
        from nvsubquadratic.ops.spectral_masking import spectral_downsampling2d_bhl

        # Convert to BHL format if needed
        if not is_bhl_input:
            x = rearrange(x, "b h w c -> b c h w")

        # Get input spatial shape
        H, W = x.shape[2], x.shape[3]

        # Step 1: Optional pre-downsampling convolution
        if self._has_conv:
            # Check if conv accepts is_bhl_input (e.g., CKConvND) or not (e.g., Conv2d)
            if hasattr(self.conv, "forward") and "is_bhl_input" in self.conv.forward.__code__.co_varnames:
                x = self.conv(x, is_bhl_input=True)
            else:
                x = self.conv(x)

        # Step 2: Generate spectral mask for current input size
        # The mask module computes the crop size based on its learned stride
        spectral_mask = self.spectral_mask((H, W))  # [1, sM_x, sM_y]

        # Add channel dimension for broadcasting: [1, sM_x, sM_y] -> [1, 1, sM_x, sM_y]
        # Then expand to match channels: [1, C, sM_x, sM_y]
        C_out = x.shape[1]
        spectral_mask = spectral_mask.unsqueeze(1).expand(-1, C_out, -1, -1)

        # Step 3: Apply spectral downsampling
        y = spectral_downsampling2d_bhl(x, spectral_mask)

        # Convert back to channels-last if input was channels-last
        if not is_bhl_input:
            y = rearrange(y, "b c h w -> b h w c")

        if return_intermediates:
            return y, {"spectral_mask": spectral_mask}
        return y

    def forward_conv_only(self, x: torch.Tensor, is_bhl_input: bool = False) -> torch.Tensor:
        """Apply only the convolution (skip spectral downsampling).

        Useful for debugging/validation to see what the conv does before downsampling.

        Args:
            x: Input tensor of shape [B, H, W, C] (channels-last) or [B, C, H, W] (BHL).
            is_bhl_input: Whether the input is in BHL format.

        Returns:
            Tensor after convolution (same spatial size as input).
        """
        if not self._has_conv:
            return x

        if not is_bhl_input:
            x = rearrange(x, "b h w c -> b c h w")

        # Check if conv accepts is_bhl_input (e.g., CKConvND) or not (e.g., Conv2d)
        if hasattr(self.conv, "forward") and "is_bhl_input" in self.conv.forward.__code__.co_varnames:
            y = self.conv(x, is_bhl_input=True)
        else:
            y = self.conv(x)

        if not is_bhl_input:
            y = rearrange(y, "b c h w -> b h w c")

        return y

    def forward_spectral_only(self, x: torch.Tensor, is_bhl_input: bool = False) -> torch.Tensor:
        """Apply only the spectral downsampling (skip convolution).

        Useful for debugging/validation to see pure spectral downsampling behavior.

        Args:
            x: Input tensor of shape [B, H, W, C] (channels-last) or [B, C, H, W] (BHL).
            is_bhl_input: Whether the input is in BHL format.

        Returns:
            Spectrally downsampled tensor.
        """
        from nvsubquadratic.ops.spectral_masking import spectral_downsampling2d_bhl

        if not is_bhl_input:
            x = rearrange(x, "b h w c -> b c h w")

        H, W = x.shape[2], x.shape[3]
        C = x.shape[1]

        # Generate spectral mask
        spectral_mask = self.spectral_mask((H, W))  # [1, sM_x, sM_y]
        spectral_mask = spectral_mask.unsqueeze(1).expand(-1, C, -1, -1)

        # Apply spectral downsampling
        y = spectral_downsampling2d_bhl(x, spectral_mask)

        if not is_bhl_input:
            y = rearrange(y, "b c h w -> b h w c")

        return y

    def extra_repr(self) -> str:
        """Additional string when printing the module."""
        return (
            f"data_dim={self.data_dim}, "
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"has_conv={self._has_conv}"
        )


class DualPathPatchify(torch.nn.Module):
    """Dual-path patchification combining spectral and spatial downsampling.

    This module combines:
    - **Spectral path**: Anti-aliased low-pass filtering via SpectralPatchify
    - **Spatial path**: Standard strided convolution preserving high-frequency (aliased) content

    The outputs are summed to provide both clean low-frequencies and aliased high-frequencies,
    enabling perfect reconstruction when paired with DualPathUnpatchify.

    The spectral path's stride is learnable, and the spatial path adapts to match.

    Args:
        in_features: Number of input channels.
        out_features: Number of output channels (should be >= stride² x in_features to avoid bottleneck).
        data_dim: Spatial dimensionality (currently only 2 is supported).
        init_stride: Initial stride value for both paths.
        max_stride: Maximum stride the architecture supports (used for spatial path kernel size).
        spectral_mask_cfg: LazyConfig for the spectral mask (e.g., SpectralLinearMaskND).
        conv_cfg: Optional LazyConfig for the spectral path's pre-conv.
        freeze_spectral_mask: Whether to freeze the spectral mask parameters.

    Input shape:  [B, C, H, W] (BHL format)
    Output shape: [B, out_features, H/stride, W/stride]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        spectral_patchify_cfg: LazyConfig,  # LazyConfig["SpectralPatchify"]
        # conv_cfg: LazyConfig | None,
        # init_stride: int = 4,
        max_stride: int = 16,
        freeze_spectral_mask: bool = False,
        **kwargs,  # For compatibility with ResidualNetwork and other callers
    ):
        """Initialize the DualPathPatchify module."""
        super().__init__()

        assert data_dim == 2, f"DualPathPatchify only supports data_dim=2, got {data_dim}"

        # kwargs are ignored - for compatibility with ResidualNetwork fallback (e.g., hidden_dim)

        self.data_dim = data_dim
        self.in_features = in_features
        self.out_features = out_features
        # self.init_stride = init_stride
        self.max_stride = max_stride

        # Import here to avoid circular imports
        # from nvsubquadratic.modules.masks_nd import SpectralLinearMaskND

        # # Default spectral mask config if not provided
        # if spectral_mask_cfg is None:
        #     spectral_mask_cfg = LazyConfig(SpectralLinearMaskND)(
        #         data_dim=data_dim,
        #         transition_fraction=0.1,
        #         init_stride_value=float(init_stride),
        #         max_stride_value=float(max_stride),
        #     )

        # # Default conv config if not provided
        # if conv_cfg is None:
        #     conv_cfg = LazyConfig(torch.nn.Conv2d)(
        #         in_channels=in_features,
        #         out_channels=out_features,
        #         kernel_size=6,
        #         padding="same",
        #     )

        # === SPECTRAL PATH ===
        self.spectral_patchify = instantiate(spectral_patchify_cfg)
        # self.spectral_patchify = SpectralPatchify(
        #     in_features=in_features,
        #     out_features=out_features,
        #     data_dim=data_dim,
        #     spectral_mask_cfg=spectral_mask_cfg,
        #     conv_cfg=conv_cfg,
        # )
        assert self.max_stride == self.spectral_patchify.spectral_mask.max_stride_value, (
            "max_stride must be equal to the max_stride of the spectral mask"
        )

        # === SPATIAL PATH ===
        # Conv with kernel_size=max_stride, stride=1, padding=0
        # Output is then subsampled using the spectral path's stride
        self.spatial_conv = torch.nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=max_stride,
            stride=1,
            padding=0,
        )

        # Freeze spectral mask if requested
        if freeze_spectral_mask:
            for param in self.spectral_patchify.spectral_mask.parameters():
                param.requires_grad = False

    def get_stride(self) -> torch.Tensor:
        """Get the current stride from the spectral path.

        Returns:
            torch.Tensor: Stride per dimension, shape [data_dim].
        """
        return self.spectral_patchify.get_stride()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with dual-path processing.

        Args:
            x: Input tensor [B, C, H, W] (BHL format).

        Returns:
            Output tensor [B, out_features, H/stride, W/stride].
        """
        # Get current stride from spectral path
        current_stride = self.get_stride()

        # === SPECTRAL PATH ===
        x_spectral = self.spectral_patchify(x, is_bhl_input=True)

        # === SPATIAL PATH ===
        # Apply conv (stride=1, no padding)
        x_spatial = self.spatial_conv(x)

        # Subsample using spectral path's stride (per-dimension)
        stride_h = max(1, round(current_stride[0].item()))
        stride_w = max(1, round(current_stride[1].item()))
        x_spatial = x_spatial[:, :, ::stride_h, ::stride_w]

        # Resize spatial to match spectral if needed
        if x_spatial.shape[2:] != x_spectral.shape[2:]:
            x_spatial = F.interpolate(x_spatial, size=x_spectral.shape[2:], mode="bilinear")

        # Combine paths (sum)
        output = x_spectral + x_spatial

        return output

    def extra_repr(self) -> str:
        """Additional string when printing the module."""
        return f"in_features={self.in_features}, out_features={self.out_features}, max_stride={self.max_stride}"


class DualPathUnpatchify(torch.nn.Module):
    """Dual-path unpatchification combining spectral and spatial upsampling.

    This module combines:
    - **Spectral path**: Bilinear interpolation upsampling via SpectralUnpatchify
    - **Spatial path**: PixelShuffle-based learned upsampling

    The outputs are summed to reconstruct both low and high frequency content.

    Args:
        in_features: Number of input channels.
        out_features: Number of output channels.
        data_dim: Spatial dimensionality (currently only 2 is supported).
        max_stride: Maximum stride the architecture supports (used for PixelShuffle).
        output_proj_cfg: Optional LazyConfig for the spectral path's output projection.
        interpolation_mode: Interpolation mode for spectral upsampling ('bilinear' or 'nearest').

    Input shape:  [B, in_features, H, W] (BHL format)
    Output shape: [B, out_features, H*stride, W*stride]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        spectral_unpatchify_cfg: LazyConfig,  # LazyConfig["SpectralUnpatchify"]
        max_stride: int = 16,
        # output_proj_cfg: LazyConfig | None = None,
        interpolation_mode: str = "bilinear",
        **kwargs,  # For compatibility with ResidualNetwork and other callers
    ):
        """Initialize the DualPathUnpatchify module."""
        super().__init__()

        assert data_dim == 2, f"DualPathUnpatchify only supports data_dim=2, got {data_dim}"

        # kwargs are ignored - for compatibility with ResidualNetwork fallback (e.g., hidden_dim)

        self.data_dim = data_dim
        self.in_features = in_features
        self.out_features = out_features
        self.max_stride = max_stride
        self.interpolation_mode = interpolation_mode

        # Default output projection if not provided
        # if output_proj_cfg is None:
        #     output_proj_cfg = LazyConfig(torch.nn.Conv2d)(
        #         in_channels=in_features,
        #         out_channels=out_features,
        #         kernel_size=6,
        #         padding="same",
        #     )

        # === SPECTRAL PATH ===
        self.spectral_unpatchify = instantiate(spectral_unpatchify_cfg)
        # self.spectral_unpatchify = SpectralUnpatchify(
        #     in_features=in_features,
        #     out_features=out_features,
        #     data_dim=data_dim,
        #     output_proj_cfg=output_proj_cfg,
        #     interpolation_mode=interpolation_mode,
        # )

        # === SPATIAL PATH ===
        # PixelShuffle-based upsampling:
        # 1. Conv to expand channels: in_features -> out_features * max_stride^2
        # 2. PixelShuffle to rearrange channels to spatial dimensions
        self.spatial_expand = torch.nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features * max_stride * max_stride,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.spatial_shuffle = torch.nn.PixelShuffle(max_stride)

    def forward(
        self,
        x: torch.Tensor,
        target_shape: tuple[int, int],
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward pass with dual-path upsampling.

        Args:
            x: Input tensor [B, in_features, H, W] (BHL format).
            target_shape: Target spatial shape (H_out, W_out) for output.
            return_intermediates: If True, return empty dict (for API compatibility).

        Returns:
            Output tensor [B, out_features, H_out, W_out], or tuple with empty dict.
        """
        # === SPECTRAL PATH ===
        x_spectral = self.spectral_unpatchify(x, target_shape=target_shape, is_bhl_input=True)

        # === SPATIAL PATH ===
        # PixelShuffle upsampling
        x_spatial = self.spatial_expand(x)
        x_spatial = self.spatial_shuffle(x_spatial)

        # Resize to target shape if needed
        if x_spatial.shape[2:] != target_shape:
            x_spatial = F.interpolate(x_spatial, size=target_shape, mode="bilinear")

        # Combine paths (sum)
        output = x_spectral + x_spatial

        if return_intermediates:
            return output, {}
        return output

    def extra_repr(self) -> str:
        """Additional string when printing the module."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"max_stride={self.max_stride}, interpolation_mode={self.interpolation_mode}"
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    print(f"Using device: {device}\n")

    print("=" * 60)
    print("Test 1: Patchify and Unpatchify (Conv-based)")
    print("=" * 60)

    # Example: 2D image B x 64 x 64 x 3 (channels-last)
    B, H, W, hidden_dim = 2, 64, 64, 3
    embedding_dim = 32
    patch_size = 8
    x = torch.randn(B, H, W, hidden_dim)

    # Patchify layer (ViT-style: stride = patch_size)
    patchify_layer = Patchify(
        in_features=hidden_dim,
        out_features=embedding_dim,
        data_dim=2,
        patch_size=patch_size,
        # stride defaults to patch_size (non-overlapping)
    )

    # Unpatchify layer (inverse of patchify)
    unpatchify_layer = Unpatchify(
        in_features=embedding_dim,
        out_features=hidden_dim,
        data_dim=2,
        patch_size=patch_size,
        # stride defaults to patch_size
    )

    # Run layers
    patchify_layer.to(device)
    unpatchify_layer.to(device)

    y = patchify_layer(x)
    x_rec = unpatchify_layer(y)

    print(f"Input shape:      {tuple(x.shape)}")
    print(f"Patched shape:    {tuple(y.shape)}")
    print(f"Reconstructed:    {tuple(x_rec.shape)}")
    assert x_rec.shape == x.shape, "Reconstructed shape does not match input shape"
    print("✓ Conv-based patchify/unpatchify shape check passed.")

    print("\n" + "=" * 60)
    print("Test 2: SpectralUnpatchify with Linear output projection")
    print("=" * 60)

    # Small input (simulating patchified/downsampled data)
    B, H_small, W_small, C = 2, 16, 16, 32
    out_features = 64
    x_small = torch.randn(B, H_small, W_small, C)

    # SpectralUnpatchify with Linear output projection
    spectral_unpatchify_linear = SpectralUnpatchify(
        in_features=C,
        out_features=out_features,
        data_dim=2,
        output_proj_cfg=LazyConfig(torch.nn.Linear),
        interpolation_mode="bilinear",
    ).to(device)

    y_up = spectral_unpatchify_linear(x_small, target_shape=(64, 64), is_bhl_input=False)
    print(f"Input shape:      {tuple(x_small.shape)}")
    print("Target shape:     (64, 64)")
    print(f"Upsampled shape:  {tuple(y_up.shape)}")
    assert y_up.shape == (B, 64, 64, out_features), f"Expected (2, 64, 64, 64), got {y_up.shape}"
    print("✓ SpectralUnpatchify with Linear projection passed.")

    print("\n" + "=" * 60)
    print("Test 3: SpectralUnpatchify with different interpolation modes")
    print("=" * 60)

    for mode in ["bilinear", "bicubic", "nearest"]:
        spectral_unpatchify_mode = SpectralUnpatchify(
            in_features=C,
            out_features=C,
            data_dim=2,
            output_proj_cfg=LazyConfig(torch.nn.Linear),
            interpolation_mode=mode,
        ).to(device)

        y_mode = spectral_unpatchify_mode(x_small, target_shape=(64, 64))
        print(f"  {mode:10s}: {tuple(x_small.shape)} -> {tuple(y_mode.shape)}")
        assert y_mode.shape == (B, 64, 64, C)
    print("✓ All interpolation modes passed.")

    print("\n" + "=" * 60)
    print("Test 4: SpectralUnpatchify with CKConv output projection")
    print("=" * 60)

    from nvsubquadratic.modules.ckconv_nd import CKConvND
    from nvsubquadratic.modules.kernels_nd import RandomFourierKernelND

    # Create CKConv config for output projection (non-depthwise)
    # Note: hidden_dim will be overridden by in_features * out_features
    ckconv_cfg = LazyConfig(CKConvND)(
        data_dim=2,
        kernel_cfg=LazyConfig(RandomFourierKernelND)(
            out_dim=C * C,  # Must match hidden_dim = in_features * out_features
            data_dim=2,
            mlp_hidden_dim=32,
            num_layers=3,
            embedding_dim=32,
            omega_0=1.0,
            L_cache=32,
            use_bias=True,
            nonlinear_cfg=LazyConfig(torch.nn.GELU)(),
        ),
        mask_cfg=LazyConfig(torch.nn.Identity)(),
        grid_type="single",
        fft_padding="zero",
        use_shortcut=False,
        is_depthwise=False,
    )

    spectral_unpatchify_ckconv = SpectralUnpatchify(
        in_features=C,
        out_features=C,
        data_dim=2,
        output_proj_cfg=ckconv_cfg,
        interpolation_mode="bilinear",
    ).to(device)

    y_refined = spectral_unpatchify_ckconv(x_small, target_shape=(64, 64))
    print(f"Input shape:      {tuple(x_small.shape)}")
    print("Target shape:     (64, 64)")
    print(f"Refined shape:    {tuple(y_refined.shape)}")
    assert y_refined.shape == (B, 64, 64, C), f"Expected (2, 64, 64, 32), got {y_refined.shape}"
    print("✓ SpectralUnpatchify with CKConv output projection passed.")

    print("\n" + "=" * 60)
    print("Test 5: SpectralPatchify (spectral masking only)")
    print("=" * 60)

    from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND

    # Test spectral patchify without conv (pure spectral)
    B, H, W, C = 2, 64, 64, 32
    stride = 2.0
    x_full = torch.randn(B, H, W, C)

    spectral_patch_no_conv = SpectralPatchify(
        in_features=C,
        out_features=C,  # Must match when no conv
        data_dim=2,
        spectral_mask_cfg=LazyConfig(SpectralGaussianMaskND)(data_dim=2, clip_value=0.1, init_stride_value=stride),
        conv_cfg=None,  # No convolution
    ).to(device)

    y_ds = spectral_patch_no_conv(x_full, is_bhl_input=False)
    expected_shape = (B, int(H / stride), int(W / stride), C)
    print(f"Input shape:      {tuple(x_full.shape)}")
    print(f"Stride:           {stride}")
    print(f"Output shape:     {tuple(y_ds.shape)} (expected {expected_shape})")
    assert y_ds.shape == expected_shape, f"Expected {expected_shape}, got {y_ds.shape}"
    print("✓ SpectralPatchify (no conv) passed.")

    print("\n" + "=" * 60)
    print("Test 6: SpectralPatchify with CKConv pre-processing")
    print("=" * 60)

    # Test spectral patchify with conv (non-depthwise)
    # For non-depthwise, hidden_dim = c_in * c_out and kernel out_dim must match hidden_dim
    out_features = C  # Keep same channels
    hidden_dim_nondepth = C * out_features  # = 32 * 32 = 1024
    spectral_patch_with_conv = SpectralPatchify(
        in_features=C,
        out_features=out_features,
        data_dim=2,
        spectral_mask_cfg=LazyConfig(SpectralGaussianMaskND)(data_dim=2, clip_value=0.1, init_stride_value=stride),
        conv_cfg=LazyConfig(CKConvND)(
            data_dim=2,
            hidden_dim=hidden_dim_nondepth,  # For non-depthwise: c_in * c_out
            kernel_cfg=LazyConfig(RandomFourierKernelND)(
                out_dim=hidden_dim_nondepth,  # Must match hidden_dim
                data_dim=2,
                mlp_hidden_dim=64,
                num_layers=3,
                embedding_dim=32,
                omega_0=1.0,
                L_cache=32,
                use_bias=True,
                nonlinear_cfg=LazyConfig(torch.nn.GELU)(),
            ),
            mask_cfg=LazyConfig(torch.nn.Identity)(),
            grid_type="single",
            fft_padding="zero",
            use_shortcut=False,
            is_depthwise=False,
        ),
    ).to(device)

    y_ds_conv = spectral_patch_with_conv(x_full, is_bhl_input=False)
    expected_shape = (B, int(H / stride), int(W / stride), C)  # Preserves channels
    print(f"Input shape:      {tuple(x_full.shape)}")
    print(f"Output shape:     {tuple(y_ds_conv.shape)} (expected {expected_shape})")
    assert y_ds_conv.shape == expected_shape, f"Expected {expected_shape}, got {y_ds_conv.shape}"
    print("✓ SpectralPatchify with CKConv passed.")

    print("\n" + "=" * 60)
    print("Test 7: SpectralPatchify step-by-step validation")
    print("=" * 60)

    # Test intermediate outputs
    y_ds_full, intermediates = spectral_patch_with_conv(x_full, is_bhl_input=False, return_intermediates=True)
    print(f"Output shape:     {tuple(y_ds_full.shape)}")
    print(f"Spectral mask:    {tuple(intermediates['spectral_mask'].shape)}")

    # Test conv-only mode
    y_conv_only = spectral_patch_with_conv.forward_conv_only(x_full, is_bhl_input=False)
    print(f"Conv-only output: {tuple(y_conv_only.shape)} (same spatial size as input)")
    assert y_conv_only.shape[:3] == x_full.shape[:3], "Conv-only should preserve spatial dims"

    # Test spectral-only mode
    y_spectral_only = spectral_patch_with_conv.forward_spectral_only(x_full, is_bhl_input=False)
    print(f"Spectral-only:    {tuple(y_spectral_only.shape)} (downsampled, same channels)")
    assert y_spectral_only.shape == (B, int(H / stride), int(W / stride), C), "Spectral-only shape mismatch"

    print("✓ Step-by-step validation passed.")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
