"""CKConv-based Patchify/Unpatchify with learnable stride.

This module implements patchification using continuous kernels (CKConv) with
learnable stride via blending between adjacent integer strides.

Key features:
- SIREN MLP generates kernels at any size (parameter efficient)
- Learnable stride via differentiable blending
- Shared kernel between patchify and unpatchify
- Achieves zero reconstruction loss

Usage:
    # Create shared kernel
    shared_kernel = SIRENKernelWrapper(...)

    # Create patchify/unpatchify with shared kernel
    patchify = CKConvPatchify(shared_kernel=shared_kernel, ...)
    unpatchify = CKConvUnpatchify(shared_kernel=shared_kernel, ...)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsubquadratic.modules.kernels_nd import SIRENKernelND


class SIRENKernelWrapper(nn.Module):
    """Wrapper around SIRENKernelND for easy kernel generation at any size.

    Handles the coordinate generation and kernel extraction automatically.

    Args:
        in_channels: Number of input channels for the convolution kernel
        out_channels: Number of output channels for the convolution kernel
        data_dim: Spatial dimensions (1 or 2)
        hidden_dim: Hidden dimension of the SIREN MLP
        num_layers: Number of layers in the SIREN MLP
        omega_0: Frequency scaling for first SIREN layer
        hidden_omega_0: Frequency scaling for hidden SIREN layers
        L_cache: Cache size for kernel generation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        data_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        omega_0: float = 30.0,
        hidden_omega_0: float = 1.0,
        L_cache: int = 16,
    ):
        """Initialize the SIRENKernelWrapper."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.data_dim = data_dim
        self.L_cache = L_cache

        # SIREN outputs in_channels * out_channels values per spatial location
        self.kernel = SIRENKernelND(
            out_dim=in_channels * out_channels,
            data_dim=data_dim,
            mlp_hidden_dim=hidden_dim,
            num_layers=num_layers,
            embedding_dim=hidden_dim,
            omega_0=omega_0,
            hidden_omega_0=hidden_omega_0,
            L_cache=L_cache,
            use_bias=True,
        )

    def _rescaling_factor(self) -> float:
        """Compute the rescaling factor for the kernel.

        For a convolution, the fan-in is in_channels * kernel_size^data_dim.
        To preserve variance, we scale by 1/sqrt(fan_in).

        Returns:
            Rescaling factor to multiply the kernel by
        """
        fan_in = self.in_channels * (self.L_cache**self.data_dim)
        return fan_in**-0.5

    def forward(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate kernel of specified shape.

        Args:
            shape: Target kernel shape (e.g., (4, 4) for 2D)
            device: Device to create kernel on

        Returns:
            Kernel tensor of shape [out_channels, in_channels, *shape]
        """
        # SIRENKernelND expects seq_lens as a tuple and returns (kernel, grid)
        # kernel shape is [1, *spatial_dims, out_dim] where out_dim = in_channels * out_channels
        target_size = shape[0]  # Assume square kernel

        # Request the kernel at the target size
        # SIRENKernelND generates a grid of size (2*L-1) in each dimension
        # So we need L such that 2*L-1 >= target_size => L >= (target_size+1)/2
        request_L = max(self.L_cache, (target_size + 1) // 2 + 1)

        # Generate kernel - pass as tuple for each dimension
        if self.data_dim == 2:
            seq_lens = (request_L, request_L)
        elif self.data_dim == 1:
            seq_lens = (request_L,)
        else:
            raise NotImplementedError(f"data_dim={self.data_dim} not supported")

        full_kernel, _ = self.kernel(seq_lens)  # [1, *spatial_dims, in_channels * out_channels]

        # Rescale the kernel by the rescaling factor
        full_kernel = full_kernel * self._rescaling_factor()

        # Reshape from [1, H, W, in*out] to [in*out, H, W]
        if self.data_dim == 2:
            full_kernel = full_kernel.squeeze(0).permute(2, 0, 1)  # [in*out, H, W]
        elif self.data_dim == 1:
            full_kernel = full_kernel.squeeze(0).permute(1, 0)  # [in*out, L]

        # Extract center portion of target_size
        full_size = full_kernel.shape[1]
        center = full_size // 2
        half = target_size // 2

        if self.data_dim == 2:
            start = center - half
            end = start + target_size
            kernel = full_kernel[:, start:end, start:end]
        elif self.data_dim == 1:
            start = center - half
            end = start + target_size
            kernel = full_kernel[:, start:end]

        return kernel.to(device)


class CKConvPatchify(nn.Module):
    """CKConv-based patchification with learnable stride.

    Uses a continuous kernel (SIREN MLP) to generate convolution weights
    at any stride size. Learnable stride via blending between adjacent
    integer strides.

    Args:
        in_features: Number of input channels
        out_features: Number of output channels (hidden dimension)
        data_dim: Spatial dimensions (1 or 2)
        init_stride: Initial stride value
        max_stride: Maximum stride value
        kernel_size: Kernel size (if None, uses stride size for non-overlapping)
        freeze_stride: If True, stride is fixed and not learnable
        kernel_hidden_dim: Hidden dimension for kernel MLP
        kernel_num_layers: Number of layers in kernel MLP
        shared_kernel: Optional shared kernel module (for weight sharing with unpatchify)

    Note:
        - If kernel_size == stride: Non-overlapping patches (perfect reconstruction possible)
        - If kernel_size > stride: Overlapping patches (richer context, no perfect reconstruction)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int = 2,
        init_stride: int = 4,
        max_stride: int = 16,
        kernel_size: int | None = None,
        freeze_stride: bool = False,
        kernel_hidden_dim: int = 64,
        kernel_num_layers: int = 3,
        shared_kernel: SIRENKernelWrapper | None = None,
    ):
        """Initialize the CKConvPatchify layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.data_dim = data_dim
        self.init_stride = init_stride
        self.max_stride = max_stride
        self.kernel_size = kernel_size  # None means use stride
        self.freeze_stride = freeze_stride

        # Stride parameter (learnable or fixed based on freeze_stride)
        self.log_stride = nn.Parameter(
            torch.tensor([math.log(init_stride)] * data_dim),
            requires_grad=not freeze_stride,
        )

        # Continuous kernel (SIREN MLP)
        # L_cache should be at least the max of kernel_size and max_stride
        l_cache = max(max_stride, kernel_size or max_stride)
        if shared_kernel is not None:
            self.shared_kernel = shared_kernel
        else:
            self.shared_kernel = SIRENKernelWrapper(
                in_channels=in_features,
                out_channels=out_features,
                data_dim=data_dim,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
                omega_0=30.0,
                hidden_omega_0=1.0,
                L_cache=l_cache,
            )

    def get_stride(self) -> torch.Tensor:
        """Get current stride value (continuous)."""
        return torch.exp(self.log_stride)

    def _patchify_at_stride(self, x: torch.Tensor, stride_int: int, device) -> torch.Tensor:
        """Run patchification at a specific integer stride."""
        # Use fixed kernel_size if provided, otherwise use stride (non-overlapping)
        ks = self.kernel_size if self.kernel_size is not None else stride_int
        kernel = self.shared_kernel((ks, ks), device)
        kernel = kernel.view(self.out_features, self.in_features, ks, ks)

        # Compute padding for 'same'-like output size
        # For kernel_size > stride (overlapping), we need padding
        padding = (ks - stride_int) // 2
        return F.conv2d(x, kernel, stride=stride_int, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with learnable stride via blending.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Patchified tensor [B, out_features, H/stride, W/stride]
        """
        device = x.device
        current_stride = self.get_stride()
        stride_val = current_stride.mean()

        # Get floor and ceil strides
        stride_lo = int(torch.floor(stride_val).item())
        stride_hi = int(torch.ceil(stride_val).item())
        stride_lo = max(1, min(stride_lo, self.max_stride))
        stride_hi = max(1, min(stride_hi, self.max_stride))

        # Compute blend factor (differentiable!)
        alpha = stride_val - stride_lo

        if stride_lo == stride_hi:
            # Exactly integer stride - no blending needed
            return self._patchify_at_stride(x, stride_lo, device)
        else:
            # Blend between two strides
            out_lo = self._patchify_at_stride(x, stride_lo, device)
            out_hi = self._patchify_at_stride(x, stride_hi, device)

            # Resize to match (different strides give different output sizes)
            target_size = out_lo.shape[2:]
            if out_hi.shape[2:] != target_size:
                out_hi = F.interpolate(out_hi, size=target_size, mode="bilinear", align_corners=False)

            return (1 - alpha) * out_lo + alpha * out_hi


class CKConvUnpatchify(nn.Module):
    """CKConv-based unpatchification with learnable stride.

    Uses a continuous kernel (SIREN MLP) to generate transposed convolution
    weights at any stride size. Should share the kernel with CKConvPatchify
    for optimal reconstruction.

    Args:
        in_features: Number of input channels (hidden dimension)
        out_features: Number of output channels
        data_dim: Spatial dimensions (1 or 2)
        init_stride: Initial stride value (should match patchify)
        max_stride: Maximum stride value
        kernel_size: Kernel size (if None, uses stride size for non-overlapping)
        freeze_stride: If True, stride is fixed and not learnable
        kernel_hidden_dim: Hidden dimension for kernel MLP
        kernel_num_layers: Number of layers in kernel MLP
        shared_kernel: Optional shared kernel module (for weight sharing with patchify)
        target_size: Target output size (H, W) - if None, computed from stride

    Note:
        - If kernel_size == stride: Non-overlapping patches (perfect reconstruction possible)
        - If kernel_size > stride: Overlapping patches (richer context, no perfect reconstruction)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int = 2,
        init_stride: int = 4,
        max_stride: int = 16,
        kernel_size: int | None = None,
        freeze_stride: bool = False,
        kernel_hidden_dim: int = 64,
        kernel_num_layers: int = 3,
        shared_kernel: SIRENKernelWrapper | None = None,
        target_size: int | tuple[int, ...] | None = None,
    ):
        """Initialize the CKConvUnpatchify layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.data_dim = data_dim
        self.init_stride = init_stride
        self.max_stride = max_stride
        self.kernel_size = kernel_size  # None means use stride
        self.freeze_stride = freeze_stride
        self.target_size = (
            target_size if isinstance(target_size, tuple) else (target_size, target_size) if target_size else None
        )

        # Stride parameter (learnable or fixed based on freeze_stride)
        self.log_stride = nn.Parameter(
            torch.tensor([math.log(init_stride)] * data_dim),
            requires_grad=not freeze_stride,
        )

        # Continuous kernel (SIREN MLP)
        # Note: For conv_transpose2d, kernel shape is [in_channels, out_channels, H, W]
        # So we swap in_features and out_features compared to patchify
        # L_cache should be at least the max of kernel_size and max_stride
        l_cache = max(max_stride, kernel_size or max_stride)
        if shared_kernel is not None:
            self.shared_kernel = shared_kernel
        else:
            # Create own kernel (not recommended - sharing is better)
            self.shared_kernel = SIRENKernelWrapper(
                in_channels=out_features,  # Swapped for transpose
                out_channels=in_features,  # Swapped for transpose
                data_dim=data_dim,
                hidden_dim=kernel_hidden_dim,
                num_layers=kernel_num_layers,
                omega_0=30.0,
                hidden_omega_0=1.0,
                L_cache=l_cache,
            )

    def get_stride(self) -> torch.Tensor:
        """Get current stride value (continuous)."""
        return torch.exp(self.log_stride)

    def _unpatchify_at_stride(self, x: torch.Tensor, stride_int: int, target_size: tuple, device) -> torch.Tensor:
        """Run unpatchification at a specific integer stride."""
        # Use fixed kernel_size if provided, otherwise use stride (non-overlapping)
        ks = self.kernel_size if self.kernel_size is not None else stride_int
        kernel = self.shared_kernel((ks, ks), device)
        # For conv_transpose2d, kernel shape is [in_channels, out_channels, H, W]
        kernel = kernel.view(self.in_features, self.out_features, ks, ks)

        # Compute padding for proper output size
        # For kernel_size > stride (overlapping), we need padding
        padding = (ks - stride_int) // 2
        output = F.conv_transpose2d(x, kernel, stride=stride_int, padding=padding)

        # Ensure output matches target size
        if output.shape[2:] != target_size:
            output = F.interpolate(output, size=target_size, mode="bilinear", align_corners=False)

        return output

    def forward(self, x: torch.Tensor, target_size: tuple[int, int] | None = None) -> torch.Tensor:
        """Forward pass with learnable stride via blending.

        Args:
            x: Input tensor [B, hidden, H, W]
            target_size: Target output size (H, W). If None, uses self.target_size or computes from stride.

        Returns:
            Unpatchified tensor [B, out_features, target_H, target_W]
        """
        device = x.device
        B, C, H, W = x.shape
        current_stride = self.get_stride()
        stride_val = current_stride.mean()

        # Determine target size
        if target_size is None:
            target_size = self.target_size
        if target_size is None:
            # Compute from stride
            stride_int = int(torch.round(stride_val).item())
            target_size = (H * stride_int, W * stride_int)

        # Get floor and ceil strides
        stride_lo = int(torch.floor(stride_val).item())
        stride_hi = int(torch.ceil(stride_val).item())
        stride_lo = max(1, min(stride_lo, self.max_stride))
        stride_hi = max(1, min(stride_hi, self.max_stride))

        # Compute blend factor (differentiable!)
        alpha = stride_val - stride_lo

        if stride_lo == stride_hi:
            # Exactly integer stride - no blending needed
            return self._unpatchify_at_stride(x, stride_lo, target_size, device)
        else:
            # Blend between two strides
            out_lo = self._unpatchify_at_stride(x, stride_lo, target_size, device)
            out_hi = self._unpatchify_at_stride(x, stride_hi, target_size, device)

            return (1 - alpha) * out_lo + alpha * out_hi


class CKConvPatchifyUnpatchify(nn.Module):
    """Combined CKConv Patchify + Unpatchify with shared kernel.

    This is a convenience class that creates both patchify and unpatchify
    with a shared kernel. The stride is synchronized between both.

    For use in ResidualNetwork, use CKConvPatchify and CKConvUnpatchify
    separately with a shared kernel passed to both.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        data_dim: int = 2,
        init_stride: int = 4,
        max_stride: int = 16,
        kernel_size: int | None = None,
        freeze_stride: bool = False,
        kernel_hidden_dim: int = 64,
        kernel_num_layers: int = 3,
        target_size: int | tuple[int, ...] | None = None,
    ):
        """Initialize the CKConvDualPath module."""
        super().__init__()
        out_channels = out_channels or in_channels

        # L_cache should be at least the max of kernel_size and max_stride
        l_cache = max(max_stride, kernel_size or max_stride)

        # Create shared kernel (for patchify: in_channels -> hidden_channels)
        self.shared_kernel = SIRENKernelWrapper(
            in_channels=in_channels,
            out_channels=hidden_channels,
            data_dim=data_dim,
            hidden_dim=kernel_hidden_dim,
            num_layers=kernel_num_layers,
            omega_0=30.0,
            hidden_omega_0=1.0,
            L_cache=l_cache,
        )

        # Create patchify
        self.patchify = CKConvPatchify(
            in_features=in_channels,
            out_features=hidden_channels,
            data_dim=data_dim,
            init_stride=init_stride,
            max_stride=max_stride,
            kernel_size=kernel_size,
            freeze_stride=freeze_stride,
            shared_kernel=self.shared_kernel,
        )

        # Create unpatchify (shares kernel)
        self.unpatchify = CKConvUnpatchify(
            in_features=hidden_channels,
            out_features=out_channels,
            data_dim=data_dim,
            init_stride=init_stride,
            max_stride=max_stride,
            kernel_size=kernel_size,
            freeze_stride=freeze_stride,
            shared_kernel=self.shared_kernel,
            target_size=target_size,
        )

        # Synchronize stride parameters
        # Delete unpatchify's stride and use patchify's
        del self.unpatchify.log_stride
        self.unpatchify.log_stride = self.patchify.log_stride

    def get_stride(self) -> torch.Tensor:
        """Get current stride value."""
        return self.patchify.get_stride()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: patchify then unpatchify."""
        target_size = x.shape[2:]
        latent = self.patchify(x)
        output = self.unpatchify(latent, target_size=target_size)
        return output
