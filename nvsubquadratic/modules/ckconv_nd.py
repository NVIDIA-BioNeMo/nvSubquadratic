# TODO: Add license header here


"""CKConv (long-convolution) implementation for ND signals."""

import math
from typing import Literal

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_bhl,
    circular_fftconv1d_bhl_w_reshape,
    circular_fftconv2d_bhl,
    circular_fftconv2d_bhl_w_reshape,
    circular_fftconv3d_bhl,
    circular_fftconv3d_bhl_w_reshape,
)
from nvsubquadratic.ops.fftconv import (
    fftconv1d_bhl,
    fftconv1d_bhl_w_reshape,
    fftconv2d_bhl,
    fftconv2d_bhl_w_reshape,
    fftconv3d_bhl,
    fftconv3d_bhl_w_reshape,
)


class CKConvND(torch.nn.Module):
    """CKConv (long-convolution) implementation for ND signals."""

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        kernel_cfg: LazyConfig,
        mask_cfg: LazyConfig,
        grid_type: Literal["double", "single"],
        fft_padding: Literal["zero", "circular"],
    ):
        """Initialize the CKConvND.

        Args:
            data_dim: Dimension of input data (1D for sequences, 2D for images, 3D for videos, etc.).
            hidden_dim: Hidden dimension.
            kernel_cfg: LazyConfig for the kernel.
            mask_cfg: LazyConfig for the mask.
            grid_type: Type of grid to use.
            fft_padding: Boundary behavior of the FFT convolution. 'zero' uses zero-padding with
                cropping (conventional FFT-based conv). 'circular' uses periodic
                (wrap-around) convolution implemented via frequency-domain phase ramps.
        """
        assert grid_type in ["double", "single"], f"Invalid grid type: {grid_type}. Must be 'double' or 'single'."
        assert fft_padding in ["zero", "circular"], (
            f"Invalid FFT padding: {fft_padding}. Must be 'zero' or 'circular'."
        )
        if fft_padding == "circular":
            # Circular (periodic) convolution only makes sense with kernel size == input size,
            # which corresponds to 'single' grid type in this CKConv setup.
            assert grid_type == "single", (
                "fft_padding='circular' requires grid_type='single' (kernel size equals input size)."
            )

        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.fft_padding = fft_padding

        # Construct kernel and mask
        self.kernel = instantiate(kernel_cfg)
        self.mask = instantiate(mask_cfg)

        # Construct shortcut projection
        self.shortcut = torch.nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
        bounds = math.sqrt(1.0 / hidden_dim)
        self.shortcut.data.uniform_(-bounds, bounds)

        # Define FFT operation depending on padding and dimensionality
        if fft_padding == "circular":
            if data_dim == 1:
                self.fftconv_fn = circular_fftconv1d_bhl_w_reshape
                self.fftconv_fn_bhl_input = circular_fftconv1d_bhl
            elif data_dim == 2:
                self.fftconv_fn = circular_fftconv2d_bhl_w_reshape
                self.fftconv_fn_bhl_input = circular_fftconv2d_bhl
            elif data_dim == 3:
                self.fftconv_fn = circular_fftconv3d_bhl_w_reshape
                self.fftconv_fn_bhl_input = circular_fftconv3d_bhl
            else:
                raise ValueError(f"Unsupported number of spatial dimensions: {data_dim}")
        else:  # "zero"
            if data_dim == 1:
                self.fftconv_fn = fftconv1d_bhl_w_reshape
                self.fftconv_fn_bhl_input = fftconv1d_bhl
            elif data_dim == 2:
                self.fftconv_fn = fftconv2d_bhl_w_reshape
                self.fftconv_fn_bhl_input = fftconv2d_bhl
            elif data_dim == 3:
                self.fftconv_fn = fftconv3d_bhl_w_reshape
                self.fftconv_fn_bhl_input = fftconv3d_bhl
            else:
                raise ValueError(f"Unsupported number of spatial dimensions: {data_dim}")

        # Define the grid type
        self.grid_type = grid_type

    @torch.compiler.disable()
    def apply_convolution(
        self, x: torch.Tensor, shortcut: torch.Tensor, conv_kernel: torch.Tensor, is_bhl_input: bool = False
    ):
        """Apply convolution using the provided kernel and shortcut.

        Uses a separate function to allow disabling torch.compile for this part.

        Args:
            x (torch.Tensor): Input tensor.
            shortcut (torch.Tensor): Shortcut parameter.
            conv_kernel (torch.Tensor): Convolution kernel.
            is_bhl_input (bool): Whether the input is in BHL format.

        Returns:
            torch.Tensor: Output tensor after applying convolution.
        """
        if is_bhl_input:
            # Apply kernel
            conv_kernel = rearrange(
                conv_kernel, "b ... c -> b c ..."
            )  # Reshape kernel to [B, C, * spatial_dims] (Kernels are always in BLH format)
            x_dtype = x.dtype
            x = self.fftconv_fn_bhl_input(
                x.to(torch.float32),
                conv_kernel.to(torch.float32),
                shortcut.to(torch.float32),
            )
            return x.to(x_dtype)
        else:
            # Apply kernel
            x_dtype = x.dtype
            x = self.fftconv_fn(
                x.to(torch.float32),
                conv_kernel.to(torch.float32),
                shortcut.to(torch.float32),
            )
            return x.to(x_dtype)

    def forward(
        self, x: torch.Tensor, is_bhl_input: bool = False, cp_group: torch.distributed.ProcessGroup = None
    ) -> torch.Tensor:
        """Forward pass of the CKConvND.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, * spatial_dims, hidden_dim) or (batch_size, hidden_dim, * spatial_dims)
            is_bhl_input (bool): Whether the input is in BHL format, i.e., (batch_size, hidden_dim, * spatial_dims).
                Default is False.
            cp_group (torch.distributed.ProcessGroup): Context parallel process group.
                Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, * spatial_dims, hidden_dim) or (batch_size, hidden_dim, * spatial_dims)
        """
        # Get the spatial dimensions from the input tensor
        if is_bhl_input:
            spatial_dims = x.shape[2:]  # [* spatial_dims]
        else:
            spatial_dims = x.shape[1:-1]  # [* spatial_dims]

        if self.grid_type == "single":
            # Take half the spatial dimensions for the grid cache. Since the grid cache is
            # double the size of the input, when we take half the spatial dimensions, we get
            # a convolutional kernel with size equal to the input.
            grid_lens = [(seq_len + 1) // 2 for seq_len in spatial_dims]
        else:  # "double"
            # Take the full spatial dimensions for the grid cache. Since the grid cache is
            # double the size of the input, when we take the full spatial dimensions, we get
            # a convolutional kernel with size equal to twice the input.
            grid_lens = spatial_dims

        # Compute kernel
        conv_kernel, grid = self.kernel(grid_lens)

        # Apply mask to kernel
        if not isinstance(self.mask, torch.nn.Identity):
            conv_kernel = self.mask(grid=grid, x=conv_kernel)

        # Handle context parallelism by slicing the kernel to match input channel dimensions
        if cp_group is not None and cp_group.size() > 1:
            cp_world_size = cp_group.size()
            cp_rank = cp_group.rank()

            # Get the channel dimension (last dimension in BLH format)
            kernel_channels = conv_kernel.shape[-1]
            channels_per_rank = kernel_channels // cp_world_size

            # Slice the kernel along the channel dimension for this CP rank
            start_idx = cp_rank * channels_per_rank
            end_idx = start_idx + channels_per_rank
            conv_kernel = conv_kernel[..., start_idx:end_idx]

            # Also slice the shortcut parameter
            shortcut = self.shortcut[start_idx:end_idx]
        else:
            shortcut = self.shortcut

        # Apply convolution
        out = self.apply_convolution(x, shortcut, conv_kernel, is_bhl_input=is_bhl_input)

        return out
