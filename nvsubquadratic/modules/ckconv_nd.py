# David W. Romero, 2025-09-09

"""CKConv (long-convolution) implementation for ND signals."""

import math
from typing import Literal

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate
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
    ):
        """Initialize the CKConvND.

        Args:
            data_dim: Dimension of input data.
            hidden_dim: Hidden dimension.
            kernel_cfg: LazyConfig for the kernel.
            mask_cfg: LazyConfig for the mask.
            grid_type: Type of grid to use.
        """
        assert grid_type in ["double", "single"], f"Invalid grid type: {grid_type}. Must be 'double' or 'single'."

        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        # Construct kernel and mask
        self.kernel = instantiate(kernel_cfg)
        self.mask = instantiate(mask_cfg)

        # Construct shortcut projection
        self.shortcut = torch.nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
        bounds = math.sqrt(1.0 / hidden_dim)
        self.shortcut.data.uniform_(-bounds, bounds)

        # Define FFT operation
        if data_dim == 1:
            # self.fftconv_fn = fftconv1d
            self.fftconv_fn = fftconv1d_bhl_w_reshape
            self.fftconv_fn_bhl_input = fftconv1d_bhl
        elif data_dim == 2:
            # self.fftconv_fn = fftconv2d
            self.fftconv_fn = fftconv2d_bhl_w_reshape
            self.fftconv_fn_bhl_input = fftconv2d_bhl
        elif data_dim == 3:
            # self.fftconv_fn = fftconv3d
            self.fftconv_fn = fftconv3d_bhl_w_reshape
            self.fftconv_fn_bhl_input = fftconv3d_bhl
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {data_dim}")

        # Define the grid type
        self.grid_type = grid_type

    def forward(self, x: torch.Tensor, is_bhl_input: bool = False) -> torch.Tensor:
        """Forward pass of the CKConvND.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, * spatial_dims, hidden_dim) or (batch_size, hidden_dim, * spatial_dims)
            is_bhl_input (bool): Whether the input is in BHL format, i.e., (batch_size, hidden_dim, * spatial_dims).
                Default is False.

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

        if is_bhl_input:
            # Apply kernel
            conv_kernel = rearrange(
                conv_kernel, "b ... c -> b c ..."
            )  # Reshape kernel to [B, C, * spatial_dims] (Kernels are always in BLH format)
            x_dtype = x.dtype
            x = self.fftconv_fn_bhl_input(
                x.to(torch.float32),
                conv_kernel.to(torch.float32),
                self.shortcut.to(torch.float32),
            )
            return x.to(x_dtype)
        else:
            # Apply kernel
            x_dtype = x.dtype
            x = self.fftconv_fn(
                x.to(torch.float32),
                conv_kernel.to(torch.float32),
                self.shortcut.to(torch.float32),
            )
            return x.to(x_dtype)
