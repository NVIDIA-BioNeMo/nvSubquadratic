# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
            data_dim: Dimension of input data (1D for sequences, 2D for images, 3D for videos, etc.).
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
