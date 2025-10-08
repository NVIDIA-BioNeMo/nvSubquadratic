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

from typing import Literal

import torch
import torch.distributed as dist
from einops import rearrange
from torch.autograd.function import Function


__all__ = ["AllToAllSingleFunction"]


def _get_zigzag_indices(N: int, device: torch.device | None = None) -> torch.Tensor:
    """Generates the zigzag indices for rearrangement.

    Args:
        N (int): The total number of chunks.
        device (torch.device): The device on which to create tensors.

    Returns:
        torch.Tensor: The zigzag indices.
    """
    half_N = (N + 1) // 2
    idx1 = torch.arange(half_N, device=device)
    idx2 = torch.arange(N - 1, half_N - 1, -1, device=device)
    zigzag_idx = torch.empty(N, dtype=torch.long, device=device)
    zigzag_idx[0::2] = idx1
    zigzag_idx[1::2] = idx2
    return zigzag_idx


def _get_inverse_zigzag_indices(N: int, device: torch.device | None = None) -> torch.Tensor:
    """Generates the inverse zigzag indices for rearrangement.

    Args:
        N (int): The total number of chunks.
        device (torch.device): The device on which to create tensors.

    Returns:
        torch.Tensor: The inverse zigzag indices.
    """
    half_N = N // 2
    idx1 = torch.arange(half_N, device=device)
    idx2 = torch.arange(N - 1, half_N - 1, -1, device=device)
    zigzag_idx = torch.empty(N, dtype=torch.long, device=device)
    zigzag_idx[0::2] = idx1
    zigzag_idx[1::2] = idx2
    inverse_zigzag_idx = torch.argsort(zigzag_idx)
    return inverse_zigzag_idx


def all_to_all_single_fn(
    group: dist.ProcessGroup,
    type: Literal["split_to_full", "full_to_split"],
    input: torch.Tensor,
    with_zigzag_splitting: bool = True,
) -> torch.Tensor:
    """Autograd-aware all_to_all_single communication function for 1D, 2D, and 3D tensors.

    This function performs all-to-all communication with optional zigzag splitting for load balancing.
    Zigzag splitting is applied to the third dimension (shape[2]) which is treated as the "temporal"
    or "sequence" dimension across all tensor types.

    Communication Pattern:
        - split_to_full: Gathers the full sequence while splitting across channels
        - full_to_split: Splits across sequence length while gathering channels

    Zigzag Splitting:
        - Applied to shape[2] (seq_len/height/temporal_len) for 1D/2D/3D tensor types
        - Helps balance communication load across devices
        - Spatial relationships are preserved since the full spatial structure is reconstructed
          during the all-gather process while distributing the hidden dimension

    Args:
        group (dist.ProcessGroup): The process group for communication.
        type (str): Either 'split_to_full' or 'full_to_split' to specify the communication pattern.
        input (torch.Tensor): Input tensor to be communicated.
            For 1D: Shape should be (batch_size, hidden_size, seq_len).
            For 2D: Shape should be (batch_size, hidden_size, height, width).
            For 3D: Shape should be (batch_size, hidden_size, temporal_len, height, width).
        with_zigzag_splitting (bool, optional): Whether to apply zigzag splitting. Defaults to True.
            Applied to shape[2] (seq_len/height/temporal_len) for 1D/2D/3D tensor types.

    Returns:
        torch.Tensor: Output tensor after communication with same shape as input.
    """
    world_size = dist.get_world_size(group=group)
    input_ndim = input.ndim

    if input_ndim not in [3, 4, 5]:
        raise ValueError(
            f"Unsupported input tensor dimension: {input_ndim}. Expected 3D (1D), 4D (2D), or 5D (3D) tensors."
        )

    # Determine model dimensionality from input tensor dimension
    dimensionality = input_ndim - 2

    if type == "split_to_full":
        # Given a split sequence, it gathers the whole sequence, while splitting across the channels dimension.
        # Unpack shape: B, D, local_length for 1D or B, D, H, W for 2D or B, D, t, H, W for 3D
        B, D, local_length, *_ = input.shape
        L = local_length * world_size
        d = D // world_size

        # Define reshape patterns based on dimensionality
        if dimensionality == 1:
            input_pattern = "B (cp d) l -> cp B d l"
            output_pattern = "cp B d l -> B d (cp l)"
        elif dimensionality == 2:
            input_pattern = "B (cp d) H W -> cp B d H W"
            output_pattern = "cp B d H W -> B d (cp H) W"
        elif dimensionality == 3:
            input_pattern = "B (cp d) t H W -> cp B d t H W"
            output_pattern = "cp B d t H W -> B d (cp t) H W"

        # Reshape and permute input for communication
        input_reshaped = rearrange(input, input_pattern, cp=world_size).contiguous()

        # Perform all_to_all_single communication
        output_reshaped = torch.empty_like(input_reshaped)
        dist.all_to_all_single(output_reshaped, input_reshaped, group=group)

        # Permute and reshape output back to original form
        output = rearrange(output_reshaped, output_pattern, cp=world_size).contiguous()

        # Apply zigzag splitting to shape[2] (temporal/sequence dimension) for all cases
        if with_zigzag_splitting:
            num_chunks = 2 * world_size

            # Ensure L is divisible by num_chunks
            if L % num_chunks != 0:
                raise ValueError(f"Spatial dimension length {L} is not divisible by num_chunks {num_chunks}")

            unzigzagged_split_length = L // num_chunks  # Length of each small chunk
            device = output.device
            inverse_zigzag_idx = _get_inverse_zigzag_indices(num_chunks, device=device)

            # Vectorized rearrangement using inverse zigzag indices
            # Reshape to (B, d, num_chunks, unzigzagged_split_length, ...) and apply inverse zigzag
            # 1D: (B, d, L) -> (B, d, num_chunks, unzigzagged_split_length)
            # 2D: (B, d, L, W) -> (B, d, num_chunks, unzigzagged_split_length, W)
            # 3D: (B, d, L, H, W) -> (B, d, num_chunks, unzigzagged_split_length, H, W)

            # Get spatial dimensions (everything after L)
            spatial_dims = list(output.shape[3:]) if input_ndim > 3 else []

            # Reshape with explicit spatial dimensions
            reshape_dims_1 = [B, d, num_chunks, unzigzagged_split_length] + spatial_dims
            reshape_dims_2 = [B, d, L] + spatial_dims

            output = (
                output.reshape(reshape_dims_1).index_select(dim=2, index=inverse_zigzag_idx).reshape(reshape_dims_2)
            )

        return output

    elif type == "full_to_split":
        # Given a full sequence split across channels, splits across the sequence length while gathering the channels.
        # Unpack shape: B, d, L for 1D or B, d, H, W for 2D or B, d, T, H, W for 3D
        B, d, L, *_ = input.shape

        # Define reshape patterns based on dimensionality
        if dimensionality == 1:
            input_pattern = "b d (cp l) -> cp b d l"
            output_pattern = "cp b d l -> b (cp d) l"
        elif dimensionality == 2:
            input_pattern = "B d (cp H) W -> cp B d H W"
            output_pattern = "cp B d H W -> B (cp d) H W"
        elif dimensionality == 3:
            input_pattern = "B d (cp t) H W -> cp B d t H W"
            output_pattern = "cp B d t H W -> B (cp d) t H W"

        # Apply zigzag splitting to shape[2] (temporal/sequence dimension) for all cases
        if with_zigzag_splitting:
            num_chunks = 2 * world_size
            chunk_length = L // num_chunks  # Length of each small chunk
            device = input.device
            zigzag_idx = _get_zigzag_indices(num_chunks, device=device)

            # Ensure L is divisible by num_chunks
            if L % num_chunks != 0:
                raise ValueError(f"Spatial dimension length {L} is not divisible by num_chunks {num_chunks}")

            # Vectorized rearrangement using zigzag indices
            # Reshape to (B, d, num_chunks, chunk_length, ...) and apply zigzag
            # 1D: (B, d, L) -> (B, d, num_chunks, chunk_length)
            # 2D: (B, d, L, W) -> (B, d, num_chunks, chunk_length, W)
            # 3D: (B, d, L, H, W) -> (B, d, num_chunks, chunk_length, H, W)

            # Get spatial dimensions (everything after L)
            spatial_dims = list(input.shape[3:]) if input_ndim > 3 else []

            # Reshape with explicit spatial dimensions
            reshape_dims_1 = [B, d, num_chunks, chunk_length] + spatial_dims
            reshape_dims_2 = [B, d, L] + spatial_dims

            input = input.reshape(reshape_dims_1).index_select(dim=2, index=zigzag_idx).reshape(reshape_dims_2)

        # Reshape and permute inputs for communication
        input_reshaped = rearrange(input, input_pattern, cp=world_size).contiguous()

        # Perform all_to_all_single communication
        output_reshaped = torch.empty_like(input_reshaped)
        dist.all_to_all_single(output_reshaped, input_reshaped, group=group)

        # Permute and reshape outputs back to original form
        output = rearrange(output_reshaped, output_pattern, cp=world_size).contiguous()

        return output

    else:
        raise ValueError(f"Unknown type {type}")


class AllToAllSingleFunction(Function):
    """Custom autograd function for all_to_all_single communication with optional zigzag splitting.

    A custom autograd function for performing all_to_all_single communication with optional zigzag splitting.
    Supports both 1D and 3D tensors.

    Attributes:
    - ctx: A context object that stores information for the forward and backward passes.
    - group: The process group for communication.
    - type: The type of communication pattern ('split_to_full' or 'full_to_split').
    - with_zigzag_splitting: A boolean indicating whether to apply zigzag splitting (1D only).
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        type: Literal["split_to_full", "full_to_split"],
        with_zigzag_splitting: bool,
    ):
        """Forward pass for the AllToAllSingleFunction."""
        ctx.group = group
        ctx.type = type
        ctx.with_zigzag_splitting = with_zigzag_splitting

        # Detach input_tensor to prevent PyTorch from tracking operations inside the communication
        input_tensor = input_tensor.detach()

        # Perform the communication operation
        output = all_to_all_single_fn(
            group=ctx.group, type=ctx.type, input=input_tensor, with_zigzag_splitting=ctx.with_zigzag_splitting
        )

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass for the AllToAllSingleFunction."""
        # The backward pass will perform the reverse communication
        grad_input = all_to_all_single_fn(
            group=ctx.group,
            type="split_to_full" if ctx.type != "split_to_full" else "full_to_split",
            input=grad_output,
            with_zigzag_splitting=ctx.with_zigzag_splitting,
        )
        # Return the gradient w.r.t. the input_tensor and None for other arguments
        return grad_input, None, None, None
