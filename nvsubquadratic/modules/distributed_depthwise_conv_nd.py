# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Distributed depthwise convolution wrappers for Context Parallelism (CP).

This module provides 1-D, 2-D, and 3-D *depthwise* convolution layers that work
transparently under **Context Parallelism** (CP), where the channel dimension is
split across ``cp_world_size`` ranks.

**Context Parallelism and channel slicing**

In the Hyena / sequence-mixer CP setup each rank holds a contiguous slice of the
channel axis: rank ``r`` processes channels ``[r·C/P, (r+1)·C/P)`` where ``C`` is
the total hidden dimension and ``P`` is the CP world size.  The depthwise
convolution weight needs to match — if ``cp_group`` is provided, :meth:`forward`
expands the full weight tensor then slices the appropriate channel range before
calling ``F.convNd``.  No inter-rank communication is required.

**Group-based weight sharing**

To reduce the parameter count, weights are stored for ``num_groups`` prototypes
and expanded to ``hidden_dim`` channels via ``repeat_interleave`` along the
channel axis at each forward pass:

.. code-block:: text

    weight: [G, *kernel]            G = num_groups ≤ C
    expanded: [C, *kernel]          via repeat_interleave(C//G, dim=0)

Setting ``num_groups=None`` (or ``num_groups=hidden_dim``) gives each channel its
own independent filter (standard depthwise convolution).  Smaller ``num_groups``
reduces memory at the cost of expressivity.

**Initialisation**

Weights and biases are drawn from ``Uniform(-b, b)`` where
``b = 1 / sqrt(prod(kernel_size))``, matching the default ``nn.Conv*`` scheme.

Classes:
    DistributedDepthwiseConv1d: 1-D variant; supports ``causal=True`` for left-only
        padding (autoregressive / time-axis use).
    DistributedDepthwiseConv2d: 2-D variant; always uses ``padding="same"``.
    DistributedDepthwiseConv3d: 3-D variant; always uses ``padding="same"``.

Example::

    # 1-D causal depthwise conv with 128 weight groups on 384 channels
    conv1d = DistributedDepthwiseConv1d(hidden_dim=384, kernel_size=3, num_groups=128, causal=True)

    # 2-D depthwise conv
    conv2d = DistributedDepthwiseConv2d(hidden_dim=384, kernel_size=3, num_groups=128)

    # 3-D depthwise conv, no weight sharing
    conv3d = DistributedDepthwiseConv3d(hidden_dim=1024, kernel_size=3, num_groups=None)
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DistributedDepthwiseConv1d(nn.Module):
    """1-D depthwise convolution with CP-aware channel slicing and weight sharing.

    Stores a compact weight of shape ``[G, K]`` (``G`` groups, kernel size ``K``)
    and expands it to ``[C, K]`` at each forward pass via ``repeat_interleave``.
    When ``cp_group`` is provided, an additional slice ``[r·(C/P) : (r+1)·(C/P)]``
    is taken before calling ``F.conv1d``, so each CP rank only processes its
    local channel shard.

    Supports two padding modes:

    - ``causal=True``: left-only pad of ``(K-1)`` before the conv; output length
      equals input length with no future dependency.
    - ``causal=False`` (default): ``padding="same"`` (symmetric); suitable for
      spatial axes in multi-dimensional Hyena.

    Attributes:
        hidden_dim (int): Total number of input/output channels ``C``.
        kernel_size (int): Convolution kernel size ``K``.
        causal (bool): Whether left-only (causal) padding is used.
        num_groups (int): Number of weight prototype groups ``G``.
        group_dim (int): Channels per group ``C // G``.
        weight (nn.Parameter): Filter weights of shape ``[G, K]``.
        bias (nn.Parameter | None): Optional bias of shape ``[G]``.

    Args:
        hidden_dim: Total number of input/output channels ``C``.
        kernel_size: Convolution kernel size ``K``.
        causal: Apply left-only padding.  Default ``False``.
        num_groups: Weight prototype groups ``G``.  ``None`` → ``G = C``
            (standard depthwise, no sharing).
        bias: Include a learnable bias.  Default ``False``.
        dtype: Parameter dtype.  Default ``torch.float32``.
        device: Parameter device.  Defaults to ``cuda:current`` if available.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        causal: bool = False,
        num_groups: Optional[int] = None,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialise DistributedDepthwiseConv1d.

        Args:
            hidden_dim: Total number of input/output channels ``C``.
            kernel_size: Convolution kernel size ``K``.
            causal: Apply left-only causal padding.  Default ``False``.
            num_groups: Weight prototype groups ``G ≤ C``.  ``None`` → ``G = C``.
            bias: Include a learnable bias.  Default ``False``.
            dtype: Parameter dtype.  Default ``torch.float32``.
            device: Parameter device.  Defaults to current CUDA device.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.causal = causal

        # Determine device safely
        if device is None:
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

        # Handle num_groups for weight sharing
        # If num_groups is None, each channel gets its own weight (standard depthwise)
        # If num_groups < hidden_dim, weights are shared across groups
        self.num_groups = num_groups if num_groups is not None else hidden_dim
        self.group_dim = self.hidden_dim // self.num_groups

        # Weight shape: [num_groups, kernel_size]
        weight_shape = [self.num_groups, kernel_size]
        self.weight = nn.Parameter(torch.empty(weight_shape, device=self.device, dtype=self.dtype))

        # Initialize bias if enabled
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_groups, device=self.device, dtype=self.dtype))
        else:
            self.bias = None

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights and bias using uniform distribution."""
        bounds = math.sqrt(1.0 / self.kernel_size)
        torch.nn.init.uniform_(self.weight, a=-bounds, b=bounds)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, a=-bounds, b=bounds)

    def forward(self, x: torch.Tensor, cp_group: Optional[torch.distributed.ProcessGroup] = None) -> torch.Tensor:
        """Apply 1-D depthwise convolution with optional CP channel slicing.

        The full weight ``[G, K]`` is expanded to ``[C, K]`` via
        ``repeat_interleave``.  If ``cp_group`` is given, only the slice
        for the current rank is retained before calling ``F.conv1d``.

        Args:
            x: Input tensor of shape ``[B, C_local, L]`` where

               * ``C_local = hidden_dim`` when not using CP, or
               * ``C_local = hidden_dim // cp_world_size`` on CP rank ``r``.

            cp_group: Context-parallel process group.  ``None`` → single-device
                mode (no slicing).

        Returns:
            torch.Tensor: Output of shape ``[B, C_local, L]``; same length as
            input for stride=1.

        Raises:
            AssertionError: If ``x.ndim != 3``.
            RuntimeError: If ``x.shape[1]`` does not match the expected local
                channel count after CP slicing.
        """
        assert x.ndim == 3, f"Input must be 3D [batch, channels, length], got {x.ndim}D"

        # Expand weights to match all channels via repeat_interleave
        # Shape: [num_groups, kernel_size] -> [hidden_dim, kernel_size]
        weight = self.weight.repeat_interleave(self.group_dim, dim=0)
        bias = self.bias.repeat_interleave(self.group_dim, dim=0) if self.bias is not None else None

        # Slice for CP after expansion
        if cp_group is not None and cp_group.size() > 1:
            cp_rank = cp_group.rank()
            cp_world_size = cp_group.size()

            # Each rank gets a slice of the expanded channels
            local_channels = self.hidden_dim // cp_world_size
            start_idx = cp_rank * local_channels
            end_idx = start_idx + local_channels

            weight = weight[start_idx:end_idx]
            bias = bias[start_idx:end_idx] if bias is not None else None

        # Verify dimensions match
        expected_channels = weight.shape[0]
        if x.shape[1] != expected_channels:
            raise RuntimeError(
                f"Input has {x.shape[1]} channels, but expected {expected_channels} "
                f"(hidden_dim={self.hidden_dim}, CP size={cp_group.size() if cp_group else 1})"
            )

        # Perform depthwise convolution
        # For depthwise: groups = number of output channels
        if self.causal:
            pad_size = self.kernel_size - 1
            x = F.pad(x, (pad_size, 0))

            output = F.conv1d(
                x,
                rearrange(weight, "c k -> c 1 k"),  # [channels, 1, kernel_size]
                bias=bias,
                stride=1,
                padding=0,
                groups=weight.shape[0],  # Depthwise: each channel has its own filter
            )
        else:
            output = F.conv1d(
                x,
                rearrange(weight, "c k -> c 1 k"),  # [channels, 1, kernel_size]
                bias=bias,
                stride=1,
                padding="same",
                groups=weight.shape[0],  # Depthwise: each channel has its own filter
            )

        return output


class DistributedDepthwiseConv2d(nn.Module):
    """2-D depthwise convolution with CP-aware channel slicing and weight sharing.

    Stores weights of shape ``[G, Kh, Kw]`` and expands to ``[C, Kh, Kw]`` at
    runtime via ``repeat_interleave``.  When ``cp_group`` is provided, the
    appropriate channel slice for the current rank is extracted before calling
    ``F.conv2d(padding="same")``.

    Expects input in **channel-first** layout: ``[B, C_local, H, W]``.

    Attributes:
        hidden_dim (int): Total number of channels ``C``.
        kernel_size (Tuple[int, int]): 2-D kernel dimensions ``(Kh, Kw)``.
        num_groups (int): Weight prototype groups ``G``.
        group_dim (int): Channels per group ``C // G``.
        weight (nn.Parameter): Shape ``[G, Kh, Kw]``.
        bias (nn.Parameter | None): Shape ``[G]`` or ``None``.

    Args:
        hidden_dim: Total number of channels ``C``.
        kernel_size: Kernel size; ``int`` → ``(K, K)``.
        num_groups: Weight prototype groups ``G``.  ``None`` → ``G = C``.
        bias: Include a learnable bias.  Default ``False``.
        dtype: Parameter dtype.  Default ``torch.float32``.
        device: Parameter device.  Defaults to current CUDA device.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int]],
        num_groups: Optional[int] = None,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialise DistributedDepthwiseConv2d.

        Args:
            hidden_dim: Total number of input/output channels ``C``.
            kernel_size: Kernel size; ``int`` is broadcast to ``(K, K)``.
            num_groups: Weight prototype groups ``G ≤ C``.  ``None`` → ``G = C``.
            bias: Include a learnable bias.  Default ``False``.
            dtype: Parameter dtype.  Default ``torch.float32``.
            device: Parameter device.  Defaults to current CUDA device.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        # Determine device safely
        if device is None:
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

        # Handle num_groups for weight sharing
        self.num_groups = num_groups if num_groups is not None else hidden_dim
        self.group_dim = self.hidden_dim // self.num_groups

        # Weight shape: [num_groups, kernel_h, kernel_w]
        weight_shape = [self.num_groups] + list(self.kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape, device=self.device, dtype=self.dtype))

        # Initialize bias if enabled
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_groups, device=self.device, dtype=self.dtype))
        else:
            self.bias = None

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights and bias using uniform distribution."""
        bounds = math.sqrt(1.0 / math.prod(self.kernel_size))
        torch.nn.init.uniform_(self.weight, a=-bounds, b=bounds)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, a=-bounds, b=bounds)

    def forward(self, x: torch.Tensor, cp_group: Optional[torch.distributed.ProcessGroup] = None) -> torch.Tensor:
        """Apply 2-D depthwise convolution with optional CP channel slicing.

        Args:
            x: Input tensor of shape ``[B, C_local, H, W]``.
            cp_group: Context-parallel process group.  ``None`` → single-device.

        Returns:
            torch.Tensor: Output of shape ``[B, C_local, H, W]`` (same spatial
            size because ``padding="same"``).

        Raises:
            AssertionError: If ``x.ndim != 4``.
            RuntimeError: If ``x.shape[1]`` does not match the expected local
                channel count after CP slicing.
        """
        assert x.ndim == 4, f"Input must be 4D [batch, channels, height, width], got {x.ndim}D"

        # Expand weights to match all channels via repeat_interleave
        weight = self.weight.repeat_interleave(self.group_dim, dim=0)
        bias = self.bias.repeat_interleave(self.group_dim, dim=0) if self.bias is not None else None

        # Slice for CP after expansion
        if cp_group is not None and cp_group.size() > 1:
            cp_rank = cp_group.rank()
            cp_world_size = cp_group.size()

            local_channels = self.hidden_dim // cp_world_size
            start_idx = cp_rank * local_channels
            end_idx = start_idx + local_channels

            weight = weight[start_idx:end_idx]
            bias = bias[start_idx:end_idx] if bias is not None else None

        # Verify dimensions
        if x.shape[1] != weight.shape[0]:
            raise RuntimeError(
                f"Input has {x.shape[1]} channels, but expected {weight.shape[0]} "
                f"(hidden_dim={self.hidden_dim}, CP size={cp_group.size() if cp_group else 1})"
            )

        # Perform depthwise convolution
        output = F.conv2d(
            x,
            rearrange(weight, "c h w -> c 1 h w"),  # [channels, 1, kernel_h, kernel_w]
            bias=bias,
            stride=1,
            padding="same",
            groups=weight.shape[0],
        )

        return output


class DistributedDepthwiseConv3d(nn.Module):
    """3-D depthwise convolution with CP-aware channel slicing and weight sharing.

    Stores weights of shape ``[G, Kd, Kh, Kw]`` and expands to ``[C, Kd, Kh, Kw]``
    at runtime via ``repeat_interleave``.  When ``cp_group`` is provided, only
    the current rank's channel slice is passed to ``F.conv3d(padding="same")``.

    Expects input in **channel-first** layout: ``[B, C_local, D, H, W]``.

    Attributes:
        hidden_dim (int): Total number of channels ``C``.
        kernel_size (Tuple[int, int, int]): 3-D kernel dimensions ``(Kd, Kh, Kw)``.
        num_groups (int): Weight prototype groups ``G``.
        group_dim (int): Channels per group ``C // G``.
        weight (nn.Parameter): Shape ``[G, Kd, Kh, Kw]``.
        bias (nn.Parameter | None): Shape ``[G]`` or ``None``.

    Args:
        hidden_dim: Total number of channels ``C``.
        kernel_size: Kernel size; ``int`` → ``(K, K, K)``.
        num_groups: Weight prototype groups ``G``.  ``None`` → ``G = C``.
        bias: Include a learnable bias.  Default ``False``.
        dtype: Parameter dtype.  Default ``torch.float32``.
        device: Parameter device.  Defaults to current CUDA device.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        num_groups: Optional[int] = None,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialise DistributedDepthwiseConv3d.

        Args:
            hidden_dim: Total number of input/output channels ``C``.
            kernel_size: Kernel size; ``int`` is broadcast to ``(K, K, K)``.
            num_groups: Weight prototype groups ``G ≤ C``.  ``None`` → ``G = C``.
            bias: Include a learnable bias.  Default ``False``.
            dtype: Parameter dtype.  Default ``torch.float32``.
            device: Parameter device.  Defaults to current CUDA device.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

        # Determine device safely
        if device is None:
            device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

        # Handle num_groups for weight sharing
        self.num_groups = num_groups if num_groups is not None else hidden_dim
        self.group_dim = self.hidden_dim // self.num_groups

        # Weight shape: [num_groups, kernel_d, kernel_h, kernel_w]
        weight_shape = [self.num_groups] + list(self.kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape, device=self.device, dtype=self.dtype))

        # Initialize bias if enabled
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_groups, device=self.device, dtype=self.dtype))
        else:
            self.bias = None

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights and bias using uniform distribution."""
        bounds = math.sqrt(1.0 / math.prod(self.kernel_size))
        torch.nn.init.uniform_(self.weight, a=-bounds, b=bounds)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, a=-bounds, b=bounds)

    def forward(self, x: torch.Tensor, cp_group: Optional[torch.distributed.ProcessGroup] = None) -> torch.Tensor:
        """Apply 3-D depthwise convolution with optional CP channel slicing.

        Args:
            x: Input tensor of shape ``[B, C_local, D, H, W]``.
            cp_group: Context-parallel process group.  ``None`` → single-device.

        Returns:
            torch.Tensor: Output of shape ``[B, C_local, D, H, W]`` (same
            spatial size because ``padding="same"``).

        Raises:
            AssertionError: If ``x.ndim != 5``.
            RuntimeError: If ``x.shape[1]`` does not match the expected local
                channel count after CP slicing.
        """
        assert x.ndim == 5, f"Input must be 5D [batch, channels, depth, height, width], got {x.ndim}D"

        # Expand weights to match all channels via repeat_interleave
        weight = self.weight.repeat_interleave(self.group_dim, dim=0)
        bias = self.bias.repeat_interleave(self.group_dim, dim=0) if self.bias is not None else None

        # Slice for CP after expansion
        if cp_group is not None and cp_group.size() > 1:
            cp_rank = cp_group.rank()
            cp_world_size = cp_group.size()

            local_channels = self.hidden_dim // cp_world_size
            start_idx = cp_rank * local_channels
            end_idx = start_idx + local_channels

            weight = weight[start_idx:end_idx]
            bias = bias[start_idx:end_idx] if bias is not None else None

        # Verify dimensions
        if x.shape[1] != weight.shape[0]:
            raise RuntimeError(
                f"Input has {x.shape[1]} channels, but expected {weight.shape[0]} "
                f"(hidden_dim={self.hidden_dim}, CP size={cp_group.size() if cp_group else 1})"
            )

        # Perform depthwise convolution
        output = F.conv3d(
            x,
            rearrange(weight, "c d h w -> c 1 d h w"),  # [channels, 1, kernel_d, kernel_h, kernel_w]
            bias=bias,
            stride=1,
            padding="same",
            groups=weight.shape[0],
        )

        return output
