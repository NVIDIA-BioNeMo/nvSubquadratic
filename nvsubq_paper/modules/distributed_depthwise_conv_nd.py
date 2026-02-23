# TODO: Add license header here

"""Distributed depthwise convolution wrappers for Context Parallelism.

This module provides distributed depthwise-only convolution wrappers that handle:

1. **Context Parallel (CP) Communication**: Handles CP-aware communication patterns
   with direct channel slicing.

2. **Depthwise-only Design**: Distributed weight management where:
   - Weight shape: [num_groups, kernel_size...] (no extra dimension)
   - Direct repeat_interleave for weight expansion
   - in_channels == out_channels (depthwise property)

Key Features:
- Depthwise convolutions only (no standard grouped convolutions)
- Group-based parameter sharing to reduce memory usage
- Distributed CP slicing logic

Example Usage:
    # 1D depthwise convolution with 128 groups for weight sharing
    conv1d = DistributedDepthwiseConv1d(
        hidden_dim=384, kernel_size=3, num_groups=128, causal=True
    )

    # 2D depthwise convolution
    conv2d = DistributedDepthwiseConv2d(
        hidden_dim=384, kernel_size=3, num_groups=128
    )

    # 3D depthwise convolution
    conv3d = DistributedDepthwiseConv3d(
        hidden_dim=1024, kernel_size=3, num_groups=None  # None = no weight sharing
    )
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DistributedDepthwiseConv1d(nn.Module):
    """1D depthwise convolution for CP parallelism.

    This implements a depthwise convolution where:
    - in_channels == out_channels (depthwise property)
    - Weights are shared across groups via num_groups parameter
    - Context Parallel slicing is done directly on the channel dimension

    Attributes:
        hidden_dim (int): Number of input/output channels
        kernel_size (int): Size of the convolution kernel
        num_groups (int): Number of groups for weight sharing (reduces parameters)
        group_dim (int): Channels per group (hidden_dim // num_groups)
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
        """Initialize DistributedDepthwiseConv1d.

        Args:
            hidden_dim: Number of input and output channels
            kernel_size: Size of the convolving kernel
            causal: If True, applies causal padding
            num_groups: Number of groups for weight sharing (if None, uses hidden_dim - no sharing)
            bias: If True, adds a learnable bias
            dtype: Data type for parameters
            device: Device for parameters
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
        """Forward pass with optional context parallelism.

        Args:
            x: Input tensor of shape [batch, channels, length]
               - Without CP: channels = hidden_dim
               - With CP: channels = hidden_dim // cp_world_size
            cp_group: Context parallel process group (None for no CP)

        Returns:
            Output tensor of same shape as input
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
    """2D depthwise convolution for CP parallelism.

    This implements a depthwise convolution where:
    - in_channels == out_channels (depthwise property)
    - Weights are shared across groups via num_groups parameter
    - Context Parallel slicing is done directly on the channel dimension
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
        """Initialize DistributedDepthwiseConv2d.

        Args:
            hidden_dim: Number of input and output channels
            kernel_size: Size of the convolving kernel
            num_groups: Number of groups for weight sharing (if None, uses hidden_dim)
            bias: If True, adds a learnable bias
            dtype: Data type for parameters
            device: Device for parameters
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
        """Forward pass with optional context parallelism.

        Args:
            x: Input tensor of shape [batch, channels, height, width]
            cp_group: Context parallel process group (None for no CP)

        Returns:
            Output tensor of same shape as input
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
    """3D depthwise convolution for CP parallelism.

    This implements a depthwise convolution where:
    - in_channels == out_channels (depthwise property)
    - Weights are shared across groups via num_groups parameter
    - Context Parallel slicing is done directly on the channel dimension
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
        """Initialize DistributedDepthwiseConv3d.

        Args:
            hidden_dim: Number of input and output channels
            kernel_size: Size of the convolving kernel
            num_groups: Number of groups for weight sharing (if None, uses hidden_dim)
            bias: If True, adds a learnable bias
            dtype: Data type for parameters
            device: Device for parameters
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
        """Forward pass with optional context parallelism.

        Args:
            x: Input tensor of shape [batch, channels, depth, height, width]
            cp_group: Context parallel process group (None for no CP)

        Returns:
            Output tensor of same shape as input
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
