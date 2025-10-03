"""Distributed-aware convolution wrappers for TP/CP parallelism.

This module provides distributed-aware wrappers for 1D, 2D, and 3D convolutions that handle:

1. **Tensor Parallel (TP) Weight Sharding**: Weights are automatically sharded across TP ranks
   based on the configured number of groups and group dimensions.

2. **Context Parallel (CP) Communication**: Placeholder for CP-aware communication patterns
   that would handle overlapping regions and sequence splitting.

3. **Proper Weight Dimensionality**: Weight shapes are determined based on:
   - num_groups: Number of groups for weight sharing (reduces parameters)
   - use_depthwise_grouping: Whether to use depthwise convolution grouping
   - TP world size: Automatic sharding across tensor parallel ranks

Key Features:
- Drop-in replacements for torch.nn.Conv1d/Conv2d/Conv3d
- Automatic weight initialization with proper TP-aware seeding
- Group-based parameter sharing to reduce memory usage
- Support for both standard and depthwise grouped convolutions

Weight Dimensionality Logic (adapted from Hyena implementation):
- width_per_tp_group = hidden_size // tp_world_size
- num_groups_per_tp = num_groups // tp_world_size
- group_dim = width_per_tp_group // num_groups_per_tp
- Weight shape: [num_groups_per_tp, group_dim, *kernel_size] or [num_groups_per_tp, 1, *kernel_size]
- Runtime expansion: weight.repeat_interleave(group_dim, dim=0) to match input channels

Example Usage:
    # 1D convolution with 128 groups for weight sharing
    conv1d = DistributedConv1d(
        in_channels=384, out_channels=384, kernel_size=3,
        num_groups=128, use_depthwise_grouping=True
    )

    # 2D convolution with standard grouping
    conv2d = DistributedConv2d(
        in_channels=256, out_channels=512, kernel_size=3,
        num_groups=64, use_depthwise_grouping=False
    )
"""

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def initialize_affine_weight_gpu(weight: torch.Tensor, init_method, partition_dim: int, stride: int = 1) -> None:
    """Initialize affine weight for model parallel on GPU."""
    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride
    init_method(weight.data)  # modify the data in place


def get_groups_and_group_sizes(
    hidden_size: int, num_groups: int, world_size: int, expand_factor: float = 1.0
) -> Tuple[int, int, int]:
    """Get the groups and group sizes for the model.

    Args:
        hidden_size: The hidden size of the model
        num_groups: The number of groups for convolution
        world_size: The tensor parallel world size
        expand_factor: Factor to expand the number of groups

    Returns:
        Tuple of (width_per_tp_group, num_groups_per_tp, group_dim)
    """
    width_per_tp_group = divide(hidden_size, world_size)
    num_groups_per_tp = int(divide(num_groups, world_size) * expand_factor)
    group_dim = width_per_tp_group // num_groups_per_tp
    return width_per_tp_group, num_groups_per_tp, group_dim


def slice_weight_for_context_parallel(
    weight: torch.Tensor, cp_group: torch.distributed.ProcessGroup = None
) -> torch.Tensor:
    """Slice weight tensor for context parallel processing.

    Args:
        weight: Weight tensor to slice
        cp_group: Context parallel process group

    Returns:
        Sliced weight tensor for current CP rank
    """
    if cp_group is None:
        return weight

    cp_rank = cp_group.rank()
    cp_world_size = cp_group.size()

    weight_per_rank = weight.shape[0] // cp_world_size
    start_idx = cp_rank * weight_per_rank
    end_idx = start_idx + weight_per_rank
    return weight[start_idx:end_idx]


def slice_bias_for_context_parallel(
    bias: torch.Tensor, cp_group: torch.distributed.ProcessGroup = None
) -> torch.Tensor:
    """Slice bias tensor for context parallel processing.

    Args:
        bias: Bias tensor to slice
        cp_group: Context parallel process group
    Returns:
        Sliced bias tensor for current CP rank
    """
    if cp_group is None:
        return bias

    cp_rank = cp_group.rank()
    cp_world_size = cp_group.size()

    bias_per_rank = bias.shape[0] // cp_world_size
    start_idx = cp_rank * bias_per_rank
    end_idx = start_idx + bias_per_rank
    return bias[start_idx:end_idx]


def prepare_depthwise_weights_for_cp(
    weight: torch.Tensor,
    actual_in_channels: int,
    group_dim: int,
    cp_group: torch.distributed.ProcessGroup = None,
) -> Tuple[torch.Tensor, int]:
    """Prepare depthwise convolution weights for context parallel processing.

    Args:
        weight: Base weight tensor [num_groups_per_tp, 1, *kernel_size]
        actual_in_channels: Number of input channels after CP split
        group_dim: Group dimension for weight repetition
        cp_group: Context parallel process group

    Returns:
        Tuple of (prepared_weight, groups) for convolution
    """
    if cp_group is not None and cp_group.size() > 1:
        # Slice weights for this CP rank
        weight_slice = slice_weight_for_context_parallel(weight, cp_group)

        # Calculate effective repetition factor for this CP rank
        effective_group_dim = actual_in_channels // weight_slice.shape[0]
        prepared_weight = weight_slice.repeat_interleave(effective_group_dim, dim=0)
        groups = actual_in_channels
    else:
        # Standard case: expand weights to match all output channels
        prepared_weight = weight.repeat_interleave(group_dim, dim=0)
        groups = weight.shape[0] * group_dim  # Total output channels

    return prepared_weight, groups


def prepare_standard_weights_for_cp(
    weight: torch.Tensor,
    out_channels: int,
    group_dim: int,
    groups: int,
    kernel_size: Tuple[int, ...],
    cp_group: torch.distributed.ProcessGroup = None,
) -> Tuple[torch.Tensor, int]:
    """Prepare standard grouped convolution weights for context parallel processing.

    Args:
        weight: Base weight tensor [num_groups_per_tp, group_dim, *kernel_size]
        out_channels: Total output channels
        group_dim: Group dimension
        groups: Number of convolution groups
        kernel_size: Convolution kernel size
        cp_group: Context parallel process group

    Returns:
        Tuple of (prepared_weight, groups) for convolution
    """
    if cp_group is not None and cp_group.size() > 1:
        # Slice weights for this CP rank
        weight_slice = slice_weight_for_context_parallel(weight, cp_group)

        effective_out_channels = out_channels // cp_group.size()
        effective_group_dim = group_dim // cp_group.size()
        prepared_weight = weight_slice.view(effective_out_channels, effective_group_dim, *kernel_size)
        effective_groups = groups // cp_group.size()
    else:
        prepared_weight = weight.view(out_channels, group_dim, *kernel_size)
        effective_groups = groups

    return prepared_weight, effective_groups


class DistributedConv1d(nn.Module):
    """Distributed-aware 1D convolution wrapper for TP/CP parallelism.

    This class wraps torch.nn.Conv1d and handles:
    - Tensor Parallel (TP) weight sharding
    - Context Parallel (CP) communication
    - Proper weight dimensionality based on num_groups
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int], str] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        num_groups: Optional[int] = None,
        use_depthwise_grouping: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize DistributedConv1d.

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to both sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input channels to output channels
            bias: If True, adds a learnable bias to the output
            padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'
            num_groups: Number of groups for weight sharing (if None, uses groups)
            use_depthwise_grouping: Whether to use depthwise grouping
            dtype: Data type for parameters
            device: Device for parameters
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.padding = padding
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups = groups
        self.bias_enabled = bias
        self.padding_mode = padding_mode
        self.use_depthwise_grouping = use_depthwise_grouping

        # Determine device safely
        if device is None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = torch.device("cpu")
        self.device = device

        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

        # Get parallel sizes
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        self.model_parallel_rank = get_tensor_model_parallel_rank()

        # Handle num_groups for weight sharing
        if num_groups is None:
            self.num_groups = groups
        else:
            self.num_groups = num_groups

        # Calculate group dimensions based on parallel settings
        self.width_per_tp_group, self.num_groups_per_tp, self.group_dim = get_groups_and_group_sizes(
            self.out_channels, self.num_groups, self.model_parallel_size
        )

        # Determine weight shape based on grouping strategy
        if self.use_depthwise_grouping:
            # Depthwise: [num_groups_per_tp, 1, kernel_size]
            weight_shape = [self.num_groups_per_tp, 1] + list(self.kernel_size)
            self.conv_groups = self.width_per_tp_group
        else:
            # Standard grouped: [num_groups_per_tp, group_dim, kernel_size]
            weight_shape = [self.num_groups_per_tp, self.group_dim] + list(self.kernel_size)
            self.conv_groups = self.num_groups_per_tp

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(weight_shape, device=self.device, dtype=self.dtype))
        setattr(self.weight, "tensor_model_parallel", True)

        # Initialize using standard Conv1d initialization
        bounds = math.sqrt(1.0 / (self.in_channels * math.prod(self.kernel_size)))
        conv_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
        initialize_affine_weight_gpu(self.weight, conv_init_method, partition_dim=0)

        # Initialize bias if enabled
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.empty(self.width_per_tp_group, device=self.device, dtype=self.dtype))
            setattr(self.bias, "tensor_model_parallel", True)
            bounds = math.sqrt(1.0 / (self.in_channels * math.prod(self.kernel_size)))
            bias_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
            self.bias.data = bias_init_method(self.bias.data)
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = 1
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, cp_group: torch.distributed.ProcessGroup = None) -> torch.Tensor:
        """Forward pass with distributed awareness.

        Args:
            x: Input tensor of shape [batch_size, in_channels, length]
            cp_group: Context parallel process group

        Returns:
            Output tensor of shape [batch_size, out_channels, output_length]
        """
        assert x.ndim == 3, f"Input must be 3D [batch, channels, length], got {x.ndim}D"

        # Get context parallel world size to handle channel dimension adjustments
        cp_world_size = cp_group.size() if cp_group is not None else 1

        # Adjust channel dimensions for context parallel
        actual_in_channels = x.shape[1]
        if cp_group is not None and cp_world_size > 1:
            expected_in_channels = self.in_channels // cp_group.size()
        else:
            # When not using CP or CP world size is 1, expect full input channels
            expected_in_channels = self.in_channels

        # Verify input channels match expected dimensions
        if actual_in_channels != expected_in_channels:
            raise RuntimeError(
                f"Input has {actual_in_channels} channels, but expected {expected_in_channels} "
                f"(original: {self.in_channels}, CP world size: {cp_world_size})"
            )

        # For grouped convolution, PyTorch expects:
        # weight: [out_channels, in_channels // groups, kernel_size]
        # where groups divides both in_channels and out_channels

        # Prepare weights and groups for convolution
        if self.use_depthwise_grouping:
            weight, groups = prepare_depthwise_weights_for_cp(
                self.weight, actual_in_channels, self.group_dim, cp_group
            )
        else:
            weight, groups = prepare_standard_weights_for_cp(
                self.weight, self.out_channels, self.group_dim, self.groups, self.kernel_size, cp_group
            )

        # Handle bias
        bias = None
        if self.bias_enabled and self.bias is not None:
            if self.use_depthwise_grouping:
                # For depthwise, bias follows the same pattern as weights
                if cp_group is not None and cp_world_size > 1:
                    bias_slice = slice_bias_for_context_parallel(self.bias, cp_group)
                    effective_group_dim = actual_in_channels // bias_slice.shape[0]
                    bias = bias_slice.repeat_interleave(effective_group_dim, dim=0)
                else:
                    bias = self.bias.repeat_interleave(self.group_dim, dim=0)
            else:
                # For standard grouping, slice bias to match effective output channels
                bias = slice_bias_for_context_parallel(self.bias, cp_group)

        # Apply padding if needed
        if isinstance(self.padding, str):
            # Handle padding modes like 'same', 'valid'
            if self.padding == "same":
                pad_total = self.kernel_size[0] - 1
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                x = F.pad(x, (pad_left, pad_right), mode="constant", value=0)
                padding = 0
            else:
                padding = 0
        else:
            padding = self.padding

        # Perform convolution
        output = F.conv1d(x, weight, bias, self.stride, padding, self.dilation, groups)

        return output


class DistributedConv2d(nn.Module):
    """Distributed-aware 2D convolution wrapper for TP/CP parallelism.

    This class wraps torch.nn.Conv2d and handles:
    - Tensor Parallel (TP) weight sharding
    - Context Parallel (CP) communication
    - Proper weight dimensionality based on num_groups
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        num_groups: Optional[int] = None,
        use_depthwise_grouping: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize DistributedConv2d.

        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to all four sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input channels to output channels
            bias: If True, adds a learnable bias to the output
            padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'
            num_groups: Number of groups for weight sharing (if None, uses groups)
            use_depthwise_grouping: Whether to use depthwise grouping
            dtype: Data type for parameters
            device: Device for parameters
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.bias_enabled = bias
        self.padding_mode = padding_mode
        self.use_depthwise_grouping = use_depthwise_grouping

        # Determine device safely
        if device is None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = torch.device("cpu")
        self.device = device

        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

        # Get parallel sizes
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        self.model_parallel_rank = get_tensor_model_parallel_rank()

        # Handle num_groups for weight sharing
        if num_groups is None:
            self.num_groups = groups
        else:
            self.num_groups = num_groups

        # Calculate group dimensions based on parallel settings
        self.width_per_tp_group, self.num_groups_per_tp, self.group_dim = get_groups_and_group_sizes(
            self.out_channels, self.num_groups, self.model_parallel_size
        )

        # Determine weight shape based on grouping strategy
        if self.use_depthwise_grouping:
            # Depthwise: [num_groups_per_tp, 1, kernel_h, kernel_w]
            weight_shape = [self.num_groups_per_tp, 1] + list(self.kernel_size)
            self.conv_groups = self.width_per_tp_group
        else:
            # Standard grouped: [num_groups_per_tp, group_dim, kernel_h, kernel_w]
            weight_shape = [self.num_groups_per_tp, self.group_dim] + list(self.kernel_size)
            self.conv_groups = self.num_groups_per_tp

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(weight_shape, device=self.device, dtype=self.dtype))
        setattr(self.weight, "tensor_model_parallel", True)

        # Initialize using standard Conv2d initialization
        bounds = math.sqrt(1.0 / (self.in_channels * math.prod(self.kernel_size)))
        conv_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
        initialize_affine_weight_gpu(self.weight, conv_init_method, partition_dim=0)

        # Initialize bias if enabled
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.empty(self.width_per_tp_group, device=self.device, dtype=self.dtype))
            setattr(self.bias, "tensor_model_parallel", True)
            bounds = math.sqrt(1.0 / (self.in_channels * math.prod(self.kernel_size)))
            bias_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
            self.bias.data = bias_init_method(self.bias.data)
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = 1
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, cp_group: torch.distributed.ProcessGroup = None) -> torch.Tensor:
        """Forward pass with distributed awareness.

        Args:
            x: Input tensor of shape [batch_size, in_channels, height, width]
            cp_group: Context parallel process group

        Returns:
            Output tensor of shape [batch_size, out_channels, output_height, output_width]
        """
        assert x.ndim == 4, f"Input must be 4D [batch, channels, height, width], got {x.ndim}D"

        # Get context parallel group to handle channel dimension adjustments
        cp_world_size = cp_group.size() if cp_group is not None else 1

        # Adjust channel dimensions for context parallel
        actual_in_channels = x.shape[1]
        if cp_group is not None and cp_world_size > 1:
            expected_in_channels = self.in_channels // cp_world_size
        else:
            # When not using CP or CP world size is 1, expect full input channels
            expected_in_channels = self.in_channels

        # Verify input channels match expected dimensions
        if actual_in_channels != expected_in_channels:
            raise RuntimeError(
                f"Input has {actual_in_channels} channels, but expected {expected_in_channels} "
                f"(original: {self.in_channels}, CP world size: {cp_world_size})"
            )

        # For grouped convolution, PyTorch expects:
        # weight: [out_channels, in_channels // groups, kernel_h, kernel_w]
        # where groups divides both in_channels and out_channels

        # Prepare weights and groups for convolution
        if self.use_depthwise_grouping:
            weight, groups = prepare_depthwise_weights_for_cp(self.weight, x.shape[1], self.group_dim, cp_group)
        else:
            weight, groups = prepare_standard_weights_for_cp(
                self.weight, self.out_channels, self.group_dim, self.groups, self.kernel_size, cp_group
            )

        # Handle bias
        bias = None
        if self.bias_enabled and self.bias is not None:
            if self.use_depthwise_grouping:
                # For depthwise, bias follows the same pattern as weights
                if cp_group is not None and cp_world_size > 1:
                    bias_slice = slice_bias_for_context_parallel(self.bias, cp_group)
                    effective_group_dim = x.shape[1] // bias_slice.shape[0]
                    bias = bias_slice.repeat_interleave(effective_group_dim, dim=0)
                else:
                    bias = self.bias.repeat_interleave(self.group_dim, dim=0)
            else:
                # For standard grouping, slice bias to match effective output channels
                bias = slice_bias_for_context_parallel(self.bias, cp_group)

        # Apply padding if needed
        if isinstance(self.padding, str):
            # Handle padding modes like 'same', 'valid'
            if self.padding == "same":
                pad_h = self.kernel_size[0] - 1
                pad_w = self.kernel_size[1] - 1
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
                padding = 0
            else:
                padding = 0
        else:
            padding = self.padding

        # Perform convolution
        output = F.conv2d(x, weight, bias, self.stride, padding, self.dilation, groups)

        return output


class DistributedConv3d(nn.Module):
    """Distributed-aware 3D convolution wrapper for TP/CP parallelism.

    This class wraps torch.nn.Conv3d and handles:
    - Tensor Parallel (TP) weight sharding
    - Context Parallel (CP) communication
    - Proper weight dimensionality based on num_groups
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int], str] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        num_groups: Optional[int] = None,
        use_depthwise_grouping: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize DistributedConv3d.

        Args:
            in_channels: Number of channels in the input volume
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to all six sides of the input
            dilation: Spacing between kernel elements
            groups: Number of blocked connections from input channels to output channels
            bias: If True, adds a learnable bias to the output
            padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'
            num_groups: Number of groups for weight sharing (if None, uses groups)
            use_depthwise_grouping: Whether to use depthwise grouping
            dtype: Data type for parameters
            device: Device for parameters
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.bias_enabled = bias
        self.padding_mode = padding_mode
        self.use_depthwise_grouping = use_depthwise_grouping

        # Determine device safely
        if device is None:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = torch.device("cpu")
        self.device = device

        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

        # Get parallel sizes
        self.model_parallel_size = get_tensor_model_parallel_world_size()
        self.model_parallel_rank = get_tensor_model_parallel_rank()

        # Handle num_groups for weight sharing
        if num_groups is None:
            self.num_groups = groups
        else:
            self.num_groups = num_groups

        # Calculate group dimensions based on parallel settings
        self.width_per_tp_group, self.num_groups_per_tp, self.group_dim = get_groups_and_group_sizes(
            self.out_channels, self.num_groups, self.model_parallel_size
        )

        # Determine weight shape based on grouping strategy
        if self.use_depthwise_grouping:
            # Depthwise: [num_groups_per_tp, 1, kernel_d, kernel_h, kernel_w]
            weight_shape = [self.num_groups_per_tp, 1] + list(self.kernel_size)
            self.conv_groups = self.width_per_tp_group
        else:
            # Standard grouped: [num_groups_per_tp, group_dim, kernel_d, kernel_h, kernel_w]
            weight_shape = [self.num_groups_per_tp, self.group_dim] + list(self.kernel_size)
            self.conv_groups = self.num_groups_per_tp

        # Initialize weight parameter
        self.weight = nn.Parameter(torch.empty(weight_shape, device=self.device, dtype=self.dtype))
        setattr(self.weight, "tensor_model_parallel", True)

        # Initialize using standard Conv3d initialization
        bounds = math.sqrt(1.0 / (self.in_channels * math.prod(self.kernel_size)))
        conv_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
        initialize_affine_weight_gpu(self.weight, conv_init_method, partition_dim=0)

        # Initialize bias if enabled
        if self.bias_enabled:
            self.bias = nn.Parameter(torch.empty(self.width_per_tp_group, device=self.device, dtype=self.dtype))
            setattr(self.bias, "tensor_model_parallel", True)
            bounds = math.sqrt(1.0 / (self.in_channels * math.prod(self.kernel_size)))
            bias_init_method = partial(torch.nn.init.uniform_, a=-bounds, b=bounds)
            self.bias.data = bias_init_method(self.bias.data)
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = 1
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, cp_group: torch.distributed.ProcessGroup = None) -> torch.Tensor:
        """Forward pass with distributed awareness.

        Args:
            x: Input tensor of shape [batch_size, in_channels, depth, height, width]
            cp_group: Context parallel process group

        Returns:
            Output tensor of shape [batch_size, out_channels, output_depth, output_height, output_width]
        """
        assert x.ndim == 5, f"Input must be 5D [batch, channels, depth, height, width], got {x.ndim}D"

        # Get context parallel world size to handle channel dimension adjustments
        cp_world_size = cp_group.size() if cp_group is not None else 1

        # Adjust channel dimensions for context parallel
        actual_in_channels = x.shape[1]
        if cp_group is not None and cp_world_size > 1:
            expected_in_channels = self.in_channels // cp_world_size
        else:
            # When not using CP or CP world size is 1, expect full input channels
            expected_in_channels = self.in_channels

        # Verify input channels match expected dimensions
        if actual_in_channels != expected_in_channels:
            raise RuntimeError(
                f"Input has {actual_in_channels} channels, but expected {expected_in_channels} "
                f"(original: {self.in_channels}, CP world size: {cp_world_size})"
            )

        # For grouped convolution, PyTorch expects:
        # weight: [out_channels, in_channels // groups, kernel_d, kernel_h, kernel_w]
        # where groups divides both in_channels and out_channels

        # Prepare weights and groups for convolution
        if self.use_depthwise_grouping:
            weight, groups = prepare_depthwise_weights_for_cp(self.weight, x.shape[1], self.group_dim, cp_group)
        else:
            weight, groups = prepare_standard_weights_for_cp(
                self.weight, self.out_channels, self.group_dim, self.groups, self.kernel_size, cp_group
            )

        # Handle bias
        bias = None
        if self.bias_enabled and self.bias is not None:
            if self.use_depthwise_grouping:
                # For depthwise, bias follows the same pattern as weights
                if cp_group is not None and cp_world_size > 1:
                    bias_slice = slice_bias_for_context_parallel(self.bias, cp_group)
                    effective_group_dim = x.shape[1] // bias_slice.shape[0]
                    bias = bias_slice.repeat_interleave(effective_group_dim, dim=0)
                else:
                    bias = self.bias.repeat_interleave(self.group_dim, dim=0)
            else:
                # For standard grouping, slice bias to match effective output channels
                bias = slice_bias_for_context_parallel(self.bias, cp_group)

        # Apply padding if needed
        if isinstance(self.padding, str):
            # Handle padding modes like 'same', 'valid'
            if self.padding == "same":
                pad_d = self.kernel_size[0] - 1
                pad_h = self.kernel_size[1] - 1
                pad_w = self.kernel_size[2] - 1
                pad_front = pad_d // 2
                pad_back = pad_d - pad_front
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back), mode="constant", value=0)
                padding = 0
            else:
                padding = 0
        else:
            padding = self.padding

        # Perform convolution
        output = F.conv3d(x, weight, bias, self.stride, padding, self.dilation, groups)

        return output
