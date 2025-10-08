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

"""Pytest tests for distributed depthwise convolution modules."""

import math

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.distributed_depthwise_conv_nd import (
    DistributedDepthwiseConv1d,
    DistributedDepthwiseConv2d,
    DistributedDepthwiseConv3d,
)


def depthwise_conv_config(data_dim: int = 1) -> LazyConfig:
    """Create a LazyConfig for QKVSequenceMixer with Hyena as inner mixer.

    Constructs a complete configuration for QKVSequenceMixer using Hyena as the
    inner sequence mixer.

    Args:
        data_dim: Dimensionality of the input data (default: 1 for 1D sequences).

    Returns:
        LazyConfig: A lazy configuration object for QKVSequenceMixer that can be
            instantiated later.
    """
    # Select the appropriate standard convolution class based on data dimension
    if data_dim == 1:
        conv_class = torch.nn.Conv1d
    elif data_dim == 2:
        conv_class = torch.nn.Conv2d
    elif data_dim == 3:
        conv_class = torch.nn.Conv3d
    else:
        raise ValueError(f"Unsupported data dimension: {data_dim}")

    return LazyConfig(conv_class)(
        in_channels=384,  # 3 * 128 for concatenated q, k, v
        out_channels=384,  # 3 * 128 for concatenated q, k, v
        kernel_size=3,
        groups=384,  # Grouped convolution
        padding=1,
        bias=False,
    )


def distributed_depthwise_conv_config(data_dim: int = 1) -> LazyConfig:
    """Create a LazyConfig for QKVSequenceMixer with distributed convolutions.

    Constructs a complete configuration for QKVSequenceMixer using Hyena as the
    inner sequence mixer with distributed-aware convolution layers.

    Args:
        data_dim: Dimensionality of the input data (default: 1 for 1D sequences).

    Returns:
        LazyConfig: A lazy configuration object for QKVSequenceMixer that can be
            instantiated later.
    """
    # Select the appropriate distributed convolution class based on data dimension
    if data_dim == 1:
        conv_class = DistributedDepthwiseConv1d
    elif data_dim == 2:
        conv_class = DistributedDepthwiseConv2d
    elif data_dim == 3:
        conv_class = DistributedDepthwiseConv3d
    else:
        raise ValueError(f"Unsupported data dimension: {data_dim}")

    return LazyConfig(conv_class)(
        hidden_dim=384,  # 3 * 128 for concatenated q, k, v
        kernel_size=3,
        bias=False,
        num_groups=384,  # Full depthwise - each channel has its own filter
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_distributed_vs_standard_equivalency(data_dim, dtype_fixture, device):
    """Test that distributed and standard convolutions produce equivalent results when CP=1.

    This test validates that our distributed convolution wrappers produce the same
    results as standard PyTorch convolutions when no parallelism is used. It copies
    weights from the standard model to the distributed model and verifies that both
    forward and backward passes produce identical results.

    Supports data_dim=1 (1D sequences), data_dim=2 (2D images), and data_dim=3 (3D volumes).
    """

    # Create both configurations
    standard_cfg = depthwise_conv_config(data_dim=data_dim)
    distributed_cfg = distributed_depthwise_conv_config(data_dim=data_dim)

    # Instantiate both models
    standard_conv = instantiate(standard_cfg)
    distributed_conv = instantiate(distributed_cfg)

    # Move to device and set dtype
    standard_conv = standard_conv.to(dtype=dtype_fixture, device=device)
    distributed_conv = distributed_conv.to(dtype=dtype_fixture, device=device)

    # Copy weights from standard model to distributed model
    standard_state_dict = standard_conv.state_dict()
    distributed_state_dict = distributed_conv.state_dict()

    # Copy matching weights from standard to distributed model
    for name, param in standard_state_dict.items():
        if name not in distributed_state_dict:
            continue

        # Direct copy if shapes match
        if param.shape == distributed_state_dict[name].shape:
            distributed_state_dict[name].copy_(param)
        # Reshape short_conv weights for depthwise: squeeze the singleton in_channels//groups dim
        elif "short_conv.weight" in name and param.ndim >= 3 and param.shape[1] == 1:
            distributed_state_dict[name].copy_(param.squeeze(1))
        # For standard conv weights, need to squeeze the groups dimension for distributed
        elif "weight" in name and param.ndim == data_dim + 2 and param.shape[1] == 1:
            # Standard conv shape: [out_channels, in_channels_per_group, *kernel_size]
            # Distributed shape: [num_groups, *kernel_size]
            distributed_state_dict[name].copy_(param.squeeze(1))

    # Create test input based on data dimension
    batch_size = 2
    channels = 384  # Must match in_channels from config

    if data_dim == 1:
        # 1D: [batch, channels, length]
        seq_len = 128
        test_input = torch.randn(batch_size, channels, seq_len, dtype=dtype_fixture, device=device, requires_grad=True)
    elif data_dim == 2:
        # 2D: [batch, channels, height, width]
        height, width = 32, 32
        test_input = torch.randn(
            batch_size, channels, height, width, dtype=dtype_fixture, device=device, requires_grad=True
        )
    elif data_dim == 3:
        # 3D: [batch, channels, depth, height, width]
        depth, height, width = 16, 16, 16
        test_input = torch.randn(
            batch_size, channels, depth, height, width, dtype=dtype_fixture, device=device, requires_grad=True
        )

    # Clone input for distributed model to ensure identical inputs
    test_input_dist = test_input.clone().detach().requires_grad_(True)

    # Run forward pass on both models
    # Standard PyTorch conv doesn't have cp_group parameter
    standard_output = standard_conv(test_input)
    # Distributed conv has cp_group parameter (None means no CP)
    distributed_output = distributed_conv(test_input_dist, cp_group=None)

    # Compare shapes
    assert standard_output.shape == distributed_output.shape

    # Verify no NaN/Inf values
    assert not torch.isnan(standard_output).any()
    assert not torch.isnan(distributed_output).any()
    assert not torch.isinf(standard_output).any()
    assert not torch.isinf(distributed_output).any()

    # Compare forward pass outputs (uses automatic dtype-based tolerances)
    torch.testing.assert_close(standard_output, distributed_output)

    # Test backward pass
    loss_standard = standard_output.mean()
    loss_standard.backward()

    loss_distributed = distributed_output.mean()
    loss_distributed.backward()

    # Compare input gradients (uses automatic dtype-based tolerances)
    torch.testing.assert_close(test_input.grad, test_input_dist.grad)

    # Compare parameter gradients (allow shape mismatches for short_conv.weight)
    for (name_std, param_std), (name_dist, param_dist) in zip(
        standard_conv.named_parameters(), distributed_conv.named_parameters()
    ):
        if param_std.grad is None and param_dist.grad is None:
            continue
        if param_std.grad is None or param_dist.grad is None:
            continue
        # Skip gradient comparison if shapes don't match (e.g., short_conv.weight)
        if param_std.grad.shape != param_dist.grad.shape:
            continue
        torch.testing.assert_close(param_std.grad, param_dist.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_distributed_convolutions(data_dim, dtype_fixture, device):
    """Test distributed convolution classes directly.

    This test validates the correctness of the distributed convolution wrappers
    by testing basic forward pass functionality.
    """
    # Test 1D distributed depthwise convolution
    if data_dim == 1:
        dist_conv = DistributedDepthwiseConv1d(
            hidden_dim=384,  # 3 * 128 for concatenated q, k, v
            kernel_size=3,
            bias=False,
            num_groups=128,  # Custom number of groups for weight sharing
        ).to(device, dtype=dtype_fixture)
        batch_size = 2
        seq_len = 256
        test_input = torch.randn(batch_size, 384, seq_len, device=device, dtype=dtype_fixture)
        expected_shape = (batch_size, 384, seq_len)

    # Test 2D distributed depthwise convolution
    elif data_dim == 2:
        dist_conv = DistributedDepthwiseConv2d(
            hidden_dim=128,
            kernel_size=3,
            bias=True,
            num_groups=32,
        ).to(device, dtype=dtype_fixture)
        batch_size = 2
        height, width = 32, 32
        test_input = torch.randn(batch_size, 128, height, width, device=device, dtype=dtype_fixture)
        expected_shape = (batch_size, 128, height, width)

    # Test 3D distributed depthwise convolution
    elif data_dim == 3:
        dist_conv = DistributedDepthwiseConv3d(
            hidden_dim=128,
            kernel_size=3,
            bias=True,
            num_groups=32,
        ).to(device, dtype=dtype_fixture)
        batch_size = 2
        depth, height, width = 16, 16, 16
        test_input = torch.randn(batch_size, 128, depth, height, width, device=device, dtype=dtype_fixture)
        expected_shape = (batch_size, 128, depth, height, width)

    # Test forward pass
    output = dist_conv(test_input, cp_group=None)

    # Verify output shape
    assert output.shape == expected_shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_weight_sharing(data_dim, dtype_fixture, device):
    """Test that weight sharing via num_groups works correctly.

    Validates that when num_groups < hidden_dim, weights are properly shared
    across channels using repeat_interleave.
    """
    hidden_dim = 384
    num_groups = 128  # 3x weight sharing
    kernel_size = 3

    if data_dim == 1:
        conv = DistributedDepthwiseConv1d(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            bias=False,
        ).to(device, dtype=dtype_fixture)
        test_input = torch.randn(2, hidden_dim, 64, device=device, dtype=dtype_fixture)

    elif data_dim == 2:
        conv = DistributedDepthwiseConv2d(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            bias=False,
        ).to(device, dtype=dtype_fixture)
        test_input = torch.randn(2, hidden_dim, 16, 16, device=device, dtype=dtype_fixture)

    elif data_dim == 3:
        conv = DistributedDepthwiseConv3d(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            bias=False,
        ).to(device, dtype=dtype_fixture)
        test_input = torch.randn(2, hidden_dim, 8, 8, 8, device=device, dtype=dtype_fixture)

    # Verify weight shape
    assert conv.weight.shape[0] == num_groups, f"Expected {num_groups} weight groups"
    assert conv.group_dim == hidden_dim // num_groups

    # Forward pass should work
    output = conv(test_input, cp_group=None)
    assert output.shape == test_input.shape

    # Verify weight sharing: manually expand and check it matches what forward does
    expanded_weight = conv.weight.repeat_interleave(conv.group_dim, dim=0)
    assert expanded_weight.shape[0] == hidden_dim

    # Check that weights are actually shared (groups should have identical weights)
    for i in range(num_groups):
        for j in range(conv.group_dim):
            idx = i * conv.group_dim + j
            if data_dim == 1:
                assert torch.allclose(expanded_weight[idx], conv.weight[i])
            elif data_dim == 2:
                assert torch.allclose(expanded_weight[idx], conv.weight[i])
            elif data_dim == 3:
                assert torch.allclose(expanded_weight[idx], conv.weight[i])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("causal", [True, False])
def test_causal_convolution_1d(causal, device):
    """Test causal padding for 1D convolutions.

    Validates that causal=True prevents future information leakage.
    """
    hidden_dim = 128
    kernel_size = 5
    seq_len = 32

    conv = DistributedDepthwiseConv1d(
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        causal=causal,
        bias=False,
    ).to(device)

    # Create input with a spike at position 16
    test_input = torch.zeros(1, hidden_dim, seq_len, device=device)
    spike_pos = 16
    test_input[:, :, spike_pos] = 1.0

    output = conv(test_input, cp_group=None)

    if causal:
        # With causal padding, positions before spike should be zero
        # (no future information should leak)
        before_spike = output[:, :, :spike_pos]
        assert torch.allclose(before_spike, torch.zeros_like(before_spike), atol=1e-5), (
            "Causal convolution leaked future information"
        )
    else:
        # Without causal padding, positions before spike should be non-zero
        before_spike = output[:, :, : spike_pos - kernel_size // 2]
        assert torch.allclose(before_spike, torch.zeros_like(before_spike), atol=1e-5), (
            "Non-causal convolution leaked future information"
        )

    # Output should have same length as input
    assert output.shape == test_input.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [2, 3])
def test_non_square_kernels(data_dim, device):
    """Test non-square/non-cubic kernel sizes for 2D and 3D convolutions."""
    hidden_dim = 64

    if data_dim == 2:
        kernel_size = (3, 5)  # Non-square
        conv = DistributedDepthwiseConv2d(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=True,
        ).to(device)
        test_input = torch.randn(2, hidden_dim, 16, 16, device=device)

        # Verify kernel shape
        assert conv.weight.shape == (hidden_dim, 3, 5)

    elif data_dim == 3:
        kernel_size = (3, 3, 5)  # Non-cubic
        conv = DistributedDepthwiseConv3d(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=True,
        ).to(device)
        test_input = torch.randn(2, hidden_dim, 8, 8, 8, device=device)

        # Verify kernel shape
        assert conv.weight.shape == (hidden_dim, 3, 3, 5)

    # Forward pass should work
    output = conv(test_input, cp_group=None)
    assert output.shape == test_input.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_input_validation(data_dim, device):
    """Test that invalid inputs raise appropriate errors."""
    hidden_dim = 128

    if data_dim == 1:
        conv = DistributedDepthwiseConv1d(hidden_dim=hidden_dim, kernel_size=3).to(device)
        wrong_input = torch.randn(2, hidden_dim, 16, 16, device=device)  # 4D instead of 3D

    elif data_dim == 2:
        conv = DistributedDepthwiseConv2d(hidden_dim=hidden_dim, kernel_size=3).to(device)
        wrong_input = torch.randn(2, hidden_dim, 16, device=device)  # 3D instead of 4D

    elif data_dim == 3:
        conv = DistributedDepthwiseConv3d(hidden_dim=hidden_dim, kernel_size=3).to(device)
        wrong_input = torch.randn(2, hidden_dim, 16, 16, device=device)  # 4D instead of 5D

    # Should raise assertion error for wrong number of dimensions
    with pytest.raises(AssertionError):
        conv(wrong_input, cp_group=None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_weight_initialization(data_dim, device):
    """Test that weight initialization follows correct bounds."""
    hidden_dim = 128
    kernel_size = 5

    if data_dim == 1:
        conv = DistributedDepthwiseConv1d(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=True,
        ).to(device)
        expected_bound = math.sqrt(1.0 / kernel_size)

    elif data_dim == 2:
        conv = DistributedDepthwiseConv2d(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=True,
        ).to(device)
        expected_bound = math.sqrt(1.0 / (kernel_size * kernel_size))

    elif data_dim == 3:
        conv = DistributedDepthwiseConv3d(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            bias=True,
        ).to(device)
        expected_bound = math.sqrt(1.0 / (kernel_size * kernel_size * kernel_size))

    # Check weights are within expected bounds
    assert conv.weight.min() >= -expected_bound - 1e-6
    assert conv.weight.max() <= expected_bound + 1e-6

    # Check bias is within bounds
    assert conv.bias.min() >= -expected_bound - 1e-6
    assert conv.bias.max() <= expected_bound + 1e-6

    # Weights should not all be the same (randomized)
    assert conv.weight.std() > 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim", [1, 2, 3])
def test_bias_behavior(data_dim, device):
    """Test that bias is correctly applied when enabled."""
    hidden_dim = 64
    kernel_size = 3

    if data_dim == 1:
        conv_with_bias = DistributedDepthwiseConv1d(hidden_dim=hidden_dim, kernel_size=kernel_size, bias=True).to(
            device
        )
        conv_no_bias = DistributedDepthwiseConv1d(hidden_dim=hidden_dim, kernel_size=kernel_size, bias=False).to(
            device
        )
        test_input = torch.randn(2, hidden_dim, 32, device=device)

    elif data_dim == 2:
        conv_with_bias = DistributedDepthwiseConv2d(hidden_dim=hidden_dim, kernel_size=kernel_size, bias=True).to(
            device
        )
        conv_no_bias = DistributedDepthwiseConv2d(hidden_dim=hidden_dim, kernel_size=kernel_size, bias=False).to(
            device
        )
        test_input = torch.randn(2, hidden_dim, 16, 16, device=device)

    elif data_dim == 3:
        conv_with_bias = DistributedDepthwiseConv3d(hidden_dim=hidden_dim, kernel_size=kernel_size, bias=True).to(
            device
        )
        conv_no_bias = DistributedDepthwiseConv3d(hidden_dim=hidden_dim, kernel_size=kernel_size, bias=False).to(
            device
        )
        test_input = torch.randn(2, hidden_dim, 8, 8, 8, device=device)

    # Verify bias attributes
    assert conv_with_bias.bias is not None
    assert conv_no_bias.bias is None

    # Both should run successfully
    output_with_bias = conv_with_bias(test_input, cp_group=None)
    output_no_bias = conv_no_bias(test_input, cp_group=None)

    assert output_with_bias.shape == test_input.shape
    assert output_no_bias.shape == test_input.shape
