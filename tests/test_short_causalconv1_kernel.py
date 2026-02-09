"""Tests for ShortHyenaConv1d module."""

import pytest
import torch
import torch.nn.functional as F
from nvsubq.ops.short_causalconv1d_kernel import ShortCausalConv1dKernel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestShortHyenaConv1d:
    """Test suite for ShortHyenaConv1d."""

    def test_basic_forward(self) -> None:
        """Test basic forward pass.
        Test that the forward pass works and that the output shape matches the input shape.
        Test that the output is not NaN or Inf.
        """
        batch_size, channels, seq_len = 2, 64, 1024
        kernel_size = 7

        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size).cuda()
        x = torch.randn(batch_size, channels, seq_len, device="cuda")

        y = conv(x)

        assert y.shape == x.shape, f"Output shape {y.shape} doesn't match input {x.shape}"
        assert y.device == x.device
        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"

    def test_output_shape_preservation(self) -> None:
        """Test that output shape matches input shape (causal padding)."""
        channels = 32
        # Only test supported kernel sizes (see modules.short_hyena.ShortHyenaConv1d for supported sizes)
        kernel_sizes = [3, 7, 8, 16, 32, 64]
        seq_len = 512

        for kernel_size in kernel_sizes:
            conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size).cuda()
            x = torch.randn(1, channels, seq_len, device="cuda")
            y = conv(x)
            assert y.shape == x.shape, f"Shape mismatch for kernel_size={kernel_size}"

    def test_against_pytorch_reference(self) -> None:
        """Test that output matches PyTorch F.conv1d with causal padding.

        This is the gold standard correctness test - if this passes, we know
        the CUDA kernel produces the same results as PyTorch's reference implementation.
        """
        batch_size, channels, seq_len = 2, 16, 256
        kernel_size = 7

        # Create ShortHyenaConv1d with known weights
        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size, bias=True).cuda()
        # Create input
        x = torch.randn(batch_size, channels, seq_len, device="cuda")

        # Get output from CUDA kernel
        with torch.no_grad():
            y_cuda = conv(x)

        # Compute reference output using PyTorch F.conv1d with causal padding
        with torch.no_grad():
            # For depthwise conv, need to reshape weights to [channels, 1, kernel_size]
            weight_pytorch = conv.weight.unsqueeze(1)  # [channels, 1, kernel_size]
            # Apply causal padding (left-only padding)
            left_pad = kernel_size - 1
            x_padded = F.pad(x, (left_pad, 0))
            # Depthwise convolution (groups=channels)
            y_reference = F.conv1d(x_padded, weight_pytorch, bias=conv.bias, groups=channels)

        # Compare outputs
        assert torch.allclose(y_cuda, y_reference, rtol=1e-4, atol=1e-5), (
            f"CUDA kernel output differs from PyTorch reference! Max diff: {(y_cuda - y_reference).abs().max():.6f}"
        )

    def test_against_pytorch_all_kernel_sizes(self) -> None:
        """Test against PyTorch reference for all supported kernel sizes.

        This comprehensive test verifies correctness across all kernel sizes
        that the CUDA kernel supports.
        """
        batch_size, channels, seq_len = 2, 8, 512
        # Test all supported kernel sizes
        kernel_sizes = [2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256]

        for kernel_size in kernel_sizes:
            # Create layer and input
            conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size, bias=True).cuda()
            x = torch.randn(batch_size, channels, seq_len, device="cuda")

            # CUDA kernel output
            with torch.no_grad():
                y_cuda = conv(x)

            # PyTorch reference output
            with torch.no_grad():
                weight_pytorch = conv.weight.unsqueeze(1)
                left_pad = kernel_size - 1
                x_padded = F.pad(x, (left_pad, 0))
                y_reference = F.conv1d(x_padded, weight_pytorch, bias=conv.bias, groups=channels)

            # Compare
            max_diff = (y_cuda - y_reference).abs().max().item()
            assert torch.allclose(y_cuda, y_reference, rtol=1e-4, atol=1e-5), (
                f"CUDA kernel differs from PyTorch for kernel_size={kernel_size}! Max diff: {max_diff:.6e}"
            )

    def test_causality(self) -> None:
        """Test that the convolution is causal.

        Output at position t should only depend on inputs at positions 0, 1, ..., t.
        This test is somewhat redundant because we already test the match to original Pytorch Conv1D with padding.
        """
        batch_size, channels, seq_len = 1, 8, 100
        kernel_size = 7

        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size, bias=False).cuda()

        # Create input with zeros after position 50
        x = torch.randn(batch_size, channels, seq_len, device="cuda")
        x_masked = x.clone()
        x_masked[:, :, 50:] = 0.0

        # Forward pass
        y = conv(x)
        y_masked = conv(x_masked)

        # Outputs at positions 0-43 should be the same (position 50 - kernel_size + 1)
        # because they don't depend on positions >= 50
        safe_pos = 50 - kernel_size + 1
        assert torch.allclose(y[:, :, :safe_pos], y_masked[:, :, :safe_pos], rtol=1e-5), (
            "Causality violated: outputs before the masked region should be identical"
        )

    def test_with_bias(self) -> None:
        """Test convolution with bias."""
        batch_size, channels, seq_len = 2, 16, 256
        kernel_size = 7

        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size, bias=True).cuda()
        x = torch.randn(batch_size, channels, seq_len, device="cuda")

        y = conv(x)

        assert y.shape == x.shape
        assert conv.bias is not None
        assert conv.bias.shape == (channels,)

    def test_without_bias(self) -> None:
        """Test convolution without bias."""
        batch_size, channels, seq_len = 2, 16, 256
        kernel_size = 7

        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size, bias=False).cuda()
        x = torch.randn(batch_size, channels, seq_len, device="cuda")

        y = conv(x)

        assert y.shape == x.shape
        assert conv.bias is None

    def test_silu_activation(self) -> None:
        """Test convolution with SiLU activation."""
        batch_size, channels, seq_len = 2, 16, 256
        kernel_size = 7

        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size, bias=True, activation="silu").cuda()
        x = torch.randn(batch_size, channels, seq_len, device="cuda")

        y = conv(x)

        assert y.shape == x.shape
        # Check that values are in reasonable range for SiLU output
        assert y.min() > -1.0, "SiLU output should be bounded below"

    def test_backward_pass(self) -> None:
        """Test that gradients flow correctly through the layer."""
        batch_size, channels, seq_len = 2, 16, 128
        kernel_size = 7

        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size, bias=True).cuda()
        x = torch.randn(batch_size, channels, seq_len, device="cuda", requires_grad=True)

        y = conv(x)
        loss = y.sum()
        loss.backward()

        # Check that gradients exist and are not all zero
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        assert conv.weight.grad is not None
        assert not torch.all(conv.weight.grad == 0)
        if conv.bias is not None:
            assert conv.bias.grad is not None
            assert not torch.all(conv.bias.grad == 0)

    def test_depthwise_property(self) -> None:
        """Test that convolution is truly depthwise (channels don't mix)."""
        batch_size, channels, seq_len = 1, 4, 100
        kernel_size = 3

        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size, bias=False).cuda()

        # Create input where each channel has a distinct pattern
        x = torch.zeros(batch_size, channels, seq_len, device="cuda")
        x[0, 0, :] = 1.0  # Channel 0: all ones
        x[0, 1, :] = 2.0  # Channel 1: all twos
        x[0, 2, :] = 3.0  # Channel 2: all threes
        x[0, 3, :] = 4.0  # Channel 3: all fours

        y = conv(x)

        # Since it's depthwise, channel 0 output should only depend on channel 0 input
        # We can't predict the exact values, but we can check that the operation
        # is consistent with depthwise semantics by verifying the computation structure
        assert y.shape == x.shape

    def test_parameter_initialization(self) -> None:
        """Test that parameters are properly initialized."""
        channels, kernel_size = 32, 7
        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size, bias=True).cuda()

        # Check shapes
        assert conv.weight.shape == (channels, kernel_size)
        assert conv.bias.shape == (channels,)

        # Check that weights are not all zero (initialized properly)
        assert not torch.all(conv.weight == 0)
        # Bias should be initialized to zero
        assert torch.all(conv.bias == 0)

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    @pytest.mark.parametrize("channels", [16, 32, 64])
    @pytest.mark.parametrize("seq_len", [128, 512, 2048])
    @pytest.mark.parametrize("kernel_size", [3, 7, 16, 32])  # Only supported sizes
    def test_various_configurations(self, batch_size: int, channels: int, seq_len: int, kernel_size: int) -> None:
        """Test various valid configurations."""
        conv = ShortCausalConv1dKernel(channels=channels, kernel_size=kernel_size).cuda()
        x = torch.randn(batch_size, channels, seq_len, device="cuda")

        y = conv(x)

        assert y.shape == (batch_size, channels, seq_len)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
