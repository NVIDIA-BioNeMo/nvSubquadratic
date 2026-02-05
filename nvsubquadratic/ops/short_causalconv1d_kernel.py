# TODO: Add license header here

"""Short-range Hyena convolution using nvSubQuadratic-ops kernels.

PyTorch- wrapper around the `causal_conv1d` kernel from subquadratic-ops-torch-cu12.
causal_conv1d is optimized for short-range dependencies.

Supported kernel sizes: 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256

For very large kernel sizes (> 128), consider using long_hyena.py (FFT-based) instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Try to import optimized kernel if available
try:
    from subquadratic_ops_torch.causal_conv1d import causal_conv1d

    _CUDA_KERNEL_AVAILABLE = True
except ImportError:
    _CUDA_KERNEL_AVAILABLE = False
    causal_conv1d = None


class ShortCausalConv1dKernel(nn.Module):
    """Depth-wise causal 1D convolution using optimized CUDA kernels.

    This module wraps the `causal_conv1d` kernel from subquadratic-ops-torch,
    providing an optimized implementation for short-range causal convolutions.
    It's designed as a drop-in replacement for the short convolution component
    in Hyena layers.



    Args:
        channels (int): Number of input/output channels (for depth-wise conv).
        kernel_size (int): Size of the convolutional kernel. Must be one of:
            2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, or 256.
        bias (bool): If True, adds a learnable bias to the output. Default: True.
        activation (str): Activation function to apply. Options: "identity", "silu".
            Default: "identity".

    Raises:
        ImportError: If subquadratic-ops-torch-cu12 is not installed.
        ValueError: If kernel_size is not supported.

    Example:
        >>> # Create a short Hyena convolution layer
        >>> conv = ShortHyenaConv1d(channels=64, kernel_size=7, bias=True)
        >>> x = torch.randn(2, 64, 1024, device="cuda")  # [B, C, L]
        >>> y = conv(x)  # [2, 64, 1024] - same shape, causal
        >>> print(y.shape)
        torch.Size([2, 64, 1024])

    Notes:
        - This is a depthwise convolution: each channel is convolved independently (each channel has its own kernel).
        - The convolution is causal: output at position t only depends on inputs
          at positions 0, 1, ..., t.
        - For kernel sizes > 128, the FFT-based `LongHyenaFFTConv1d` should be used instead for better performance
        - Supports optional bias and activation
        - The CUDA kernel supports "silu" activation fused into the convolution
          for better performance
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
        activation: str = "identity",
    ):
        """Initialize the ShortHyenaConv1d layer."""
        super().__init__()

        if not _CUDA_KERNEL_AVAILABLE:
            raise ImportError(
                "subquadratic-ops-torch-cu12 is not installed. "
                "Please install it to use ShortHyenaConv1d. "
                "See README.md for installation instructions."
            )

        # Supported kernel sizes by the CUDA kernel
        supported_sizes = {2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256}
        if kernel_size not in supported_sizes:
            raise ValueError(
                f"kernel_size must be one of {sorted(supported_sizes)}. Got {kernel_size}. "
                f"For other kernel sizes, use PyTorch's CausalConv1D."
            )

        if activation not in ("identity", "silu"):
            raise ValueError(f"activation must be 'identity' or 'silu'. Got '{activation}'")

        self.channels = channels
        self.kernel_size = kernel_size
        self.activation = activation

        # Initialize depth-wise convolutional weights
        # Shape: (channels, kernel_size) - each channel has its own kernel
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.empty(channels))
        else:
            self.register_parameter("bias", None)

        # Initialize parameters.
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using kaiming initialization (standard PyTorch way).

        Uses Kaiming normal initialization, which is appropriate for layers
        followed by ReLU-like activations (including SiLU).
        """
        # Kaiming is also what is used in standard torch.nn.Conv1d and torch.nn.Conv2d.
        # fan_in = kernel_size (for depthwise conv, each output depends on kernel_size inputs)
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")

        # Initialize bias to zero if present
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depth-wise causal convolution.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, seq_len].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, channels, seq_len].
                The sequence length is preserved (causal padding is applied internally).

        Raises:
            AssertionError: If input tensor is not on CUDA device.
            AssertionError: If input channels don't match the layer's channels.
        """
        # Validate input
        assert x.is_cuda, "ShortHyenaConv1d requires CUDA tensors. Move input to GPU with .cuda()"
        assert x.ndim == 3, f"Input must be 3D (batch, channels, seq_len). Got shape {x.shape}"
        assert x.shape[1] == self.channels, (
            f"Input channels ({x.shape[1]}) must match layer channels ({self.channels})"
        )

        # return kernel output (causal padding is handled internally by kernel)
        return causal_conv1d(
            x,
            self.weight,
            bias=self.bias,
            activation=self.activation,
        )

    def extra_repr(self) -> str:
        """Return extra representation string for the module.

        Used when printing the module to show key configuration.
        """
        return (
            f"channels={self.channels}, "
            f"kernel_size={self.kernel_size}, "
            f"bias={self.bias is not None}, "
            f"activation='{self.activation}'"
        )


def is_cuda_kernel_available() -> bool:
    """Check if the CUDA kernel is available.

    Returns:
        bool: True if subquadratic-ops-torch-cu12 is installed and the
            causal_conv1d kernel is available.

    Example:
        >>> if is_cuda_kernel_available():
        ...     conv = ShortHyenaConv1d(channels=64, kernel_size=7)
        ... else:
        ...     # Fall back to PyTorch implementation
        ...     conv = torch.nn.Conv1d(64, 64, 7, groups=64)
    """
    return _CUDA_KERNEL_AVAILABLE
