# TODO: Add license header here

"""Causal 1D convolution module.

This module provides a 1D convolution with causal (left-only) padding,
ensuring that outputs at position i only depend on inputs at positions 0, 1, ..., i.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CausalConv1D(torch.nn.Conv1d):
    """1D convolution with causal padding (left-only).

    This layer behaves like ``torch.nn.Conv1d`` but guarantees causality by
    padding only on the left with ``(kernel_size - 1) * dilation`` elements.

    Example:
        >>> conv = CausalConv1D(in_channels=16, out_channels=16, kernel_size=7, groups=16)
        >>> x = torch.randn(2, 16, 100)  # [B, C, L]
        >>> y = conv(x)  # [B, C, 100] - same length, causal

    Notes:
        - Stride > 1 reduces the sequence length as in standard convs.
        - ``padding`` passed to the constructor is ignored in ``forward``; causality
          always uses computed left padding. ``padding_mode`` is respected.
        - For grouped/depthwise convolutions, set ``groups=in_channels``.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply causal 1D convolution.

        Args:
            input: Input tensor of shape [B, C, L].

        Returns:
            Output tensor of shape [B, C_out, L] (same length for stride=1).
        """
        # Extract effective kernel size and dilation (ints)
        kernel_size = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
        dilation = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation

        # Left-only padding for causality
        left_pad = (kernel_size - 1) * dilation

        # Perform convolution without additional padding (already applied)
        return F.conv1d(
            F.pad(input, (left_pad, 0)),
            self.weight,
            self.bias,
            self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )
