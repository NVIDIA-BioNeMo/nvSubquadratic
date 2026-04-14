# TODO: Add license header here

"""Causal 1D convolution module.

This module provides a 1D convolution with causal (left-only) padding,
ensuring that outputs at position i only depend on inputs at positions 0, 1, ..., i.

Classes:
    CausalConv1D: Conv1d subclass with configurable causal or symmetric padding.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CausalConv1D(torch.nn.Conv1d):
    """1D convolution with configurable causal (left-only) or symmetric padding.

    When ``is_causal=True`` (default), pads only on the left so outputs at
    position *i* depend only on inputs at positions 0 … i.  When
    ``is_causal=False``, uses standard symmetric padding (same as Conv1d).

    Example:
        >>> conv = CausalConv1D(in_channels=16, out_channels=16, kernel_size=7, groups=16)
        >>> x = torch.randn(2, 16, 100)  # [B, C, L]
        >>> y = conv(x)  # [B, C, 100] - same length, causal

    Notes:
        - Stride > 1 reduces the sequence length as in standard convs.
        - For grouped/depthwise convolutions, set ``groups=in_channels``.
    """

    def __init__(self, *args, is_causal: bool = True, **kwargs) -> None:
        """Initialize CausalConv1D.

        Args:
            *args: Positional arguments forwarded to ``torch.nn.Conv1d``.
            is_causal: If True, apply left-only (causal) padding. If False,
                use standard symmetric padding.
            **kwargs: Keyword arguments forwarded to ``torch.nn.Conv1d``.
        """
        super().__init__(*args, **kwargs)
        self.is_causal = is_causal

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply 1D convolution with causal or symmetric padding.

        Args:
            input: Input tensor of shape [B, C, L].

        Returns:
            Output tensor of shape [B, C_out, L] (same length for stride=1).
        """
        kernel_size = self.kernel_size[0] if isinstance(self.kernel_size, tuple) else self.kernel_size
        dilation = self.dilation[0] if isinstance(self.dilation, tuple) else self.dilation

        if self.is_causal:
            # Left-only padding for causality
            left_pad = (kernel_size - 1) * dilation
            return F.conv1d(
                F.pad(input, (left_pad, 0)),
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            # Symmetric padding (standard Conv1d behavior)
            sym_pad = ((kernel_size - 1) * dilation) // 2
            return F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=sym_pad,
                dilation=self.dilation,
                groups=self.groups,
            )
