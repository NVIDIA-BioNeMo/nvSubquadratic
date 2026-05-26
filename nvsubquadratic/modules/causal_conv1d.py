# TODO: Add license header here

"""Causal 1D convolution module.

Provides :class:`CausalConv1D`, a :class:`torch.nn.Conv1d` subclass that pads
sequences on the **left only**, enforcing the causal constraint that the output at
position ``i`` depends solely on inputs at positions ``0 … i``:

.. code-block:: text

    Causal padding (kernel_size=K, dilation=D):
        left_pad = (K - 1) * D        right_pad = 0

    For stride=1 the output length equals the input length:
        L_out = L_in   (same as "same" conv but strictly causal)

The module also supports **symmetric** (non-causal) same-padding when
``is_causal=False``, making it a drop-in for standard convolutions in contexts
that need a runtime-switchable causality flag.

**Use in Hyena short-conv configs**

In this repository, :class:`CausalConv1D` is wired into Hyena operators as the
short-conv component.  Concrete call sites include
``examples/spatial_recall_v2/mixer_defaults.py`` and
``examples/spatial_recall_1d/mixer_defaults.py``, which select
:class:`CausalConv1D` (``is_causal=True``) for the time axis and the symmetric
variant (``is_causal=False``) for spatial axes where the full context should be
visible.

**Difference from** ``subq_ops_causal_conv1d`` **/ ``causal_conv1d_custom``**

:class:`CausalConv1D` uses PyTorch's standard ``F.conv1d`` and works everywhere
(CPU, any GPU).  The modules in
:mod:`nvsubquadratic.ops.causal_conv1d_custom` and
:mod:`nvsubquadratic.modules.subq_ops_causal_conv1d` are thin wrappers around
the hand-fused ``causal_conv1d`` CUDA kernel from ``mamba_ssm``; they are
faster but require a compatible CUDA installation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CausalConv1D(torch.nn.Conv1d):
    """1D convolution with configurable causal (left-only) or symmetric padding.

    Subclasses :class:`torch.nn.Conv1d` and overrides :meth:`forward` to apply
    explicit left-only padding rather than the built-in ``padding`` argument.
    This guarantees that ``output[b, :, i]`` depends only on
    ``input[b, :, 0 … i]`` (the causal constraint).

    **Padding formulae**

    For a kernel of size ``K`` with dilation ``D``:

    - *Causal* (``is_causal=True``):
      ``left_pad = (K-1)*D``, ``right_pad = 0``.
      Output length equals input length for stride=1.
    - *Symmetric* (``is_causal=False``):
      ``sym_pad = ((K-1)*D) // 2`` on both sides.
      Equivalent to ``Conv1d(padding='same')`` for odd ``K``.

    Pass ``padding=0`` to the parent constructor (the default) — padding is
    handled explicitly in :meth:`forward`.

    Attributes:
        is_causal (bool): If ``True``, left-only padding is applied.

    Example::

        conv = CausalConv1D(in_channels=16, out_channels=16, kernel_size=7, groups=16)
        x = torch.randn(2, 16, 100)  # [B, C, L]
        y = conv(x)                  # [B, 16, 100] — same length, strictly causal

    Notes:
        - Stride > 1 reduces the output length as in standard convolutions.
        - For depthwise convolutions set ``groups=in_channels``.
        - The ``padding`` argument of the parent constructor is ignored; always
          pass ``padding=0`` (or omit it) when constructing this class.
    """

    def __init__(self, *args, is_causal: bool = True, **kwargs) -> None:
        """Initialise CausalConv1D.

        Args:
            *args: Positional arguments forwarded to :class:`torch.nn.Conv1d`
                (``in_channels``, ``out_channels``, ``kernel_size``, …).
            is_causal: If ``True`` (default), apply left-only (causal) padding.
                If ``False``, apply symmetric same-padding.
            **kwargs: Keyword arguments forwarded to :class:`torch.nn.Conv1d`.
                ``padding`` should be ``0`` (or omitted) since :meth:`forward`
                handles padding explicitly.
        """
        super().__init__(*args, **kwargs)
        self.is_causal = is_causal

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Apply 1D convolution with causal or symmetric padding.

        Args:
            input: Input tensor of shape ``[B, C_in, L]``.

        Returns:
            torch.Tensor: Output of shape ``[B, C_out, L_out]``.

            - ``L_out = L`` when ``stride=1`` (same-length output).
            - ``L_out = ceil(L / stride)`` for ``stride > 1``.
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
