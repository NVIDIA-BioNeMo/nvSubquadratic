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

"""Drop-in wrappers around :mod:`subquadratic_ops_torch` 1D causal conv kernels.

This module exposes thin Python wrappers around the upstream CUDA kernels for
*direct* (non-FFT) 1D causal convolutions:

- :func:`causal_conv1d` — depthwise causal 1D conv. Drop-in for the short
  conv path in Hyena when the host is 1D autoregressive. For long-range
  (kernel_size >= 128) prefer :mod:`fftconv_custom` instead.
  ``channel_last=True`` selects the channel-last (NWH) kernel, registered as a
  ``torch.library`` custom op with fake kernels and autograd; it is the
  ``torch.compile``-safe route.
- :func:`b2b_causal_conv1d` — fused back-to-back kernel (projection conv,
  pre-gate, mixer conv, post-gate). This is not a drop-in for any single
  module in nvSubquadratic; it's a building block for future fused-Hyena
  variants. Exposed here as a thin pass-through for callers that want to
  experiment with it directly.

.. note::
   ``subquadratic_ops_torch`` is an **optional** dependency. Importing this
   module always succeeds; a clear error is raised only when a function is
   actually called without the package installed.
"""

from __future__ import annotations


__all__ = [
    "b2b_causal_conv1d",
    "causal_conv1d",
]

import torch


_causal_conv1d = None
_b2b_causal_conv1d = None


def _get_causal_conv1d():
    global _causal_conv1d
    if _causal_conv1d is None:
        try:
            from subquadratic_ops_torch.causal_conv1d import causal_conv1d as _fn

            _causal_conv1d = _fn
        except ImportError as exc:
            raise ImportError(
                "subquadratic_ops_torch is required for causal_conv1d. "
                "Install the accelerated CUDA kernels with: pip install 'nvsubquadratic[cuda]'"
            ) from exc
    return _causal_conv1d


def _get_b2b_causal_conv1d():
    global _b2b_causal_conv1d
    if _b2b_causal_conv1d is None:
        try:
            from subquadratic_ops_torch.b2b_causal_conv1d import b2b_causal_conv1d as _fn

            _b2b_causal_conv1d = _fn
        except ImportError as exc:
            raise ImportError(
                "subquadratic_ops_torch is required for b2b_causal_conv1d. "
                "Install the accelerated CUDA kernels with: pip install 'nvsubquadratic[cuda]'"
            ) from exc
    return _b2b_causal_conv1d


# NWH custom ops whose fake kernels return the contiguous output the primitive produces.
@torch.library.custom_op(
    "nvsubquadratic::causal_conv1d_nwh",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def _causal_conv1d_nwh(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation: str) -> torch.Tensor:
    """Channel-last causal conv1d forward through the NWH primitive."""
    _get_causal_conv1d()
    return torch.ops.subquadratic_ops_torch.causal_conv1d_fwd_primitive(x, weight, bias, activation, True)


@_causal_conv1d_nwh.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation: str) -> torch.Tensor:
    """Fake kernel returning a contiguous output shaped like ``x``."""
    return x.new_empty(x.shape)


@torch.library.custom_op(
    "nvsubquadratic::causal_conv1d_nwh_backward",
    mutates_args=(),
    device_types="cuda",
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def _causal_conv1d_nwh_backward(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation: str
) -> list[torch.Tensor]:
    """Channel-last causal conv1d backward; returns ``dx`` and the packed weight and bias gradient."""
    return torch.ops.subquadratic_ops_torch.causal_conv1d_bwd_primitive(grad_out, x, weight, bias, activation, True)


@_causal_conv1d_nwh_backward.register_fake
def _(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, activation: str
) -> list[torch.Tensor]:
    """Fake kernel returning ``dx`` shaped like ``x`` and the packed weight and bias gradient."""
    return [x.new_empty(x.shape), weight.new_empty(weight.numel() + bias.numel())]


def _causal_conv1d_nwh_setup_context(ctx, inputs, output) -> None:
    """Save the inputs and activation for the backward pass."""
    x, weight, bias, activation = inputs
    ctx.save_for_backward(x, weight, bias)
    ctx.activation = activation


def _causal_conv1d_nwh_grad(ctx, grad_out):
    """Unpack the NWH backward into gradients for ``x``, ``weight`` and ``bias``."""
    x, weight, bias = ctx.saved_tensors
    dx, dwb = _causal_conv1d_nwh_backward(grad_out, x, weight, bias, ctx.activation)
    return dx, dwb[: weight.numel()].view(weight.shape), dwb[weight.numel() :], None


_causal_conv1d_nwh.register_autograd(_causal_conv1d_nwh_grad, setup_context=_causal_conv1d_nwh_setup_context)


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "identity",
    channel_last: bool = False,
) -> torch.Tensor:
    """Depthwise causal 1D conv via the subq_ops CUDA kernel.

    Args:
        x: Input tensor ``[B, C, L]``, or ``[B, L, C]`` when ``channel_last=True``.
        weight: Depthwise weight ``[C, K]``, or ``[K, C]`` when ``channel_last=True``.
        bias: Optional per-channel bias ``[C]``.
        activation: ``"identity"`` (default) or ``"silu"``.
        channel_last: Select the channel-last (NWH) kernel; the input layout and the
            weight layout follow it.  This path is safe under ``torch.compile``.

    Returns:
        Output tensor with the same shape as ``x``.
    """
    if channel_last:
        if bias is None:
            bias = x.new_zeros(weight.shape[1])
        return _causal_conv1d_nwh(x, weight, bias, activation)
    return _get_causal_conv1d()(x, weight, bias, activation)


def b2b_causal_conv1d(
    x: torch.Tensor,
    weight_proj: torch.Tensor,
    weight_mixer: torch.Tensor,
    skip_bias: torch.Tensor,
) -> torch.Tensor:
    """Back-to-back fused causal 1D conv via the subq_ops CUDA kernel.

    Fused kernel performing projection conv, pre-gate, mixer conv with skip,
    and post-gate.  See upstream docstring for the exact algorithm.

    Args:
        x: Input tensor ``[B, 3*C, L]``.
        weight_proj: Projection weight ``[3*C, K]`` (depthwise).
        weight_mixer: Mixer weight ``[C, K]`` (depthwise).
        skip_bias: Skip-bias scalar-per-channel ``[C]``.

    Returns:
        Output tensor ``[B, C, L]``.
    """
    return _get_b2b_causal_conv1d()(x, weight_proj, weight_mixer, skip_bias)
