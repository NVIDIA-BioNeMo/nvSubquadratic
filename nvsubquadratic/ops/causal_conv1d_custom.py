# TODO: Add license header here


"""Drop-in wrappers around :mod:`subquadratic_ops_torch` 1D causal conv kernels.

This module exposes thin Python wrappers around the upstream CUDA kernels for
*direct* (non-FFT) 1D causal convolutions:

- :func:`causal_conv1d` — depthwise causal 1D conv. Drop-in for the short
  conv path in Hyena when the host is 1D autoregressive. For long-range
  (kernel_size >= 128) prefer :mod:`fftconv_custom` instead.
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
                "Install it with: pip install subquadratic_ops_torch"
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
                "Install it with: pip install subquadratic_ops_torch"
            ) from exc
    return _b2b_causal_conv1d


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str = "identity",
) -> torch.Tensor:
    """Depthwise causal 1D conv via the subq_ops CUDA kernel.

    Args:
        x: Input tensor ``[B, C, L]``.
        weight: Depthwise weight ``[C, K]``.
        bias: Optional per-channel bias ``[C]``.
        activation: ``"identity"`` (default) or ``"silu"``.

    Returns:
        Output tensor ``[B, C, L]`` (same shape as input).
    """
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
