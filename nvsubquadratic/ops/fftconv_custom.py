# TODO: Add license header here


r"""Drop-in wrappers around the :mod:`subquadratic_ops_torch` custom CUDA FFT kernels.

The pure-PyTorch FFT path in :mod:`nvsubquadratic.ops.fftconv` is general
and correct, but each forward pass dispatches a chain of separate cuFFT,
element-wise multiply, and inverse-cuFFT kernels. The
:mod:`subquadratic_ops_torch` package ships hand-written CUDA kernels
(``fft_conv2d`` for 2D, ``fft_causal_conv1d`` for 1D causal) that fuse
these stages into a single launch, eliminating intermediate tensor traffic
and shaving wall-clock time on large shapes.

This module exposes those kernels through the **same API** as the PyTorch
operators in :mod:`nvsubquadratic.ops.fftconv`, so callers can switch
backends (e.g. via a ``fft_backend`` config flag) without touching their
model code.

Functions provided
------------------
2D (non-causal, zero-padded):

- ``fftconv2d_bhl`` / ``fftconv2d_bhl_chunked``: BHL layout ``[B, H, X, Y]``.
- ``fftconv2d_bhl_w_reshape`` / ``fftconv2d_bhl_w_reshape_chunked``: accepts
  BLH ``[B, X, Y, H]``, reshapes internally.
- ``fftconv2d_blh`` / ``fftconv2d_blh_chunked``: aliases for the
  ``_w_reshape`` variants (BLH naming convention).

1D causal:

- ``causal_fftconv1d_bhl`` / ``causal_fftconv1d_bhl_chunked``: BHL layout
  ``[B, H, L]``.
- ``causal_fftconv1d_bhl_w_reshape`` /
  ``causal_fftconv1d_bhl_w_reshape_chunked``: accepts BLH ``[B, L, H]``,
  reshapes internally.
- ``causal_fftconv1d_blh`` / ``causal_fftconv1d_blh_chunked``: aliases for
  the ``_w_reshape`` variants.

All functions accept any input dtype (bf16, fp16, fp32) and internally cast
to fp32 for the CUDA kernel, returning the output in the original dtype.
Shortcut semantics are identical to the torch.fft reference:
:math:`y \leftarrow y + \text{shortcut} \odot x`.

The chunked variants process channels in groups of ``chunk_size`` to reduce
peak GPU memory from the CUDA kernel's FFT intermediates — useful for very
wide hidden dims where the fused kernel's working set would otherwise
exceed device memory.

.. note::
   ``subquadratic_ops_torch`` is an **optional** dependency. Importing this
   module always succeeds; a clear ``ImportError`` is raised only when a
   function is actually called without the package installed.
"""

from __future__ import annotations


__all__ = [
    "causal_fftconv1d_bhl",
    "causal_fftconv1d_bhl_chunked",
    "causal_fftconv1d_bhl_w_reshape",
    "causal_fftconv1d_bhl_w_reshape_chunked",
    "causal_fftconv1d_blh",
    "causal_fftconv1d_blh_chunked",
    "fftconv2d_bhl",
    "fftconv2d_bhl_chunked",
    "fftconv2d_bhl_w_reshape",
    "fftconv2d_bhl_w_reshape_chunked",
    "fftconv2d_blh",
    "fftconv2d_blh_chunked",
]

import torch
from einops import rearrange


# ---------------------------------------------------------------------------
# Lazy import — cached on first use so the module can be imported without
# subquadratic_ops_torch being installed.
# ---------------------------------------------------------------------------
_fft_conv2d = None


def _get_fft_conv2d():
    """Return the ``fft_conv2d`` callable, importing on first call."""
    global _fft_conv2d
    if _fft_conv2d is None:
        try:
            from subquadratic_ops_torch.fft_conv2d import fft_conv2d

            _fft_conv2d = fft_conv2d
        except ImportError as exc:
            raise ImportError(
                "subquadratic_ops_torch is required for fft_backend='subq_ops'. "
                "Install it with: pip install subquadratic_ops_torch"
            ) from exc
    return _fft_conv2d


# ---------------------------------------------------------------------------
# Core helper — runs the CUDA kernel on fp32 tensors
# ---------------------------------------------------------------------------


def _subq_conv2d_bhl(x_fp32: torch.Tensor, k_fp32: torch.Tensor) -> torch.Tensor:
    """Call the subq_ops CUDA kernel on fp32 BHL tensors.

    Handles both shared kernels ``[1, H, Kx, Ky]`` (squeezed to ``[H, Kx, Ky]``)
    and FiLM per-sample kernels ``[B, H, Kx, Ky]`` (passed as-is).
    """
    fft_conv2d = _get_fft_conv2d()
    k = k_fp32.squeeze(0) if k_fp32.shape[0] == 1 else k_fp32
    return fft_conv2d(x_fp32.contiguous(), k.contiguous())


# ---------------------------------------------------------------------------
# Non-chunked functions
# ---------------------------------------------------------------------------


def fftconv2d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution via subq_ops CUDA kernel, BHL layout ``[B, H, X, Y]``.

    Drop-in replacement for :func:`nvsubquadratic.ops.fftconv.fftconv2d_fp32_bhl`.
    Accepts any input dtype; internally casts to fp32 for the CUDA kernel and
    returns the output in the original dtype of ``x``.

    Args:
        x: Input tensor ``[B, H, X, Y]``.
        kernel: Kernel tensor ``[1|B, H, Kx, Ky]``.
        shortcut: Optional per-channel scale ``[H]``.

    Returns:
        Output tensor ``[B, H, X, Y]`` in ``x.dtype``.
    """
    input_dtype = x.dtype
    _B, H, _X, _Y = x.shape

    y = _subq_conv2d_bhl(x.float(), kernel.float()).to(input_dtype)

    if shortcut is not None:
        y = y + shortcut.to(input_dtype).view(1, H, 1, 1) * x

    return y


def fftconv2d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution via subq_ops for BLH inputs ``[B, X, Y, H]``.

    Reshapes to BHL, runs :func:`fftconv2d_bhl`, reshapes back.
    """
    x_bhl = rearrange(x, "b x y h -> b h x y")
    kernel_bhl = rearrange(kernel, "b x y h -> b h x y")
    y_bhl = fftconv2d_bhl(x_bhl, kernel_bhl, shortcut)
    return rearrange(y_bhl, "b h x y -> b x y h")


def fftconv2d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Alias for :func:`fftconv2d_bhl_w_reshape`."""
    return fftconv2d_bhl_w_reshape(x, kernel, shortcut)


# ---------------------------------------------------------------------------
# Chunked functions — process channels in groups to reduce peak memory
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE = 128


def fftconv2d_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Channel-chunked 2D FFT convolution via subq_ops, BHL layout.

    Processes channels in groups of ``chunk_size`` to reduce peak GPU memory
    from the CUDA kernel's internal FFT intermediates.

    Args:
        x: Input tensor ``[B, H, X, Y]``.
        kernel: Kernel tensor ``[1|B, H, Kx, Ky]``.
        shortcut: Optional per-channel scale ``[H]``.
        chunk_size: Channels per chunk (default 128).

    Returns:
        Output tensor ``[B, H, X, Y]`` in ``x.dtype``.
    """
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE

    input_dtype = x.dtype
    _B, H, _X, _Y = x.shape
    x_fp32 = x.float()
    k_fp32 = kernel.float()

    chunks = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        x_c = x_fp32[:, start:end].contiguous()
        k_c = k_fp32[:, start:end].contiguous()
        chunks.append(_subq_conv2d_bhl(x_c, k_c))

    y = torch.cat(chunks, dim=1).to(input_dtype)

    if shortcut is not None:
        y = y + shortcut.to(input_dtype).view(1, H, 1, 1) * x

    return y


def fftconv2d_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Channel-chunked 2D FFT convolution via subq_ops for BLH inputs ``[B, X, Y, H]``.

    Reshapes to BHL, runs :func:`fftconv2d_bhl_chunked`, reshapes back.
    """
    x_bhl = rearrange(x, "b x y h -> b h x y")
    kernel_bhl = rearrange(kernel, "b x y h -> b h x y")
    y_bhl = fftconv2d_bhl_chunked(x_bhl, kernel_bhl, shortcut, chunk_size)
    return rearrange(y_bhl, "b h x y -> b x y h")


def fftconv2d_blh_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Alias for :func:`fftconv2d_bhl_w_reshape_chunked`."""
    return fftconv2d_bhl_w_reshape_chunked(x, kernel, shortcut, chunk_size)


# ===========================================================================
# 1D causal long FFT conv — wraps subquadratic_ops_torch.fft_causal_conv1d
# ===========================================================================

_fft_causal_conv1d = None


def _get_fft_causal_conv1d():
    """Return the ``fft_causal_conv1d`` callable, importing on first call."""
    global _fft_causal_conv1d
    if _fft_causal_conv1d is None:
        try:
            from subquadratic_ops_torch.fft_causal_conv1d import fft_causal_conv1d

            _fft_causal_conv1d = fft_causal_conv1d
        except ImportError as exc:
            raise ImportError(
                "subquadratic_ops_torch is required for fft_backend='subq_ops'. "
                "Install it with: pip install subquadratic_ops_torch"
            ) from exc
    return _fft_causal_conv1d


def _subq_causal_conv1d_bhl(x_fp32: torch.Tensor, k_fp32: torch.Tensor) -> torch.Tensor:
    """Call the subq_ops CUDA kernel on fp32 BHL tensors.

    Upstream signature: weight ``[H, K]``. Handles both shared ``[1, H, K]`` (squeezed)
    and 2D ``[H, K]`` (passed through).  Per-sample FiLM weights ``[B, H, K]`` are
    *not* supported by the upstream kernel — callers must guard against that case.
    """
    fft_causal_conv1d = _get_fft_causal_conv1d()
    if k_fp32.ndim == 3:
        if k_fp32.shape[0] != 1:
            raise NotImplementedError(
                "subquadratic_ops_torch.fft_causal_conv1d does not accept per-sample "
                f"FiLM weights. Got kernel shape {tuple(k_fp32.shape)} with batch={k_fp32.shape[0]}; "
                "expected shared kernel [1, H, K] or [H, K]."
            )
        k = k_fp32.squeeze(0)
    else:
        k = k_fp32
    return fft_causal_conv1d(x_fp32.contiguous(), k.contiguous())


def causal_fftconv1d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D causal FFT convolution via subq_ops CUDA kernel, BHL layout ``[B, H, L]``.

    Drop-in replacement for :func:`nvsubquadratic.ops.fftconv.causal_fftconv1d_fp32_bhl`.
    Accepts any input dtype; internally casts to fp32 for the CUDA kernel and
    returns the output in the original dtype of ``x``.

    Args:
        x: Input tensor ``[B, H, L]``.
        kernel: Kernel tensor ``[1, H, K]`` or ``[H, K]``.  Per-sample FiLM weights
            are not supported.
        shortcut: Optional per-channel scale ``[H]``.

    Returns:
        Output tensor ``[B, H, L]`` in ``x.dtype``.
    """
    input_dtype = x.dtype
    _B, H, _L = x.shape

    y = _subq_causal_conv1d_bhl(x.float(), kernel.float()).to(input_dtype)

    if shortcut is not None:
        y = y + shortcut.to(input_dtype).view(1, H, 1) * x

    return y


def causal_fftconv1d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D causal FFT convolution via subq_ops for BLH inputs ``[B, L, H]``.

    Reshapes to BHL, runs :func:`causal_fftconv1d_bhl`, reshapes back.
    """
    x_bhl = rearrange(x, "b l h -> b h l")
    kernel_bhl = rearrange(kernel, "b l h -> b h l")
    y_bhl = causal_fftconv1d_bhl(x_bhl, kernel_bhl, shortcut)
    return rearrange(y_bhl, "b h l -> b l h")


def causal_fftconv1d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Alias for :func:`causal_fftconv1d_bhl_w_reshape`."""
    return causal_fftconv1d_bhl_w_reshape(x, kernel, shortcut)


def causal_fftconv1d_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Channel-chunked 1D causal FFT convolution via subq_ops, BHL layout."""
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE

    input_dtype = x.dtype
    _B, H, _L = x.shape
    x_fp32 = x.float()
    k_fp32 = kernel.float()

    chunks = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        x_c = x_fp32[:, start:end].contiguous()
        k_c = k_fp32[:, start:end].contiguous()
        chunks.append(_subq_causal_conv1d_bhl(x_c, k_c))

    y = torch.cat(chunks, dim=1).to(input_dtype)

    if shortcut is not None:
        y = y + shortcut.to(input_dtype).view(1, H, 1) * x

    return y


def causal_fftconv1d_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Channel-chunked 1D causal FFT convolution via subq_ops for BLH inputs."""
    x_bhl = rearrange(x, "b l h -> b h l")
    kernel_bhl = rearrange(kernel, "b l h -> b h l")
    y_bhl = causal_fftconv1d_bhl_chunked(x_bhl, kernel_bhl, shortcut, chunk_size)
    return rearrange(y_bhl, "b h l -> b l h")


def causal_fftconv1d_blh_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Alias for :func:`causal_fftconv1d_bhl_w_reshape_chunked`."""
    return causal_fftconv1d_bhl_w_reshape_chunked(x, kernel, shortcut, chunk_size)
