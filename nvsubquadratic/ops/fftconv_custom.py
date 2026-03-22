# TODO: Add license header here


"""Drop-in wrappers around :mod:`subquadratic_ops_torch` CUDA FFT kernels.

This module mirrors the API of :mod:`nvsubquadratic.ops.fftconv` for 2D
operators while delegating the heavy lifting to the optimized CUDA kernel
provided by :mod:`subquadratic_ops_torch`.

Functions provided
------------------
- ``fftconv2d_bhl``  /  ``fftconv2d_bhl_chunked``   — BHL layout ``[B, H, X, Y]``
- ``fftconv2d_bhl_w_reshape``  /  ``fftconv2d_bhl_w_reshape_chunked``   — accepts BLH ``[B, X, Y, H]``, reshapes internally
- ``fftconv2d_blh``  /  ``fftconv2d_blh_chunked``   — aliases for the ``_w_reshape`` variants

All functions accept any input dtype (bf16, fp16, fp32) and internally cast to
fp32 for the CUDA kernel, returning the output in the original dtype. Shortcut
semantics are identical to the torch.fft reference: ``y += shortcut * x``.

The chunked variants process channels in groups of ``chunk_size`` to reduce
peak GPU memory from the CUDA kernel's FFT intermediates.

.. note::
   ``subquadratic_ops_torch`` is an **optional** dependency. Importing this
   module always succeeds; a clear error is raised only when a function is
   actually called without the package installed.
"""

from __future__ import annotations


__all__ = [
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
