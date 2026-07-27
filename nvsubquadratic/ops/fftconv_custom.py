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

2D fused (non-causal, zero-padded, **native dtype**):

- ``fused_fftconv2d_bhl`` / ``fused_fftconv2d_bhl_chunked`` and the matching
  ``_w_reshape`` / ``_blh`` variants. These wrap
  ``subquadratic_ops_torch.fused_fft_conv2d``, which runs the whole
  rfft2 → multiply → irfft2 pipeline as a single cuFFTDx kernel in the
  input dtype — no fp32 upcast — at the cost of a ``64x64`` spatial cap.

1D causal:

- ``causal_fftconv1d_bhl`` / ``causal_fftconv1d_bhl_chunked``: BHL layout
  ``[B, H, L]``.
- ``causal_fftconv1d_bhl_w_reshape`` /
  ``causal_fftconv1d_bhl_w_reshape_chunked``: accepts BLH ``[B, L, H]``,
  reshapes internally.
- ``causal_fftconv1d_blh`` / ``causal_fftconv1d_blh_chunked``: aliases for
  the ``_w_reshape`` variants.

The ``fftconv2d*`` / ``causal_fftconv1d*`` functions accept any input dtype
(bf16, fp16, fp32) and internally cast to fp32 for the CUDA kernel, returning
the output in the original dtype.  The ``fused_fftconv2d*`` functions instead
run the kernel *natively* in the input dtype, which is where their speedup
over the fp32 path comes from.  Shortcut semantics are identical throughout,
and match the torch.fft reference:
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
    "fused_fftconv2d_bhl",
    "fused_fftconv2d_bhl_chunked",
    "fused_fftconv2d_bhl_w_reshape",
    "fused_fftconv2d_bhl_w_reshape_chunked",
    "fused_fftconv2d_blh",
    "fused_fftconv2d_blh_chunked",
    "fused_fftconv2d_max_spatial",
    "fused_fftconv2d_supported",
    "load_fused_fft_conv2d",
    "resolve_fused_fft_size",
]

import torch
import torch.nn.functional as F
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
                "Install the accelerated CUDA kernels with: pip install 'nvsubquadratic[cuda]'"
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
# 2D fused FFT conv — wraps subquadratic_ops_torch.fused_fft_conv2d
#
# Unlike ``fft_conv2d`` above, this kernel runs natively in fp32/fp16/bf16 and
# fuses the filter FFT, the spectral multiply, and the inverse FFT into a
# single cuFFTDx launch.  Two contract differences drive the code below:
#
# 1. Square power-of-two FFT tile from a fixed table, with the input capped at
#    ``fft_size // 2`` — so the spatial extent is capped at 64x64.
# 2. The 'same' crop is taken at offset ``fft_size // 2``, whereas the torch
#    reference in :mod:`nvsubquadratic.ops.fftconv` crops at ``K // 2``.  We
#    reconcile the two by pre-padding the filter's top/left by the difference,
#    which shifts the convolution back by exactly that amount.  Without it the
#    fused path would be offset by ``fft_size // 2 - K // 2`` pixels relative to
#    every other backend.
# ===========================================================================

_fused_fft_conv2d = None

# FFT tile sizes the fused CUDA kernel is compiled for.
_FUSED_FFT_SIZES = (8, 16, 32, 64, 128)

# The kernel requires ``max(X, Y) <= fft_size // 2``, so the largest tile
# (128) caps the spatial extent at 64x64.
_FUSED_MAX_SPATIAL = max(_FUSED_FFT_SIZES) // 2


def _get_fused_fft_conv2d():
    """Return the ``fused_fft_conv2d`` callable, importing on first call."""
    global _fused_fft_conv2d
    if _fused_fft_conv2d is None:
        try:
            from subquadratic_ops_torch.fused_fft_conv2d import fused_fft_conv2d

            _fused_fft_conv2d = fused_fft_conv2d
        except ImportError as exc:
            raise ImportError(
                "subquadratic_ops_torch >= 0.2.2 is required for fft_backend='subq_ops_fused'. "
                "Install the accelerated CUDA kernels with: pip install 'nvsubquadratic[cuda]'"
            ) from exc
    return _fused_fft_conv2d


def load_fused_fft_conv2d() -> None:
    """Import the fused CUDA kernel now instead of on first call.

    Importing ``subquadratic_ops_torch.fused_fft_conv2d`` is what registers the
    ``torch.ops.subquadratic_ops_torch.fused_*`` operators. The wrappers here
    import it lazily, which is fine in eager mode — but a ``torch.compile``
    artifact restored from the on-disk FX cache references those operators
    without any Python call happening first, and fails with a confusing
    ``AttributeError`` on the op namespace. Callers that compile ahead of the
    first eager call should invoke this to force registration up front.

    Raises:
        ImportError: If ``subquadratic_ops_torch`` is not installed.
    """
    _get_fused_fft_conv2d()


def _next_pow2(n: int) -> int:
    """Smallest power of two >= ``n`` (with ``_next_pow2(0) == 1``)."""
    return 1 << max(n - 1, 0).bit_length()


def fused_fftconv2d_max_spatial() -> int:
    """Largest supported spatial extent per axis for the fused 2D kernel.

    Returns:
        ``64`` — the fused kernel's largest FFT tile is 128 and it requires
        ``max(X, Y) <= fft_size // 2``.
    """
    return _FUSED_MAX_SPATIAL


def resolve_fused_fft_size(
    x_dim: int,
    y_dim: int,
    k_x: int,
    k_y: int,
    fft_size: int | None = None,
) -> int:
    """Pick the FFT tile size for a fused 2D conv, or validate an explicit one.

    The tile must satisfy two constraints simultaneously:

    * ``fft_size // 2 >= max(x_dim, y_dim)`` — the kernel's own input cap.
    * ``fft_size // 2 >= ceil(k / 2)`` per axis — headroom for the top/left
      pre-pad of ``fft_size // 2 - k // 2`` that realigns the crop with the
      torch reference (see the section header).

    Args:
        x_dim: Input height ``X``.
        y_dim: Input width ``Y``.
        k_x: Kernel height ``K_x``.
        k_y: Kernel width ``K_y``.
        fft_size: Explicit tile size to validate. When ``None``, the smallest
            admissible power-of-two tile is chosen.

    Returns:
        A tile size drawn from ``(8, 16, 32, 64, 128)``.

    Raises:
        ValueError: If no admissible tile exists (shape too large for the fused
            kernel), or if an explicit ``fft_size`` is not in the supported set
            or is too small for the given shapes.
    """
    # Half-extent each of the four dims needs the tile to cover.
    needed_half = max(x_dim, y_dim, -(-k_x // 2), -(-k_y // 2))

    if fft_size is None:
        fft_size = 2 * _next_pow2(needed_half)
        if fft_size > max(_FUSED_FFT_SIZES):
            raise ValueError(
                f"Shape is too large for the fused 2D FFT kernel: input {(x_dim, y_dim)} with "
                f"kernel {(k_x, k_y)} needs fft_size={fft_size}, but the largest supported tile "
                f"is {max(_FUSED_FFT_SIZES)} (spatial dims capped at {_FUSED_MAX_SPATIAL} per axis). "
                "Use fft_backend='subq_ops' or 'torch_fft' for larger inputs."
            )
        return max(fft_size, min(_FUSED_FFT_SIZES))

    if fft_size not in _FUSED_FFT_SIZES:
        raise ValueError(f"fft_size must be one of {_FUSED_FFT_SIZES}, got {fft_size}.")
    if fft_size // 2 < needed_half:
        raise ValueError(
            f"fft_size={fft_size} is too small for input {(x_dim, y_dim)} with kernel "
            f"{(k_x, k_y)}: requires fft_size >= {2 * _next_pow2(needed_half)}."
        )
    return fft_size


def fused_fftconv2d_supported(x_dim: int, y_dim: int, k_x: int, k_y: int) -> bool:
    """Whether the fused 2D kernel can handle this input/kernel shape combination.

    A cheap, exception-free probe over the same rules as
    :func:`resolve_fused_fft_size` — useful for backend auto-selection and for
    the ``torch.compile`` pattern-match guard.

    Args:
        x_dim: Input height ``X``.
        y_dim: Input width ``Y``.
        k_x: Kernel height ``K_x``.
        k_y: Kernel width ``K_y``.

    Returns:
        ``True`` if an admissible FFT tile exists, ``False`` otherwise.
    """
    try:
        resolve_fused_fft_size(x_dim, y_dim, k_x, k_y)
    except ValueError:
        return False
    return True


def _fused_conv2d_bhl(x: torch.Tensor, kernel: torch.Tensor, fft_size: int | None) -> torch.Tensor:
    """Call the fused CUDA kernel on BHL tensors, in ``x``'s dtype.

    Pre-pads the filter's top/left by ``fft_size // 2 - K // 2`` per axis so the
    fused kernel's fixed ``fft_size // 2`` crop lands on the same window the
    torch reference produces with its ``K // 2`` crop.
    """
    fused_fft_conv2d = _get_fused_fft_conv2d()

    _B, _H, x_dim, y_dim = x.shape
    k_x, k_y = kernel.shape[-2], kernel.shape[-1]
    fft_size = resolve_fused_fft_size(x_dim, y_dim, k_x, k_y, fft_size)

    pad_x = fft_size // 2 - k_x // 2
    pad_y = fft_size // 2 - k_y // 2
    k = F.pad(kernel, (pad_y, 0, pad_x, 0))

    return fused_fft_conv2d(x.contiguous(), k.contiguous(), fft_size=fft_size)


def fused_fftconv2d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    fft_size: int | None = None,
) -> torch.Tensor:
    """Fused 2D FFT convolution, BHL layout ``[B, H, X, Y]``, native dtype.

    Drop-in replacement for :func:`nvsubquadratic.ops.fftconv.fftconv2d_fp32_bhl`
    that runs the CUDA kernel in ``x.dtype`` rather than upcasting to fp32.
    Numerically equivalent to the reference up to dtype roundoff (~3e-7
    relative in fp32, ~2e-3 in bf16).

    Args:
        x: Input tensor ``[B, H, X, Y]``, with ``X, Y <= 64``.
        kernel: Kernel tensor ``[1|B, H, Kx, Ky]``. Cast to ``x.dtype`` if needed.
        shortcut: Optional per-channel scale ``[H]``.
        fft_size: Optional explicit FFT tile size from ``(8, 16, 32, 64, 128)``.
            Defaults to the smallest admissible tile.

    Returns:
        Output tensor ``[B, H, X, Y]`` in ``x.dtype``.

    Raises:
        ValueError: If the shapes exceed what the fused kernel supports.
        ImportError: If ``subquadratic_ops_torch`` is not installed.
    """
    _B, H, _X, _Y = x.shape

    y = _fused_conv2d_bhl(x, kernel.to(x.dtype), fft_size)

    if shortcut is not None:
        y = y + shortcut.to(x.dtype).view(1, H, 1, 1) * x

    return y


def fused_fftconv2d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    fft_size: int | None = None,
) -> torch.Tensor:
    """Fused 2D FFT convolution for BLH inputs ``[B, X, Y, H]``.

    Reshapes to BHL, runs :func:`fused_fftconv2d_bhl`, reshapes back.
    """
    x_bhl = rearrange(x, "b x y h -> b h x y")
    kernel_bhl = rearrange(kernel, "b x y h -> b h x y")
    y_bhl = fused_fftconv2d_bhl(x_bhl, kernel_bhl, shortcut, fft_size)
    return rearrange(y_bhl, "b h x y -> b x y h")


def fused_fftconv2d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    fft_size: int | None = None,
) -> torch.Tensor:
    """Alias for :func:`fused_fftconv2d_bhl_w_reshape`."""
    return fused_fftconv2d_bhl_w_reshape(x, kernel, shortcut, fft_size)


def fused_fftconv2d_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
    fft_size: int | None = None,
) -> torch.Tensor:
    """Channel-chunked fused 2D FFT convolution, BHL layout.

    Processes channels in groups of ``chunk_size`` to cap the kernel's working
    set. The fused kernel's spatial tile is bounded at 64x64, so its per-call
    footprint is already small; chunking mainly helps at very large ``H``.

    Args:
        x: Input tensor ``[B, H, X, Y]``.
        kernel: Kernel tensor ``[1|B, H, Kx, Ky]``.
        shortcut: Optional per-channel scale ``[H]``.
        chunk_size: Channels per chunk (default 128).
        fft_size: Optional explicit FFT tile size.

    Returns:
        Output tensor ``[B, H, X, Y]`` in ``x.dtype``.
    """
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE

    _B, H, _X, _Y = x.shape
    k = kernel.to(x.dtype)

    chunks = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        chunks.append(_fused_conv2d_bhl(x[:, start:end].contiguous(), k[:, start:end].contiguous(), fft_size))

    y = torch.cat(chunks, dim=1)

    if shortcut is not None:
        y = y + shortcut.to(x.dtype).view(1, H, 1, 1) * x

    return y


def fused_fftconv2d_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
    fft_size: int | None = None,
) -> torch.Tensor:
    """Channel-chunked fused 2D FFT convolution for BLH inputs ``[B, X, Y, H]``."""
    x_bhl = rearrange(x, "b x y h -> b h x y")
    kernel_bhl = rearrange(kernel, "b x y h -> b h x y")
    y_bhl = fused_fftconv2d_bhl_chunked(x_bhl, kernel_bhl, shortcut, chunk_size, fft_size)
    return rearrange(y_bhl, "b h x y -> b x y h")


def fused_fftconv2d_blh_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
    fft_size: int | None = None,
) -> torch.Tensor:
    """Alias for :func:`fused_fftconv2d_bhl_w_reshape_chunked`."""
    return fused_fftconv2d_bhl_w_reshape_chunked(x, kernel, shortcut, chunk_size, fft_size)


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
                "Install the accelerated CUDA kernels with: pip install 'nvsubquadratic[cuda]'"
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
