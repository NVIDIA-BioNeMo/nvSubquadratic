# TODO: Add license header here


r"""Mixed boundary-condition FFT-based convolution operators (fp32).

This module implements N-D depthwise FFT convolutions that allow each
spatial axis to independently use either:

- **Periodic** (circular) boundary conditions  → frequency-domain phase
  ramp, no padding, no crop on that axis.
- **Non-periodic** (zero-padded "same") boundary conditions  → FFT length
  is padded up so wrap-around cancels out, centered crop on that axis.

The choice is made per axis via a ``periodic: tuple[bool, ...]`` argument
of length equal to the number of spatial dimensions.

Why this op?
------------
Many PDE datasets (e.g. Well's ``rayleigh_benard``,
``viscoelastic_instability``, ``turbulent_radiative_layer``,
``rayleigh_taylor_instability``) have boundaries that are **periodic on
some axes and non-periodic on others**. A single global ``"zero"`` or
``"circular"`` mode is incorrect for all of these: zero-padding leaks
the wall/open boundary into periodic axes, and circular wraps the
non-periodic ones. This module is the FFT-conv-side fix.

Relation to existing ops
------------------------
- All ``periodic = False`` → bit-equivalent to
  :mod:`nvsubquadratic.ops.fftconv` (linear / zero-padded "same").
- All ``periodic = True`` → bit-equivalent to
  :mod:`nvsubquadratic.ops.circular_fftconv` (circular / periodic).
- Mixed → new per-axis recipe (see :func:`_mixed_recipe`).

The op routes the all-False / all-True cases through the existing
non-mixed paths automatically, so adopters pay no overhead for the
legacy modes.

Layouts and shapes
------------------
- **BHL** (channels-first, the fast path): ``[B, H, * spatial_dims]``;
  kernel ``[1|B, H, * K_dims]``; output ``[B, H, * spatial_dims]``.
- **BLH** wrappers (``*_w_reshape``) transparently reshape BLH → BHL → BLH.

The leading kernel dimension may be ``1`` (kernel shared across the batch)
or ``B`` (per-sample kernel, e.g. FiLM-conditioned Hyena).

Shortcut
--------
Optional ``shortcut: [H]`` adds a per-channel residual scale of the input
to the convolution output:

.. math::
    y \leftarrow y + \text{shortcut} \odot x

Phase ramps
-----------
On each periodic axis we align the output to "same" convolution by an
integer pixel shift of :math:`s_d = -\lfloor (K_d - 1) / 2 \rfloor`,
implemented as a frequency-domain phase ramp
:math:`\exp(-i 2\pi f_d s_d)`. Non-periodic axes use ``s_d = 0`` (the
alignment is absorbed into the centered crop). When
``use_phase_shift=False`` we instead apply ``torch.roll`` along the
periodic axes after the inverse transform.

Caching
-------
Per-axis 1-D phase ramps are cached in a small module-level LRU
(``_MIXED_PHASE_RAMP_1D_CACHE``). The N-D ramp is constructed by
broadcasted multiplication of the relevant 1-D ramps on demand — no
N-D ramp is materialised in the cache. Axes with shift 0 contribute
nothing (they are skipped entirely).
"""

from __future__ import annotations


__all__ = [
    "mixed_fftconv1d_fp32_bhl",
    "mixed_fftconv1d_fp32_bhl_chunked",
    "mixed_fftconv1d_fp32_bhl_w_reshape",
    "mixed_fftconv1d_fp32_bhl_w_reshape_chunked",
    "mixed_fftconv2d_fp32_bhl",
    "mixed_fftconv2d_fp32_bhl_chunked",
    "mixed_fftconv2d_fp32_bhl_w_reshape",
    "mixed_fftconv2d_fp32_bhl_w_reshape_chunked",
    "mixed_fftconv3d_fp32_bhl",
    "mixed_fftconv3d_fp32_bhl_chunked",
    "mixed_fftconv3d_fp32_bhl_w_reshape",
    "mixed_fftconv3d_fp32_bhl_w_reshape_chunked",
]

import math
from collections import OrderedDict
from collections.abc import Sequence

import torch
from einops import rearrange

import nvsubquadratic.ops.fftconv as _fftconv_module


# =============================================================================
# Per-axis 1-D phase ramp cache
# =============================================================================


class _MixedPhaseRamp1DCache:
    """LRU cache of 1-D frequency-domain phase ramps used by mixed FFT conv.

    Each cached entry is a complex tensor of shape ``[F]`` (regular FFT axis)
    or ``[F // 2 + 1]`` (rfft last axis) that encodes
    ``exp(-i 2π f · s)`` for the given size ``F`` and integer shift ``s``.

    The N-D ramp is built lazily by broadcasting the relevant 1-D ramps;
    no N-D tensor is stored in the cache.
    """

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()

    @staticmethod
    def _key(
        F: int,
        s: int,
        is_rfft_axis: bool,
        device: torch.device,
        real_dtype: torch.dtype,
    ) -> tuple:
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        dev_type = device.type
        dev_idx = device.index if device.index is not None else -1
        return (F, s, is_rfft_axis, dev_type, dev_idx, complex_dtype)

    def get(
        self,
        F: int,
        s: int,
        is_rfft_axis: bool,
        device: torch.device,
        real_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the 1-D phase ramp for the given (FFT length, shift) on this axis.

        Args:
            F: FFT length along this axis (padded length for non-periodic,
                input length for periodic — caller decides which).
            s: Integer pixel shift to apply via the ramp. ``s == 0`` callers
                are expected to skip the multiply entirely; this method
                still supports ``s == 0`` (returns all-ones for completeness).
            is_rfft_axis: If True, this is the last spatial axis where the
                rfft is taken; we build a ramp of length ``F // 2 + 1`` using
                :func:`torch.fft.rfftfreq`. Otherwise we build a length-``F``
                ramp using :func:`torch.fft.fftfreq`.
            device: Target device.
            real_dtype: Real-valued dtype of the inputs (float32 or float64).
        """
        key = self._key(F, s, is_rfft_axis, device, real_dtype)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        with torch.inference_mode(False):
            with torch.no_grad():
                if is_rfft_axis:
                    freqs = torch.fft.rfftfreq(F, d=1.0, device=device, dtype=real_dtype)
                else:
                    freqs = torch.fft.fftfreq(F, d=1.0, device=device, dtype=real_dtype)
                phases = -2.0 * math.pi * (s * freqs)
                ramp = torch.complex(torch.cos(phases), torch.sin(phases))
        self._cache[key] = ramp
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return ramp


_MIXED_PHASE_RAMP_1D_CACHE = _MixedPhaseRamp1DCache(maxsize=256)


# =============================================================================
# Per-axis recipe
# =============================================================================


def _mixed_recipe(
    spatial: tuple[int, ...],
    kshape: tuple[int, ...],
    periodic: tuple[bool, ...],
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Compute the per-axis FFT length, crop window, and phase-ramp shift.

    For each spatial axis d:

    - If ``periodic[d]`` is True (periodic / circular):
      ``F_d = N_d`` (no padding), crop ``[0, N_d)`` (no crop), shift
      ``s_d = -((K_d - 1) // 2)``.
    - If ``periodic[d]`` is False (zero-padded "same"):
      ``F_d = min(N_d + (K_d + 1) // 2, 2 * N_d)``, crop
      ``[K_d // 2, K_d // 2 + N_d)``, shift ``s_d = 0`` (alignment is
      handled by the centered crop).

    Args:
        spatial: Input spatial dims ``(N_0, ..., N_{D-1})``.
        kshape:  Kernel spatial dims ``(K_0, ..., K_{D-1})``.
        periodic: Per-axis periodicity flags.

    Returns:
        ``(fft_shape, crops, shifts)``:
            - ``fft_shape``: ``(F_0, ..., F_{D-1})`` — to be passed as ``s=``.
            - ``crops``: ``((start_0, stop_0), ...)`` for the post-IFFT slice.
            - ``shifts``: ``(s_0, ..., s_{D-1})`` integer pixel shifts.
    """
    assert len(spatial) == len(kshape) == len(periodic), (
        f"Length mismatch: spatial={len(spatial)}, kernel={len(kshape)}, periodic={len(periodic)}"
    )
    fft_shape: list[int] = []
    crops: list[tuple[int, int]] = []
    shifts: list[int] = []
    for N, K, is_periodic in zip(spatial, kshape, periodic):
        if is_periodic:
            fft_shape.append(N)
            crops.append((0, N))
            shifts.append(-((K - 1) // 2))
        else:
            F = min(N + (K + 1) // 2, 2 * N)
            fft_shape.append(F)
            crops.append((K // 2, K // 2 + N))
            shifts.append(0)
    return tuple(fft_shape), tuple(crops), tuple(shifts)


def _build_nd_phase_ramp(
    fft_shape: tuple[int, ...],
    shifts: tuple[int, ...],
    device: torch.device,
    real_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Build (or fetch from cache) the broadcast N-D phase-ramp tensor.

    The returned tensor has rfft-style shape
    ``(F_0, F_1, ..., F_{D-2}, F_{D-1} // 2 + 1)`` with complex dtype,
    suitable to be multiplied with ``torch.fft.rfftn(..., s=fft_shape)``.

    Axes with ``shift == 0`` contribute a length-1 broadcast factor (no
    multiply along that axis). If every shift is zero, this function
    returns ``None`` — callers should skip the multiply entirely.

    The N-D ramp itself is not cached; only the 1-D per-axis ramps are.
    The product of 1-D ramps over broadcasted dims is materialised here
    so that the downstream multiply with ``fft_x`` is a single op.
    """
    if all(s == 0 for s in shifts):
        return None

    D = len(fft_shape)
    last_axis = D - 1
    nd_ramp: torch.Tensor | None = None

    for d, (F, s) in enumerate(zip(fft_shape, shifts)):
        if s == 0:
            continue
        is_rfft_axis = d == last_axis
        ramp_1d = _MIXED_PHASE_RAMP_1D_CACHE.get(F, s, is_rfft_axis, device, real_dtype)

        view_shape = [1] * D
        view_shape[d] = ramp_1d.shape[0]
        ramp_1d_view = ramp_1d.view(*view_shape)

        if nd_ramp is None:
            nd_ramp = ramp_1d_view
        else:
            nd_ramp = nd_ramp * ramp_1d_view
    return nd_ramp


# =============================================================================
# Argument normalisation & dispatch helpers
# =============================================================================


def _normalize_periodic(periodic: Sequence[bool] | tuple[bool, ...], data_dim: int) -> tuple[bool, ...]:
    """Normalise the ``periodic`` argument to a length-``data_dim`` tuple of bools."""
    periodic_t = tuple(bool(p) for p in periodic)
    assert len(periodic_t) == data_dim, (
        f"periodic must have length {data_dim} (data_dim), got length {len(periodic_t)}"
    )
    return periodic_t


def _dispatch_legacy_if_uniform(
    periodic: tuple[bool, ...],
    data_dim: int,
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None,
    use_phase_shift: bool,
) -> torch.Tensor | None:
    """If ``periodic`` is uniformly all-True or all-False, route to the existing op.

    Returns the output tensor if the call was dispatched, or ``None`` to
    indicate the caller should run the mixed path itself.

    The legacy linear ops do not take ``use_phase_shift`` (it does not
    apply there); we always dispatch when periodic is all-False.
    """
    if all(periodic):
        from nvsubquadratic.ops import circular_fftconv as _circ

        fn_table = {
            1: _circ.circular_fftconv1d_fp32_bhl,
            2: _circ.circular_fftconv2d_fp32_bhl,
            3: _circ.circular_fftconv3d_fp32_bhl,
        }
        return fn_table[data_dim](x, kernel, shortcut, use_phase_shift=use_phase_shift)

    if not any(periodic):
        fn_table = {
            1: _fftconv_module.fftconv1d_fp32_bhl,
            2: _fftconv_module.fftconv2d_fp32_bhl,
            3: _fftconv_module.fftconv3d_fp32_bhl,
        }
        return fn_table[data_dim](x, kernel, shortcut)
    return None


# =============================================================================
# Core mixed conv (BHL, fp32)
# =============================================================================


def _mixed_fftconv_nd_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: tuple[bool, ...],
    shortcut: torch.Tensor | None,
    use_phase_shift: bool,
    data_dim: int,
) -> torch.Tensor:
    """Shared N-D mixed-BC FFT conv body (BHL, fp32).

    The 1D/2D/3D public entry points delegate here after shape validation.
    All padding/cropping/phase-ramp logic is per axis according to
    :func:`_mixed_recipe`.
    """
    x_shape = x.shape
    assert x.ndim == 2 + data_dim, f"Expected {2 + data_dim}D input, got {x.ndim}D"
    assert kernel.ndim == 2 + data_dim, f"Expected {2 + data_dim}D kernel, got {kernel.ndim}D"

    B = x_shape[0]
    H = x_shape[1]
    spatial = tuple(x_shape[2:])
    kshape = tuple(kernel.shape[2:])

    assert kernel.shape[0] in (1, B), (
        f"Leading kernel dim must be 1 or batch_size ({B}). Got kernel.shape={tuple(kernel.shape)}."
    )
    assert kernel.shape[1] == H, f"Kernel hidden dim ({kernel.shape[1]}) must equal input hidden dim ({H})."
    # Per-axis kernel-size limits:
    # - Periodic axis: FFT length is the input length N, so the kernel must
    #   fit inside that, i.e. ``K_d <= N_d``.
    # - Non-periodic axis: FFT length is the padded length ``min(N + (K+1)//2,
    #   2N)``, so kernel sizes up to ``2 * N_d`` are well-defined (this is
    #   what the legacy ``fftconv*`` linear ops accept, and what the standard
    #   "double grid" kernel size ``2N - 1`` produces).
    for d, (N, K, is_periodic) in enumerate(zip(spatial, kshape, periodic)):
        if is_periodic:
            assert K <= N, f"K must be <= N on periodic axis {d}: K={K}, N={N}."
        else:
            assert K <= 2 * N, f"K must be <= 2*N on non-periodic axis {d}: K={K}, N={N}."

    legacy = _dispatch_legacy_if_uniform(periodic, data_dim, x, kernel, shortcut, use_phase_shift)
    if legacy is not None:
        return legacy

    x_fp32 = x.to(torch.float32)
    k_fp32 = kernel.to(torch.float32)

    fft_shape, crops, shifts = _mixed_recipe(spatial, kshape, periodic)

    fft_dims = tuple(range(2, 2 + data_dim))
    fft_x = torch.fft.rfftn(x_fp32, s=fft_shape, dim=fft_dims)
    fft_k = torch.fft.rfftn(k_fp32, s=fft_shape, dim=fft_dims)

    if use_phase_shift:
        ramp = _build_nd_phase_ramp(fft_shape, shifts, x_fp32.device, x_fp32.dtype)
        if ramp is not None:
            fft_k = fft_k * ramp

    if _fftconv_module.COMPILE_COMPATIBLE:
        fft_x = _fftconv_module._complex_mul_real(fft_x, fft_k)
    else:
        fft_x.mul_(fft_k)

    y = torch.fft.irfftn(fft_x, s=fft_shape, dim=fft_dims)

    if not use_phase_shift and any(s != 0 for s in shifts):
        roll_shifts = tuple(s for s in shifts if s != 0)
        roll_dims = tuple(2 + d for d, s in enumerate(shifts) if s != 0)
        y = torch.roll(y, shifts=roll_shifts, dims=roll_dims)

    crop_index = (slice(None), slice(None)) + tuple(slice(start, stop) for start, stop in crops)
    y = y[crop_index]

    y = y.to(x.dtype)
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (H,), f"Expected shortcut shape ({H},), got {tuple(shortcut.shape)}"
        broadcast_shape = (1, H) + (1,) * data_dim
        y = y + shortcut.view(*broadcast_shape) * x
    return y


# =============================================================================
# Public 1D / 2D / 3D entry points (BHL, fp32)
# =============================================================================


def mixed_fftconv1d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """1D mixed-BC FFT convolution (BHL layout).

    The single spatial axis can be either periodic (circular) or non-periodic
    (zero-padded "same"), selected by ``periodic`` (length 1).

    Args:
        x: Input tensor of shape ``[B, H, L]`` (any dtype, internally cast to fp32).
        kernel: Kernel tensor of shape ``[1|B, H, K]`` (any dtype, cast to fp32).
        periodic: Length-1 sequence of bools. ``periodic[0] == True`` ⇒ circular.
        shortcut: Optional per-channel scale ``[H]`` added as ``y += shortcut * x``.
        use_phase_shift: If True, align periodic axes via frequency-domain phase
            ramps. If False, align via :func:`torch.roll` on periodic axes after
            the inverse transform. The output is mathematically equivalent.

    Returns:
        Tensor of shape ``[B, H, L]`` in the original dtype of ``x``.
    """
    periodic_t = _normalize_periodic(periodic, data_dim=1)
    return _mixed_fftconv_nd_fp32_bhl(x, kernel, periodic_t, shortcut, use_phase_shift, data_dim=1)


def mixed_fftconv2d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """2D mixed-BC FFT convolution (BHL layout).

    Each of the two spatial axes independently uses periodic (circular) or
    non-periodic (zero-padded "same") boundary handling.

    Args:
        x: Input tensor of shape ``[B, H, X, Y]`` (any dtype, internally cast to fp32).
        kernel: Kernel tensor of shape ``[1|B, H, K_x, K_y]`` (any dtype, cast to fp32).
        periodic: Length-2 sequence ``(periodic_x, periodic_y)``.
        shortcut: Optional per-channel scale ``[H]`` added as ``y += shortcut * x``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.

    Returns:
        Tensor of shape ``[B, H, X, Y]`` in the original dtype of ``x``.
    """
    periodic_t = _normalize_periodic(periodic, data_dim=2)
    return _mixed_fftconv_nd_fp32_bhl(x, kernel, periodic_t, shortcut, use_phase_shift, data_dim=2)


def mixed_fftconv3d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """3D mixed-BC FFT convolution (BHL layout).

    Each of the three spatial axes independently uses periodic (circular) or
    non-periodic (zero-padded "same") boundary handling.

    Args:
        x: Input tensor of shape ``[B, H, X, Y, Z]`` (any dtype, cast to fp32).
        kernel: Kernel tensor of shape ``[1|B, H, K_x, K_y, K_z]``.
        periodic: Length-3 sequence ``(periodic_x, periodic_y, periodic_z)``.
        shortcut: Optional per-channel scale ``[H]``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.

    Returns:
        Tensor of shape ``[B, H, X, Y, Z]`` in the original dtype of ``x``.
    """
    periodic_t = _normalize_periodic(periodic, data_dim=3)
    return _mixed_fftconv_nd_fp32_bhl(x, kernel, periodic_t, shortcut, use_phase_shift, data_dim=3)


# =============================================================================
# BLH wrappers (channels-last)
# =============================================================================


def mixed_fftconv1d_fp32_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """1D mixed-BC FFT conv wrapper for BLH layout (batch, length, hidden).

    Reshapes BLH ↔ BHL around :func:`mixed_fftconv1d_fp32_bhl`. See that
    function for argument semantics.
    """
    x_bhl = rearrange(x, "b l h -> b h l")
    k_bhl = rearrange(kernel, "b k h -> b h k")
    y_bhl = mixed_fftconv1d_fp32_bhl(x_bhl, k_bhl, periodic, shortcut, use_phase_shift=use_phase_shift)
    return rearrange(y_bhl, "b h l -> b l h")


def mixed_fftconv2d_fp32_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """2D mixed-BC FFT conv wrapper for BLH layout (batch, X, Y, hidden).

    Reshapes BLH ↔ BHL around :func:`mixed_fftconv2d_fp32_bhl`.
    """
    x_bhl = rearrange(x, "b x y h -> b h x y")
    k_bhl = rearrange(kernel, "b kx ky h -> b h kx ky")
    y_bhl = mixed_fftconv2d_fp32_bhl(x_bhl, k_bhl, periodic, shortcut, use_phase_shift=use_phase_shift)
    return rearrange(y_bhl, "b h x y -> b x y h")


def mixed_fftconv3d_fp32_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """3D mixed-BC FFT conv wrapper for BLH layout (batch, X, Y, Z, hidden).

    Reshapes BLH ↔ BHL around :func:`mixed_fftconv3d_fp32_bhl`.
    """
    x_bhl = rearrange(x, "b x y z h -> b h x y z")
    k_bhl = rearrange(kernel, "b kx ky kz h -> b h kx ky kz")
    y_bhl = mixed_fftconv3d_fp32_bhl(x_bhl, k_bhl, periodic, shortcut, use_phase_shift=use_phase_shift)
    return rearrange(y_bhl, "b h x y z -> b x y z h")


# =============================================================================
# Channel-chunked variants (BHL only)
# =============================================================================


_DEFAULT_MIXED_CHUNK_SIZE = 128


def _chunked_along_channels(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None,
    chunk_size: int,
    core_fn,
) -> torch.Tensor:
    """Apply ``core_fn`` to ``[x, kernel, shortcut]`` slices of size ``chunk_size`` along H."""
    H = x.shape[1]
    if H <= chunk_size:
        return core_fn(x, kernel, shortcut)
    out_chunks: list[torch.Tensor] = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        out_chunks.append(
            core_fn(
                x[:, start:end],
                kernel[:, start:end],
                shortcut[start:end] if shortcut is not None else None,
            )
        )
    return torch.cat(out_chunks, dim=1)


def mixed_fftconv1d_fp32_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Memory-efficient 1D mixed-BC FFT conv (BHL) via channel chunking.

    See :func:`mixed_fftconv1d_fp32_bhl` for the core semantics. The work is
    identical, but it is performed on at most ``chunk_size`` channels at a
    time, lowering peak FFT-intermediate memory.
    """
    chunk = chunk_size if chunk_size is not None else _DEFAULT_MIXED_CHUNK_SIZE
    periodic_t = _normalize_periodic(periodic, data_dim=1)
    return _chunked_along_channels(
        x,
        kernel,
        shortcut,
        chunk,
        lambda xc, kc, sc: _mixed_fftconv_nd_fp32_bhl(xc, kc, periodic_t, sc, use_phase_shift, data_dim=1),
    )


def mixed_fftconv2d_fp32_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Memory-efficient 2D mixed-BC FFT conv (BHL) via channel chunking.

    See :func:`mixed_fftconv2d_fp32_bhl` for semantics.
    """
    chunk = chunk_size if chunk_size is not None else _DEFAULT_MIXED_CHUNK_SIZE
    periodic_t = _normalize_periodic(periodic, data_dim=2)
    return _chunked_along_channels(
        x,
        kernel,
        shortcut,
        chunk,
        lambda xc, kc, sc: _mixed_fftconv_nd_fp32_bhl(xc, kc, periodic_t, sc, use_phase_shift, data_dim=2),
    )


def mixed_fftconv3d_fp32_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Memory-efficient 3D mixed-BC FFT conv (BHL) via channel chunking.

    See :func:`mixed_fftconv3d_fp32_bhl` for semantics.
    """
    chunk = chunk_size if chunk_size is not None else _DEFAULT_MIXED_CHUNK_SIZE
    periodic_t = _normalize_periodic(periodic, data_dim=3)
    return _chunked_along_channels(
        x,
        kernel,
        shortcut,
        chunk,
        lambda xc, kc, sc: _mixed_fftconv_nd_fp32_bhl(xc, kc, periodic_t, sc, use_phase_shift, data_dim=3),
    )


# =============================================================================
# BLH wrappers for the channel-chunked variants
# =============================================================================


def mixed_fftconv1d_fp32_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Chunked 1D mixed-BC FFT conv wrapper for BLH layout.

    Reshapes BLH ↔ BHL around :func:`mixed_fftconv1d_fp32_bhl_chunked`.
    """
    x_bhl = rearrange(x, "b l h -> b h l")
    k_bhl = rearrange(kernel, "b k h -> b h k")
    y_bhl = mixed_fftconv1d_fp32_bhl_chunked(
        x_bhl,
        k_bhl,
        periodic,
        shortcut,
        use_phase_shift=use_phase_shift,
        chunk_size=chunk_size,
    )
    return rearrange(y_bhl, "b h l -> b l h")


def mixed_fftconv2d_fp32_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Chunked 2D mixed-BC FFT conv wrapper for BLH layout.

    Reshapes BLH ↔ BHL around :func:`mixed_fftconv2d_fp32_bhl_chunked`.
    """
    x_bhl = rearrange(x, "b x y h -> b h x y")
    k_bhl = rearrange(kernel, "b kx ky h -> b h kx ky")
    y_bhl = mixed_fftconv2d_fp32_bhl_chunked(
        x_bhl,
        k_bhl,
        periodic,
        shortcut,
        use_phase_shift=use_phase_shift,
        chunk_size=chunk_size,
    )
    return rearrange(y_bhl, "b h x y -> b x y h")


def mixed_fftconv3d_fp32_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    periodic: Sequence[bool],
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Chunked 3D mixed-BC FFT conv wrapper for BLH layout.

    Reshapes BLH ↔ BHL around :func:`mixed_fftconv3d_fp32_bhl_chunked`.
    """
    x_bhl = rearrange(x, "b x y z h -> b h x y z")
    k_bhl = rearrange(kernel, "b kx ky kz h -> b h kx ky kz")
    y_bhl = mixed_fftconv3d_fp32_bhl_chunked(
        x_bhl,
        k_bhl,
        periodic,
        shortcut,
        use_phase_shift=use_phase_shift,
        chunk_size=chunk_size,
    )
    return rearrange(y_bhl, "b h x y z -> b x y z h")
