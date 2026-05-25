# TODO: Add license header here


r"""Mixed boundary-condition FFT-based convolution operators (fp32).

What mixed boundary conditions mean
------------------------------------
Many PDE datasets have boundaries that are **periodic on some spatial axes
and non-periodic (wall or open) on others**. For example, a Rayleigh-Bénard
simulation is typically periodic in the horizontal (x) direction and bounded
by no-slip walls in the vertical (y) direction. The standard FFT convolution
operators in :mod:`nvsubquadratic.ops.fftconv` (linear / zero-padded) and
:mod:`nvsubquadratic.ops.circular_fftconv` (circular / periodic) each apply a
**global** boundary mode that is wrong for at least one axis in these mixed
cases.

This module is the fix: it lets each spatial axis independently use either

- **Periodic** (circular) boundary conditions — no padding on that axis, the
  convolution wraps around, and the "same"-alignment is achieved by a
  frequency-domain phase ramp ``exp(-i 2π f_d s_d)`` with integer shift
  ``s_d = -((K_d - 1) // 2)``.
- **Non-periodic** (zero-padded "same") boundary conditions — the FFT length
  is padded so that wrap-around cancels out, and the output is aligned by a
  centered crop rather than a phase ramp.

The choice is expressed per axis via ``periodic: Sequence[bool]`` of length
equal to the number of spatial dimensions:

.. code-block:: python

    # 2-D: x-axis periodic, y-axis zero-padded
    y = mixed_fftconv2d_fp32_bhl(x, kernel, periodic=(True, False))

    # 3-D: x and y periodic, z zero-padded  (e.g. turbulent_radiative_layer_3D)
    y = mixed_fftconv3d_fp32_bhl(x, kernel, periodic=(True, True, False))

When to use this vs. ``fftconv.py`` or ``circular_fftconv.py``
--------------------------------------------------------------
Use this module when **different spatial axes require different boundary
treatments**. For the all-same cases, prefer the dedicated modules:

- All axes non-periodic → :mod:`nvsubquadratic.ops.fftconv`.
- All axes periodic     → :mod:`nvsubquadratic.ops.circular_fftconv`.

Both degenerate cases are **automatically routed** through the legacy ops
at runtime (zero overhead), so it is safe to use this module as a single
entry point even when ``periodic`` happens to be uniform.

See ``docs/ops/MIXED_BC_PLAN.md`` for the per-axis algorithm, dataset
motivation table, and deferred work (fp16, multi-head, per-face BCs).

Algorithm overview
------------------
The N-D mixed FFT convolution is computed in **one** ``rfftn`` / ``irfftn``
call. The per-axis variation is encoded entirely in the arguments:

1. ``s=fft_shape`` — per-axis FFT lengths: ``N_d`` (periodic, no padding) or
   ``min(N_d + (K_d+1)//2, 2*N_d)`` (non-periodic, zero-pad headroom).
2. Post-IFFT crop — per-axis slice: ``[0, N_d)`` (periodic) or
   ``[K_d//2, K_d//2 + N_d)`` (non-periodic centered crop).
3. Phase ramp — applied to ``fft_k`` before the frequency-domain product:
   ``exp(-i 2π f_d s_d)`` on periodic axes; no ramp (``s_d = 0``) on
   non-periodic axes (alignment is handled by the crop).

Layouts and shapes
------------------
- **BHL** (channels-first, the fast path): ``[B, H, *spatial_dims]``;
  kernel ``[1|B, H, *K_dims]``; output ``[B, H, *spatial_dims]``.
- **BLH** wrappers (``*_w_reshape``) transparently reshape BLH → BHL → BLH.

The leading kernel dimension may be ``1`` (kernel shared across the batch)
or ``B`` (per-sample kernel, e.g. FiLM-conditioned Hyena).

Shortcut
--------
Optional ``shortcut: [H]`` adds a per-channel residual scale of the input
to the convolution output:

.. math::
    y \leftarrow y + \text{shortcut} \odot x

This is not a generic skip connection — it fuses a specific algebraic
shortcut from Hyena-style gating to avoid a separate kernel launch.

Phase ramps and the ``use_phase_shift`` flag
--------------------------------------------
On each periodic axis, "same" alignment requires shifting the output by
``s_d = -((K_d - 1) // 2)`` samples. This can be done two ways:

- ``use_phase_shift=True`` (default): multiply ``fft_k`` by the complex
  ramp ``exp(-i 2π f_d s_d)`` before the IFFT. One fused frequency-domain
  op, no data movement after the IFFT.
- ``use_phase_shift=False``: apply :func:`torch.roll` along the periodic
  axes *after* the IFFT. Mathematically identical; useful as a reference
  or when torch.compile cannot handle complex ops.

Caching
-------
Per-axis 1-D phase ramps are cached in a module-level LRU
(``_MIXED_PHASE_RAMP_1D_CACHE``). The N-D ramp is constructed on demand by
broadcasted multiplication of the relevant 1-D ramps — no N-D tensor is
stored in the cache. Axes with shift 0 contribute nothing (skipped).
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
    r"""LRU cache of 1-D frequency-domain phase ramps used by mixed FFT conv.

    On each **periodic** axis we need to shift the convolution output by
    ``s_d = -((K_d - 1) // 2)`` samples to obtain "same"-aligned output.
    In the frequency domain this shift corresponds to multiplying ``fft_k``
    by the complex exponential

    .. math::
        R_d[f] = \exp\!\left(-i\, 2\pi \frac{f}{F_d}\, s_d\right)

    where ``f`` ranges over the DFT frequencies for an FFT of length ``F_d``
    and ``s_d`` is the integer pixel shift. Non-periodic axes use ``s_d = 0``
    and contribute no ramp.

    This cache stores the 1-D ramps keyed by ``(F, s, is_rfft_axis, device,
    dtype)`` in an ordered-dict LRU so that repeated forward passes with the
    same spatial shape reuse the cached tensor without recomputation.

    The N-D ramp is **not** cached here — it is assembled by broadcasting the
    relevant 1-D ramps in :func:`_build_nd_phase_ramp` so that each 1-D entry
    is shared across all spatial configurations that share an axis.

    Attributes:
        maxsize: Maximum number of entries before the LRU evicts the oldest.
        _cache: Ordered dict mapping cache keys to complex ramp tensors.
    """

    def __init__(self, maxsize: int = 256):
        """Initialise the cache.

        Args:
            maxsize: Maximum number of (F, s, axis, device, dtype) entries to
                keep. Once exceeded, the least-recently-used entry is evicted.
                Default 256 covers many typical spatial / kernel size combos
                without meaningful memory cost (each 1-D ramp is small).
        """
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
        """Build a hashable cache key for a 1-D phase ramp.

        The key encodes every parameter that affects the ramp tensor so that
        ramps from different devices or dtypes are never confused.

        Args:
            F: FFT length along this axis.
            s: Integer pixel shift encoded by the ramp.
            is_rfft_axis: True when this is the last axis (rfft frequencies,
                length ``F // 2 + 1``); False for intermediate axes (full DFT
                frequencies, length ``F``).
            device: Target device.
            real_dtype: Real-valued input dtype (``float32`` or ``float64``).

        Returns:
            A hashable tuple suitable as a dict key.
        """
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
        r"""Return the 1-D phase ramp for the given (FFT length, shift) on this axis.

        Computes (or retrieves from the LRU cache) the complex tensor

        .. math::
            R[f] = \cos(-2\pi f s) + i\,\sin(-2\pi f s)

        where ``f`` iterates over ``torch.fft.rfftfreq(F)`` (if
        ``is_rfft_axis``) or ``torch.fft.fftfreq(F)`` (otherwise). The ramp
        is computed under ``torch.no_grad()`` and ``torch.inference_mode``
        so it does not pollute the autograd graph.

        Args:
            F: FFT length along this axis (padded length for non-periodic,
                input length for periodic — caller decides which).
            s: Integer pixel shift to apply via the ramp. Callers with
                ``s == 0`` are expected to skip the multiply entirely; this
                method still handles ``s == 0`` (returns all-ones).
            is_rfft_axis: If True, this is the last spatial axis where the
                rfft is taken; a ramp of length ``F // 2 + 1`` is built
                using :func:`torch.fft.rfftfreq`. Otherwise a full ramp of
                length ``F`` is built using :func:`torch.fft.fftfreq`.
            device: Target device for the ramp tensor.
            real_dtype: Real-valued dtype of the inputs (``float32`` or
                ``float64``); the ramp is stored in the corresponding complex
                dtype (``complex64`` or ``complex128``).

        Returns:
            1-D complex tensor of shape ``[F // 2 + 1]`` (rfft axis) or
            ``[F]`` (non-rfft axis) on ``device``.
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
    r"""Build (or fetch from cache) the broadcast N-D phase-ramp tensor.

    Assembles the N-D phase ramp

    .. math::
        R[f_0, \ldots, f_{D-1}] =
            \prod_{d:\, s_d \ne 0}
            \exp\!\left(-i\, 2\pi \frac{f_d}{F_d}\, s_d\right)

    by broadcasting individual 1-D ramps fetched from
    ``_MIXED_PHASE_RAMP_1D_CACHE``. The result has rfft-style shape
    ``(F_0, F_1, ..., F_{D-2}, F_{D-1} // 2 + 1)`` and complex dtype,
    so it can be multiplied directly with
    ``torch.fft.rfftn(x, s=fft_shape)``.

    Axes with ``shift == 0`` (non-periodic axes, and periodic axes where the
    kernel has size 1) contribute no ramp — they are skipped entirely rather
    than adding a length-1 all-ones factor. If **all** shifts are zero the
    function returns ``None`` so callers can skip the multiply with a single
    branch.

    The N-D ramp is **not** cached; only the 1-D per-axis ramps are.  The
    product is materialised here (via broadcasted multiplication) so that the
    downstream multiply with ``fft_x`` is a single fused op.

    Args:
        fft_shape: Per-axis FFT lengths ``(F_0, ..., F_{D-1})``, as returned
            by :func:`_mixed_recipe`.
        shifts: Per-axis integer pixel shifts ``(s_0, ..., s_{D-1})``.
            Negative values shift left (the typical case for periodic axes).
        device: Device on which to materialise the ramp.
        real_dtype: Real dtype of the input (determines the complex dtype of
            the ramp: ``float32`` → ``complex64``, ``float64`` → ``complex128``).

    Returns:
        Broadcast-ready complex tensor of shape
        ``(F_0, ..., F_{D-2}, F_{D-1} // 2 + 1)``, or ``None`` if every
        shift is zero.
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
    """Normalise the ``periodic`` argument to a length-``data_dim`` tuple of bools.

    Accepts any sequence (list, tuple, generator) of truthy/falsy values and
    returns a typed tuple of Python ``bool`` values, validating that the
    length matches ``data_dim``.

    Args:
        periodic: Per-axis periodicity flags. Length must equal ``data_dim``.
            Any truthy value is treated as ``True`` (periodic).
        data_dim: Expected number of spatial dimensions (1, 2, or 3).

    Returns:
        A ``tuple[bool, ...]`` of length ``data_dim``.

    Raises:
        AssertionError: If ``len(periodic) != data_dim``.
    """
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

    This ensures the all-False and all-True degenerate cases produce
    bit-identical results to :mod:`nvsubquadratic.ops.fftconv` and
    :mod:`nvsubquadratic.ops.circular_fftconv` respectively, and incur no
    overhead from the mixed-axis logic.

    - ``all(periodic) == True`` → dispatches to the corresponding
      ``circular_fftconv{1,2,3}d_fp32_bhl`` function, forwarding
      ``use_phase_shift``.
    - ``not any(periodic)`` (all False) → dispatches to the corresponding
      ``fftconv{1,2,3}d_fp32_bhl`` function. The ``use_phase_shift`` flag
      is not forwarded (it does not apply to zero-padded linear conv).

    Args:
        periodic: Normalised per-axis periodicity tuple (length ``data_dim``).
        data_dim: Number of spatial dimensions (1, 2, or 3).
        x: Input tensor (BHL layout).
        kernel: Kernel tensor (BHL layout).
        shortcut: Optional per-channel residual scale ``[H]``.
        use_phase_shift: Forwarded to the circular op when dispatching.

    Returns:
        Output tensor if the call was dispatched to a legacy op, or ``None``
        if ``periodic`` is mixed and the caller must run the mixed path.
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
    Implements the following steps:

    1. Validate shapes (ndim, batch dim, hidden dim, per-axis kernel-size
       limits).
    2. Dispatch to the legacy linear / circular op if ``periodic`` is uniform
       (via :func:`_dispatch_legacy_if_uniform`).
    3. Cast ``x`` and ``kernel`` to fp32, compute the per-axis FFT parameters
       via :func:`_mixed_recipe`, and take ``rfftn`` over all spatial dims in
       one call.
    4. Optionally apply the N-D phase ramp to ``fft_k`` (if
       ``use_phase_shift=True``).
    5. Multiply ``fft_x * fft_k`` and apply the inverse ``irfftn``.
    6. If ``use_phase_shift=False``, apply ``torch.roll`` on the periodic
       axes to achieve "same" alignment.
    7. Crop the IFFT output to the input spatial shape via the per-axis
       crop windows from :func:`_mixed_recipe`.
    8. Cast back to the original dtype of ``x`` and add the optional
       shortcut term ``y += shortcut * x``.

    Args:
        x: Input tensor of shape ``[B, H, *spatial_dims]`` (any dtype).
        kernel: Kernel tensor of shape ``[1|B, H, *K_dims]`` (any dtype).
        periodic: Normalised per-axis periodicity tuple, length ``data_dim``.
        shortcut: Optional per-channel residual scale ``[H]``. Must have the
            same dtype as ``x``.
        use_phase_shift: If True, apply the "same" shift in the frequency
            domain (faster). If False, use ``torch.roll`` after the IFFT
            (useful as a reference implementation).
        data_dim: Number of spatial dimensions (1, 2, or 3).

    Returns:
        Output tensor of shape ``[B, H, *spatial_dims]`` in the original
        dtype of ``x``.

    Raises:
        AssertionError: On shape mismatches or out-of-range kernel sizes.
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

    Reshapes BLH ↔ BHL around :func:`mixed_fftconv1d_fp32_bhl`. Prefer this
    wrapper over operating in BLH natively — internally the FFT runs on
    contiguous spatial axes (BHL), so the reshape cost is negligible compared
    to the FFT itself.

    Args:
        x: Input tensor of shape ``[B, L, H]`` (BLH, channels-last).
        kernel: Kernel tensor of shape ``[B, K, H]`` (BLH).
        periodic: Length-1 sequence of bools.
        shortcut: Optional per-channel scale ``[H]``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.

    Returns:
        Tensor of shape ``[B, L, H]`` in the original dtype of ``x``.
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

    Reshapes BLH ↔ BHL around :func:`mixed_fftconv2d_fp32_bhl`. Prefer this
    over operating in BLH natively — the FFT runs faster on contiguous spatial
    axes (BHL).

    Args:
        x: Input tensor of shape ``[B, X, Y, H]`` (BLH, channels-last).
        kernel: Kernel tensor of shape ``[B, K_x, K_y, H]`` (BLH).
        periodic: Length-2 sequence ``(periodic_x, periodic_y)``.
        shortcut: Optional per-channel scale ``[H]``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.

    Returns:
        Tensor of shape ``[B, X, Y, H]`` in the original dtype of ``x``.
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

    Reshapes BLH ↔ BHL around :func:`mixed_fftconv3d_fp32_bhl`. Prefer this
    over operating in BLH natively.

    Args:
        x: Input tensor of shape ``[B, X, Y, Z, H]`` (BLH, channels-last).
        kernel: Kernel tensor of shape ``[B, K_x, K_y, K_z, H]`` (BLH).
        periodic: Length-3 sequence ``(periodic_x, periodic_y, periodic_z)``.
        shortcut: Optional per-channel scale ``[H]``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.

    Returns:
        Tensor of shape ``[B, X, Y, Z, H]`` in the original dtype of ``x``.
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
    """Apply ``core_fn`` to ``[x, kernel, shortcut]`` slices of size ``chunk_size`` along H.

    Reduces peak GPU memory by avoiding materialising the full FFT-intermediate
    tensors (of size ``[B, H, *fft_shape]``) for all channels at once.  When
    ``H <= chunk_size`` the call is a pass-through with no slicing overhead.

    Args:
        x: Input tensor of shape ``[B, H, *spatial_dims]`` (BHL layout).
        kernel: Kernel tensor of shape ``[1|B, H, *K_dims]``. The leading dim
            is expected to be either 1 (shared kernel) or ``B``; slicing along
            dim 1 is applied uniformly regardless.
        shortcut: Optional per-channel residual scale ``[H]``, sliced to
            match each chunk. ``None`` is forwarded unchanged.
        chunk_size: Maximum number of channels to process per chunk.
        core_fn: Callable with signature ``(x_chunk, k_chunk, sc_chunk) ->
            Tensor`` that performs the actual convolution on a channel slice.

    Returns:
        Output tensor of shape ``[B, H, *spatial_dims]`` reassembled from
        chunk outputs concatenated along the channel dimension (dim 1).
    """
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

    Produces the same output as :func:`mixed_fftconv1d_fp32_bhl` but processes
    at most ``chunk_size`` channels at a time, lowering peak FFT-intermediate
    memory at the cost of multiple kernel launches.

    Args:
        x: Input tensor of shape ``[B, H, L]`` (any dtype, cast to fp32).
        kernel: Kernel tensor of shape ``[1|B, H, K]``.
        periodic: Length-1 sequence of bools. ``periodic[0] == True`` → circular.
        shortcut: Optional per-channel scale ``[H]`` added as ``y += shortcut * x``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.
        chunk_size: Number of channels per chunk. Defaults to
            ``_DEFAULT_MIXED_CHUNK_SIZE`` (128). Pass ``None`` to use the default.

    Returns:
        Tensor of shape ``[B, H, L]`` in the original dtype of ``x``.
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

    Produces the same output as :func:`mixed_fftconv2d_fp32_bhl` but processes
    at most ``chunk_size`` channels at a time to limit peak memory.

    Args:
        x: Input tensor of shape ``[B, H, X, Y]`` (any dtype, cast to fp32).
        kernel: Kernel tensor of shape ``[1|B, H, K_x, K_y]``.
        periodic: Length-2 sequence ``(periodic_x, periodic_y)``.
        shortcut: Optional per-channel scale ``[H]``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.
        chunk_size: Channels per chunk. Defaults to 128.

    Returns:
        Tensor of shape ``[B, H, X, Y]`` in the original dtype of ``x``.
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

    Produces the same output as :func:`mixed_fftconv3d_fp32_bhl` but processes
    at most ``chunk_size`` channels at a time to limit peak memory.

    Args:
        x: Input tensor of shape ``[B, H, X, Y, Z]`` (any dtype, cast to fp32).
        kernel: Kernel tensor of shape ``[1|B, H, K_x, K_y, K_z]``.
        periodic: Length-3 sequence ``(periodic_x, periodic_y, periodic_z)``.
        shortcut: Optional per-channel scale ``[H]``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.
        chunk_size: Channels per chunk. Defaults to 128.

    Returns:
        Tensor of shape ``[B, H, X, Y, Z]`` in the original dtype of ``x``.
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
    Combines channel chunking (for memory savings) with the BLH → BHL reshape
    (for FFT efficiency).

    Args:
        x: Input tensor of shape ``[B, L, H]`` (BLH, channels-last).
        kernel: Kernel tensor of shape ``[B, K, H]`` (BLH).
        periodic: Length-1 sequence of bools.
        shortcut: Optional per-channel scale ``[H]``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.
        chunk_size: Channels per chunk. Defaults to 128.

    Returns:
        Tensor of shape ``[B, L, H]`` in the original dtype of ``x``.
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

    Args:
        x: Input tensor of shape ``[B, X, Y, H]`` (BLH, channels-last).
        kernel: Kernel tensor of shape ``[B, K_x, K_y, H]`` (BLH).
        periodic: Length-2 sequence ``(periodic_x, periodic_y)``.
        shortcut: Optional per-channel scale ``[H]``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.
        chunk_size: Channels per chunk. Defaults to 128.

    Returns:
        Tensor of shape ``[B, X, Y, H]`` in the original dtype of ``x``.
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

    Args:
        x: Input tensor of shape ``[B, X, Y, Z, H]`` (BLH, channels-last).
        kernel: Kernel tensor of shape ``[B, K_x, K_y, K_z, H]`` (BLH).
        periodic: Length-3 sequence ``(periodic_x, periodic_y, periodic_z)``.
        shortcut: Optional per-channel scale ``[H]``.
        use_phase_shift: See :func:`mixed_fftconv1d_fp32_bhl`.
        chunk_size: Channels per chunk. Defaults to 128.

    Returns:
        Tensor of shape ``[B, X, Y, Z, H]`` in the original dtype of ``x``.
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
