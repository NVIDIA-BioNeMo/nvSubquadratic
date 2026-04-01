"""FP16 circular (periodic) FFT-based convolution for 1D, 2D, and 3D signals.

Leverages PyTorch's native fp16 FFT support (cuFFT C2C half-precision).
Two constraints must be satisfied:

1. cuFFT requires **power-of-2 sizes** for half-precision transforms.
   Since circular convolution uses same-size FFTs (FFT length == input length),
   the *input* spatial dimensions must themselves be powers of 2.
2. The frequency-domain products can exceed fp16's 65504 limit with real model
   weights, so we use ``norm="ortho"`` which divides both the forward and
   inverse FFT by ``sqrt(N)``, keeping intermediates in range.  The ortho pair
   computes ``circular_conv / sqrt(N)`` instead of ``circular_conv``, so we
   multiply by ``sqrt(N)`` after the inverse FFT to restore the correct scale.

Numerical stability (dual mean-centering)
------------------------------------------
Ortho normalization alone is insufficient: the DC bin and internal FFT
accumulations can still overflow in fp16 for signals with nonzero mean.
To fix this, both the input ``x`` and kernel ``k`` are mean-centered before
the forward FFT, which zeros the DC bins and reduces element magnitudes to
``O(std)``.  The exact convolution result is recovered analytically via:

- **DC correction** (T4): ``mu_x * mu_k * K`` added after the inverse FFT.
- **Centering correction** (T2, nD): an inclusion-exclusion geometric factor
  ``geo`` (cached per kernel/spatial shape) that accounts for zero-padded
  kernel positions.  Only the scalar ``k_mean / sqrt_N`` changes per call;
  ``geo`` is recomputed only when kernel or spatial dimensions change.

See ``circular_fftconv1d_fp16_bhl`` docstring for the full derivation.

Alignment
---------
Alignment is done via a frequency-domain phase ramp, the same approach used by
the fp32 variants.  The ramp is computed in float32 for precision and cast to
complex32 before the multiply so all FFT intermediates stay in half-precision.

Families provided
-----------------
- 1D: ``circular_fftconv1d_fp16_bhl`` (+``_w_reshape``)
- 2D: ``circular_fftconv2d_fp16_bhl`` (+``_w_reshape``)
- 3D: ``circular_fftconv3d_fp16_bhl`` (+``_w_reshape``)
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from einops import rearrange

from nvsubquadratic.ops.circular_fftconv import (
    _phase_ramp_cache_1d,
    _phase_ramp_cache_2d,
    _phase_ramp_cache_3d,
)


__all__ = [
    "circular_fftconv1d_fp16_bhl",
    "circular_fftconv1d_fp16_bhl_w_reshape",
    "circular_fftconv2d_fp16_bhl",
    "circular_fftconv2d_fp16_bhl_w_reshape",
    "circular_fftconv3d_fp16_bhl",
    "circular_fftconv3d_fp16_bhl_w_reshape",
]


def _is_power_of_2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# Phase ramps are computed in float32 (full-precision trig) and cast to
# complex32 so the multiply with fp16 FFT data stays in half-precision.
_PHASE_RAMP_COMPUTE_DTYPE = torch.float32
_PHASE_RAMP_TARGET_DTYPE = torch.complex32


###############################################################################
# Helpers: centering correction geometry caches (2D / 3D)
###############################################################################


def _phase_m1(N: int, rfft: bool, device: torch.device) -> torch.Tensor:
    r"""Phase factor for a unit impulse at position :math:`N-1` (i.e. :math:`-1 \bmod N`).

    Returns the 1-D DFT :math:`p[f] = e^{2\pi i f / N}` evaluated on the
    appropriate frequency grid.  This is used to build the inclusion-exclusion
    geometric correction in the nD centering formulas.

    Args:
        N: Transform length along this axis.
        rfft: If True, use ``rfftfreq`` (length ``N//2+1``, for the last axis
            of ``rfftn``); otherwise use ``fftfreq`` (length ``N``).
        device: Torch device for the output tensor.

    Returns:
        Complex64 tensor of shape ``[N//2+1]`` (rfft) or ``[N]`` (fft).
    """
    f = (torch.fft.rfftfreq if rfft else torch.fft.fftfreq)(
        N,
        device=device,
        dtype=torch.float32,
    )
    a = 2.0 * math.pi * f
    return torch.complex(a.cos(), a.sin())


class _CenteringCorrectionCache2D:
    r"""LRU cache for the 2D geometric correction factor ``geo``.

    When the kernel is smaller than the spatial dimensions along one or
    both axes (:math:`K_d < N_d`), the mean-centering decomposition
    produces a T2 correction term that depends on an inclusion-exclusion
    geometric factor.  This factor depends only on the kernel/spatial
    shape and device, so it is computed once and reused.

    The full frequency-domain correction added to :math:`\hat{k}_c` is:

    .. math::
        \frac{\mu_k}{\sqrt{N}} \cdot \text{geo}

    Only the scalar :math:`\mu_k` changes per forward call.
    """

    def __init__(self, maxsize: int = 32):
        self.maxsize = maxsize
        self._cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    @staticmethod
    def _key(K_x, K_y, X, Y, device):
        return (K_x, K_y, X, Y, device.type, device.index if device.index is not None else -1)

    def get(self, K_x: int, K_y: int, X: int, Y: int, device: torch.device) -> torch.Tensor | None:
        r"""Return cached ``geo`` of shape ``[X, Y//2+1]`` (complex64), or None.

        Returns ``None`` when no correction is needed (K_x == X and K_y == Y).
        Otherwise computes the inclusion-exclusion geometric factor over the
        corrected axes (those where ``K_d < N_d``).
        """
        cx, cy = int(K_x < X), int(K_y < Y)
        if not (cx or cy):
            return None

        key = self._key(K_x, K_y, X, Y, device)
        geo = self._cache.get(key)
        if geo is not None:
            self._cache.move_to_end(key)
            return geo

        Yf = Y // 2 + 1
        p_x = _phase_m1(X, rfft=False, device=device) if cx else None
        p_y = _phase_m1(Y, rfft=True, device=device) if cy else None

        geo = torch.zeros(X, Yf, device=device, dtype=torch.complex64)
        if cx:
            geo[:, 0] -= Y * p_x
        if cy:
            geo[0, :] -= X * p_y
        if cx and cy:
            geo += p_x[:, None] * p_y[None, :]

        self._cache[key] = geo
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return geo


class _CenteringCorrectionCache3D:
    r"""LRU cache for the 3D geometric correction factor ``geo``.

    Same structure as :class:`_CenteringCorrectionCache2D` but with
    three-axis inclusion-exclusion (signs: single terms ``-1``,
    pair terms ``+1``, triple term ``-1``).
    """

    def __init__(self, maxsize: int = 32):
        self.maxsize = maxsize
        self._cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    @staticmethod
    def _key(Kx, Ky, Kz, X, Y, Z, device):
        return (Kx, Ky, Kz, X, Y, Z, device.type, device.index if device.index is not None else -1)

    def get(self, Kx: int, Ky: int, Kz: int, X: int, Y: int, Z: int, device: torch.device) -> torch.Tensor | None:
        """Return cached ``geo`` of shape ``[X, Y, Z//2+1]`` (complex64), or None."""
        cx, cy, cz = int(Kx < X), int(Ky < Y), int(Kz < Z)
        if not (cx or cy or cz):
            return None

        key = self._key(Kx, Ky, Kz, X, Y, Z, device)
        geo = self._cache.get(key)
        if geo is not None:
            self._cache.move_to_end(key)
            return geo

        Zf = Z // 2 + 1
        p_x = _phase_m1(X, rfft=False, device=device) if cx else None
        p_y = _phase_m1(Y, rfft=False, device=device) if cy else None
        p_z = _phase_m1(Z, rfft=True, device=device) if cz else None

        geo = torch.zeros(X, Y, Zf, device=device, dtype=torch.complex64)

        # Single-axis terms (sign: -1, coeff: product of OTHER axis sizes)
        if cx:
            geo[:, 0, 0] -= (Y * Z) * p_x
        if cy:
            geo[0, :, 0] -= (X * Z) * p_y
        if cz:
            geo[0, 0, :] -= (X * Y) * p_z

        # Pair terms (sign: +1, coeff: remaining axis size)
        if cx and cy:
            geo[:, :, 0] += Z * (p_x[:, None] * p_y[None, :])
        if cx and cz:
            geo[:, 0, :] += Y * (p_x[:, None] * p_z[None, :])
        if cy and cz:
            geo[0, :, :] += X * (p_y[:, None] * p_z[None, :])

        # Triple term (sign: -1)
        if cx and cy and cz:
            geo -= p_x[:, None, None] * p_y[None, :, None] * p_z[None, None, :]

        self._cache[key] = geo
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return geo


_centering_geo_cache_2d = _CenteringCorrectionCache2D()
_centering_geo_cache_3d = _CenteringCorrectionCache3D()


###############################################################################
# 1D
###############################################################################


def circular_fftconv1d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""1D circular FFT convolution in fp16 with dual mean-centering.

    Numerically stable drop-in replacement for ``circular_fftconv1d_fp32_bhl``.
    Casts *x* and *kernel* to fp16 internally; shortcut is never cast.
    Returns in the **original** dtype of *x*.

    Requires ``L`` to be a power of 2 (cuFFT fp16 constraint).

    Root cause of NaN
    -----------------
    With ``norm="ortho"``, the FFT scales by ``1/sqrt(N)``.  Two overflow
    sources exist in fp16 (max representable value 65504):

    1. **DC-bin product overflow**: the DC bin of ``rfft(x, ortho)`` is
       ``sum(x)/sqrt(N) = mean(x)*sqrt(N)``.  The product of two DC bins
       is ``mean(x)*mean(k)*N``, which exceeds 65504 for moderate means.
    2. **Internal FFT accumulation overflow**: cuFFT accumulates partial
       sums in fp16 before dividing by ``sqrt(N)``.  For a signal with
       mean ``mu`` and ``N`` elements, internal sums reach ``~mu*N``,
       which overflows for ``mu*N > 65504``.

    Fix: dual mean-centering
    ------------------------
    Subtract the spatial mean from both ``x`` and ``k`` before the FFT.
    This zeros both DC bins (fixing #1) and reduces element magnitudes to
    ``O(std)`` (fixing #2).  The exact result is recovered analytically.

    Decomposition (circular conv of length ``L``, kernel zero-padded to ``L``)::

        y[n] = sum_m x[n-m] * k_padded[m]

    Let ``x = x_c + mu_x``, ``k_padded = k_c_padded + delta``, where
    ``delta = [mu_k, ..., mu_k, 0, ..., 0]`` (``K`` copies of ``mu_k``,
    then ``L-K`` zeros).  Expanding gives four terms::

        T1 = circ_conv(x_c, k_c_padded)    [both DC=0, safe in fp16]
        T2 = mu_k * circ_conv(x_c, delta)  [correction for centering k]
        T3 = mu_x * sum(k_c_padded) = 0    [k_c sums to zero]
        T4 = mu_x * k_sum                  [correction for centering x]

    K=L-1 correction (T2)
    ~~~~~~~~~~~~~~~~~~~~~~
    ``delta`` is all-``mu_k`` except one zero at position ``L-1``.
    ``circ_conv(x_c, delta)[n] = mu_k * sum_{m != L-1} x_c[n-m]``.
    Since ``sum(x_c) = 0``, this equals ``-mu_k * x_c[(n+1) mod L]``,
    i.e., a roll of ``x_c`` by ``-1`` scaled by ``-mu_k``.

    When a kernel-centering shift ``s`` is applied, the zero in ``delta``
    moves, and the identity becomes ``-mu_k * roll(x_c, s-1)``, or
    equivalently ``-mu_k * fft_x * phase(s-1)`` in frequency domain.

    Phase-ramp absorption
    ~~~~~~~~~~~~~~~~~~~~~
    Factor ``phase(s-1) = phase(s) * phase(-1)`` so the full effective
    kernel spectrum is::

        fft_k_eff = phase(s) * [fft_k_c - (mu_k / sqrt_N) * phase(-1)]

    The ``1/sqrt_N`` keeps frequency values small (preventing irfft
    internal overflow); ``sqrt_N`` is applied after irfft in float32.

    K=L case
    ~~~~~~~~
    No zero-padding => ``delta`` covers all ``L`` positions =>
    ``circ_conv(x_c, delta) = mu_k * sum(x_c) = 0``.  No correction.

    Args:
        x: Input tensor ``[B, H, L]`` (any dtype).
        kernel: Kernel tensor ``[1|B, H, K]`` with ``K in {L, L-1}`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (same dtype as *x*).

    Returns:
        Tensor ``[B, H, L]`` in the original dtype of *x*.
    """
    B, H, L = x.shape
    assert _is_power_of_2(L), f"L must be a power of 2 for fp16 circular FFT. Got L={L}."
    assert len(kernel.shape) == 3, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )
    _, Hk, K = kernel.shape
    assert H == Hk, "Input and kernel must have the same number of channels (H)."
    assert K <= L, f"K must be <= L. Got K={K}, L={L}."

    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)
    sqrt_N = math.sqrt(L)

    # ── Center both x and k in-place (zeros DC bins, no extra allocation) ──
    x_mean = x_fp16.mean(dim=-1, keepdim=True)  # [B, H, 1]  fp16
    x_fp16.sub_(x_mean)  # x_fp16 is now x_c

    k_mean = k_fp16.mean(dim=-1, keepdim=True)  # [1|B, H, 1]  fp16
    k_fp16.sub_(k_mean)  # k_fp16 is now k_c

    # DC correction: mu_x * mu_k * K, precomputed in fp32
    dc_corr = x_mean.float() * (k_mean.float() * K)  # [B, H, 1]  fp32

    # ── Forward FFTs in fp16 (both DC bins = 0) ──
    fft_x = torch.fft.rfft(x_fp16, n=L, dim=2, norm="ortho")
    fft_k = torch.fft.rfft(k_fp16, n=L, dim=2, norm="ortho")
    del k_fp16

    # ── Build effective kernel spectrum ──
    # fft_k is small (no batch dim), so we work in complex64 for precision
    # and cast back to complex32 before the large multiply with fft_x.
    shift = -((K - 1) // 2)

    del x_fp16  # x_c no longer needed (correction folded into fft_k)

    fft_k_eff = fft_k.to(torch.complex64)
    del fft_k

    # Build effective kernel spectrum in one expression per case:
    #   K=L  :  fft_k_eff = fft_k_c * phase(s)
    #   K=L-1:  fft_k_eff = fft_k_c * phase(s) - (mu_k / sqrt_N) * phase(s-1)
    # The K=L-1 form comes from expanding:
    #   phase(s) * [fft_k_c - (mu_k/sqrt_N) * phase(-1)]
    if K == L - 1:
        phase_s = _phase_ramp_cache_1d.get(L, shift, x.device, _PHASE_RAMP_COMPUTE_DTYPE)
        phase_sm1 = _phase_ramp_cache_1d.get(L, shift - 1, x.device, _PHASE_RAMP_COMPUTE_DTYPE)
        fft_k_eff = fft_k_eff * phase_s - (k_mean.float() / sqrt_N) * phase_sm1
    elif K == L:
        if shift != 0:
            phase_s = _phase_ramp_cache_1d.get(L, shift, x.device, _PHASE_RAMP_COMPUTE_DTYPE)
            fft_k_eff = fft_k_eff * phase_s
    else:
        raise ValueError(f"Dual-centering correction requires K=L or K=L-1, got K={K}, L={L}")

    fft_x.mul_(fft_k_eff.to(torch.complex32))

    # ── Inverse FFT (fp16) ──
    y = torch.fft.irfft(fft_x, n=L, dim=2, norm="ortho")

    # ── Undo ortho scaling + DC correction in one expression (float32) ──
    y = (y.float() * sqrt_N + dc_corr).to(x.dtype)

    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (H,)
        y = y + rearrange(shortcut, "h -> 1 h 1") * x
    return y


def circular_fftconv1d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D circular FFT convolution in fp16, for inputs with layout ``[B, L, H]``.

    Wrapper around :func:`circular_fftconv1d_fp16_bhl` that handles the
    BLH <-> BHL reshape.
    """
    x_bhl = rearrange(x, "b l h -> b h l")
    k_bhl = rearrange(kernel, "b k h -> b h k")
    y = circular_fftconv1d_fp16_bhl(x_bhl, k_bhl, shortcut)
    return rearrange(y, "b h l -> b l h")


###############################################################################
# 2D
###############################################################################


def circular_fftconv2d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""2D circular FFT convolution in fp16 with dual mean-centering.

    Numerically stable drop-in replacement for ``circular_fftconv2d_fp32_bhl``.
    See :func:`circular_fftconv1d_fp16_bhl` for the derivation;  the 2D
    centering correction uses a cached inclusion-exclusion geometric factor
    ``geo`` (see :class:`_CenteringCorrectionCache2D`).

    Requires ``X`` and ``Y`` to be powers of 2 (cuFFT fp16 constraint).

    Args:
        x: Input tensor ``[B, H, X, Y]`` (any dtype).
        kernel: Kernel tensor ``[1|B, H, K_x, K_y]`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (same dtype as *x*).

    Returns:
        Tensor ``[B, H, X, Y]`` in the original dtype of *x*.
    """
    B, H, X, Y = x.shape
    assert _is_power_of_2(X) and _is_power_of_2(Y), (
        f"Spatial dims must be powers of 2 for fp16 circular FFT. Got X={X}, Y={Y}."
    )
    assert len(kernel.shape) == 4, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )
    _, Hk, K_x, K_y = kernel.shape
    assert H == Hk, "Input and kernel must have the same number of channels (H)."
    assert K_x <= X, f"K_x must be <= X. Got K_x={K_x}, X={X}."
    assert K_y <= Y, f"K_y must be <= Y. Got K_y={K_y}, Y={Y}."

    N = X * Y
    sqrt_N = math.sqrt(N)

    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)

    # ── Center both x and k in-place ──
    x_mean = x_fp16.mean(dim=(-2, -1), keepdim=True)  # [B, H, 1, 1]
    x_fp16.sub_(x_mean)

    k_mean = k_fp16.mean(dim=(-2, -1), keepdim=True)  # [1|B, H, 1, 1]
    k_fp16.sub_(k_mean)

    dc_corr = x_mean.float() * (k_mean.float() * (K_x * K_y))

    # ── Forward FFTs (both DC bins = 0) ──
    fft_x = torch.fft.rfft2(x_fp16, s=(X, Y), dim=(2, 3), norm="ortho")
    fft_k = torch.fft.rfft2(k_fp16, s=(X, Y), dim=(2, 3), norm="ortho")
    del k_fp16

    shift_x = -((K_x - 1) // 2)
    shift_y = -((K_y - 1) // 2)

    del x_fp16  # x_c no longer needed (correction folded into fft_k)

    # Build effective kernel spectrum in complex64 (small tensor, no batch dim).
    fft_k_eff = fft_k.to(torch.complex64)
    del fft_k

    # T2: centering correction for zero-padded kernel positions.
    # The geometric factor is cached and only the scalar mu_k/sqrt_N changes.
    geo = _centering_geo_cache_2d.get(K_x, K_y, X, Y, x.device)
    if geo is not None:
        fft_k_eff = fft_k_eff + (k_mean.float() / sqrt_N) * geo

    # Apply phase ramp for kernel alignment (spatial centering).
    if shift_x != 0 or shift_y != 0:
        phase = _phase_ramp_cache_2d.get(
            X,
            Y,
            shift_x,
            shift_y,
            x.device,
            _PHASE_RAMP_COMPUTE_DTYPE,
        )
        fft_k_eff = fft_k_eff * phase

    # Multiply in complex32 (large batched tensor stays in half precision).
    fft_x.mul_(fft_k_eff.to(torch.complex32))

    # ── Inverse FFT (fp16) ──
    y = torch.fft.irfft2(fft_x, s=(X, Y), dim=(2, 3), norm="ortho")

    # ── Undo ortho scaling (×√N) and add DC correction (T4), in float32 ──
    y = (y.float() * sqrt_N + dc_corr).to(x.dtype)

    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (H,)
        y = y + rearrange(shortcut, "h -> 1 h 1 1") * x
    return y


def circular_fftconv2d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D circular FFT convolution in fp16, for inputs with layout ``[B, X, Y, H]``.

    Wrapper around :func:`circular_fftconv2d_fp16_bhl` that handles the
    BLH <-> BHL reshape.
    """
    x_bhl = rearrange(x, "b x y h -> b h x y")
    k_bhl = rearrange(kernel, "b kx ky h -> b h kx ky")
    y = circular_fftconv2d_fp16_bhl(x_bhl, k_bhl, shortcut)
    return rearrange(y, "b h x y -> b x y h")


###############################################################################
# 3D
###############################################################################


def circular_fftconv3d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""3D circular FFT convolution in fp16 with dual mean-centering.

    Numerically stable drop-in replacement for ``circular_fftconv3d_fp32_bhl``.
    See :func:`circular_fftconv1d_fp16_bhl` for the derivation;  the 3D
    centering correction uses a cached inclusion-exclusion geometric factor
    ``geo`` (see :class:`_CenteringCorrectionCache3D`).

    Requires ``X``, ``Y``, and ``Z`` to be powers of 2 (cuFFT fp16 constraint).

    Args:
        x: Input tensor ``[B, H, X, Y, Z]`` (any dtype).
        kernel: Kernel tensor ``[1|B, H, Kx, Ky, Kz]`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (same dtype as *x*).

    Returns:
        Tensor ``[B, H, X, Y, Z]`` in the original dtype of *x*.
    """
    B, H, X, Y, Z = x.shape
    assert _is_power_of_2(X) and _is_power_of_2(Y) and _is_power_of_2(Z), (
        f"Spatial dims must be powers of 2 for fp16 circular FFT. Got X={X}, Y={Y}, Z={Z}."
    )
    assert len(kernel.shape) == 5, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )
    _, Hk, Kx, Ky, Kz = kernel.shape
    assert H == Hk, "Input and kernel must have the same number of channels (H)."
    assert Kx <= X and Ky <= Y and Kz <= Z, "Kernel must be <= input along each axis."

    N = X * Y * Z
    sqrt_N = math.sqrt(N)

    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)

    # ── Center both x and k in-place ──
    x_mean = x_fp16.mean(dim=(-3, -2, -1), keepdim=True)  # [B, H, 1, 1, 1]
    x_fp16.sub_(x_mean)

    k_mean = k_fp16.mean(dim=(-3, -2, -1), keepdim=True)  # [1|B, H, 1, 1, 1]
    k_fp16.sub_(k_mean)

    dc_corr = x_mean.float() * (k_mean.float() * (Kx * Ky * Kz))

    # ── Forward FFTs (both DC bins = 0) ──
    fft_x = torch.fft.rfftn(x_fp16, s=(X, Y, Z), dim=(2, 3, 4), norm="ortho")
    fft_k = torch.fft.rfftn(k_fp16, s=(X, Y, Z), dim=(2, 3, 4), norm="ortho")
    del k_fp16

    shift_x = -((Kx - 1) // 2)
    shift_y = -((Ky - 1) // 2)
    shift_z = -((Kz - 1) // 2)

    del x_fp16  # x_c no longer needed (correction folded into fft_k)

    # Build effective kernel spectrum in complex64 (small tensor, no batch dim).
    fft_k_eff = fft_k.to(torch.complex64)
    del fft_k

    # T2: centering correction for zero-padded kernel positions.
    # The geometric factor is cached and only the scalar mu_k/sqrt_N changes.
    geo = _centering_geo_cache_3d.get(Kx, Ky, Kz, X, Y, Z, x.device)
    if geo is not None:
        fft_k_eff = fft_k_eff + (k_mean.float() / sqrt_N) * geo

    # Apply phase ramp for kernel alignment (spatial centering).
    if shift_x != 0 or shift_y != 0 or shift_z != 0:
        phase = _phase_ramp_cache_3d.get(
            X,
            Y,
            Z,
            shift_x,
            shift_y,
            shift_z,
            x.device,
            _PHASE_RAMP_COMPUTE_DTYPE,
        )
        fft_k_eff = fft_k_eff * phase

    # Multiply in complex32 (large batched tensor stays in half precision).
    fft_x.mul_(fft_k_eff.to(torch.complex32))

    # ── Inverse FFT (fp16) ──
    y = torch.fft.irfftn(fft_x, s=(X, Y, Z), dim=(2, 3, 4), norm="ortho")

    # ── Undo ortho scaling (×√N) and add DC correction (T4), in float32 ──
    y = (y.float() * sqrt_N + dc_corr).to(x.dtype)

    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (H,)
        y = y + rearrange(shortcut, "h -> 1 h 1 1 1") * x
    return y


def circular_fftconv3d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D circular FFT convolution in fp16, for inputs with layout ``[B, X, Y, Z, H]``.

    Wrapper around :func:`circular_fftconv3d_fp16_bhl` that handles the
    BLH <-> BHL reshape.
    """
    x_bhl = rearrange(x, "b x y z h -> b h x y z")
    k_bhl = rearrange(kernel, "b kx ky kz h -> b h kx ky kz")
    y = circular_fftconv3d_fp16_bhl(x_bhl, k_bhl, shortcut)
    return rearrange(y, "b h x y z -> b x y z h")
