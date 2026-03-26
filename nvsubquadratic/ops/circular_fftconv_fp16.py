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

Alignment
---------
By default (``use_phase_shift=True``), alignment is done via a frequency-domain
phase ramp, the same approach used by the fp32 variants.  The ramp is computed
in float32 for precision and cast to complex32 before the multiply so all FFT
intermediates stay in half-precision.  Set ``use_phase_shift=False`` to fall
back to spatial ``torch.roll`` after the inverse FFT.

Families provided
-----------------
- 1D: ``circular_fftconv1d_fp16_bhl`` (+``_w_reshape``)
- 2D: ``circular_fftconv2d_fp16_bhl`` (+``_w_reshape``)
- 3D: ``circular_fftconv3d_fp16_bhl`` (+``_w_reshape``)
"""

from __future__ import annotations

import math

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
# 1D
###############################################################################


def circular_fftconv1d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """1D circular FFT convolution in fp16 with ortho normalization.

    Drop-in replacement for ``circular_fftconv1d_fp32_bhl``.  Casts *x* and
    *kernel* to fp16 internally; shortcut is never cast.  Returns in the
    **original** dtype of *x*.

    Requires ``L`` to be a power of 2 (cuFFT fp16 constraint).

    Args:
        x: Input tensor ``[B, H, L]`` (any dtype).
        kernel: Kernel tensor ``[1|B, H, K]`` with ``K <= L`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (same dtype as *x*).
            Never cast; the multiply auto-upcasts.
        use_phase_shift: If True (default), align via frequency-domain phase
            ramp.  If False, align via spatial ``torch.roll``.

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

    fft_x = torch.fft.rfft(x_fp16, n=L, dim=2, norm="ortho")
    fft_k = torch.fft.rfft(k_fp16, n=L, dim=2, norm="ortho")

    shift = -((K - 1) // 2)
    if use_phase_shift and shift != 0:
        phase = _phase_ramp_cache_1d.get(L, shift, x.device, _PHASE_RAMP_COMPUTE_DTYPE)
        fft_k = fft_k * phase.to(_PHASE_RAMP_TARGET_DTYPE)

    fft_x.mul_(fft_k)

    y = torch.fft.irfft(fft_x, n=L, dim=2, norm="ortho")

    if not use_phase_shift and shift != 0:
        y = torch.roll(y, shifts=(shift,), dims=(2,))

    # Upcast before scaling to avoid fp16 overflow (sqrt_N can be large).
    y = y.to(x.dtype)
    y = y * sqrt_N
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (H,)
        y = y + rearrange(shortcut, "h -> 1 h 1") * x
    return y


def circular_fftconv1d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """1D circular FFT convolution in fp16, for inputs with layout ``[B, L, H]``.

    Wrapper around :func:`circular_fftconv1d_fp16_bhl` that handles the
    BLH <-> BHL reshape.
    """
    x_bhl = rearrange(x, "b l h -> b h l")
    k_bhl = rearrange(kernel, "b k h -> b h k")
    y = circular_fftconv1d_fp16_bhl(x_bhl, k_bhl, shortcut, use_phase_shift=use_phase_shift)
    return rearrange(y, "b h l -> b l h")


###############################################################################
# 2D
###############################################################################


def circular_fftconv2d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """2D circular FFT convolution in fp16 with ortho normalization.

    Drop-in replacement for ``circular_fftconv2d_fp32_bhl``.  Casts *x* and
    *kernel* to fp16 internally; shortcut is never cast.  Returns in the
    **original** dtype of *x*.

    Requires ``X_in`` and ``Y_in`` to be powers of 2 (cuFFT fp16 constraint).

    Args:
        x: Input tensor ``[B, H, X_in, Y_in]`` (any dtype).
        kernel: Kernel tensor ``[1|B, H, K_x, K_y]`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (same dtype as *x*).
            Never cast; the multiply auto-upcasts.
        use_phase_shift: If True (default), align via frequency-domain phase
            ramp.  If False, align via spatial ``torch.roll``.

    Returns:
        Tensor ``[B, H, X_in, Y_in]`` in the original dtype of *x*.
    """
    B, H, X_in, Y_in = x.shape
    assert _is_power_of_2(X_in) and _is_power_of_2(Y_in), (
        f"Spatial dims must be powers of 2 for fp16 circular FFT. Got X_in={X_in}, Y_in={Y_in}."
    )
    assert len(kernel.shape) == 4, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )
    _, Hk, K_x, K_y = kernel.shape
    assert H == Hk, "Input and kernel must have the same number of channels (H)."
    assert K_x <= X_in, f"K_x must be <= X_in. Got K_x={K_x}, X_in={X_in}."
    assert K_y <= Y_in, f"K_y must be <= Y_in. Got K_y={K_y}, Y_in={Y_in}."

    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)

    sqrt_N = math.sqrt(X_in * Y_in)

    fft_x = torch.fft.rfft2(x_fp16, s=(X_in, Y_in), dim=(2, 3), norm="ortho")
    fft_k = torch.fft.rfft2(k_fp16, s=(X_in, Y_in), dim=(2, 3), norm="ortho")

    shift_x = -((K_x - 1) // 2)
    shift_y = -((K_y - 1) // 2)
    if use_phase_shift and (shift_x != 0 or shift_y != 0):
        phase = _phase_ramp_cache_2d.get(
            X_in,
            Y_in,
            shift_x,
            shift_y,
            x.device,
            _PHASE_RAMP_COMPUTE_DTYPE,
        )
        fft_k = fft_k * phase.to(_PHASE_RAMP_TARGET_DTYPE)

    fft_x.mul_(fft_k)

    y = torch.fft.irfft2(fft_x, s=(X_in, Y_in), dim=(2, 3), norm="ortho")

    if not use_phase_shift and (shift_x != 0 or shift_y != 0):
        y = torch.roll(y, shifts=(shift_x, shift_y), dims=(2, 3))

    y = y.to(x.dtype)
    y = y * sqrt_N
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (H,)
        y = y + rearrange(shortcut, "h -> 1 h 1 1") * x
    return y


def circular_fftconv2d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """2D circular FFT convolution in fp16, for inputs with layout ``[B, X, Y, H]``.

    Wrapper around :func:`circular_fftconv2d_fp16_bhl` that handles the
    BLH <-> BHL reshape.
    """
    x_bhl = rearrange(x, "b x y h -> b h x y")
    k_bhl = rearrange(kernel, "b kx ky h -> b h kx ky")
    y = circular_fftconv2d_fp16_bhl(x_bhl, k_bhl, shortcut, use_phase_shift=use_phase_shift)
    return rearrange(y, "b h x y -> b x y h")


###############################################################################
# 3D
###############################################################################


def circular_fftconv3d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """3D circular FFT convolution in fp16 with ortho normalization.

    Drop-in replacement for ``circular_fftconv3d_fp32_bhl``.  Casts *x* and
    *kernel* to fp16 internally; shortcut is never cast.  Returns in the
    **original** dtype of *x*.

    Requires ``X``, ``Y``, and ``Z`` to be powers of 2 (cuFFT fp16 constraint).

    Args:
        x: Input tensor ``[B, H, X, Y, Z]`` (any dtype).
        kernel: Kernel tensor ``[1|B, H, Kx, Ky, Kz]`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (same dtype as *x*).
            Never cast; the multiply auto-upcasts.
        use_phase_shift: If True (default), align via frequency-domain phase
            ramp.  If False, align via spatial ``torch.roll``.

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
    _, Hk, Kx_, Ky_, Kz_ = kernel.shape
    assert H == Hk, "Input and kernel must have the same number of channels (H)."
    assert Kx_ <= X and Ky_ <= Y and Kz_ <= Z, "Kernel must be <= input along each axis."

    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)

    sqrt_N = math.sqrt(X * Y * Z)

    fft_x = torch.fft.rfftn(x_fp16, s=(X, Y, Z), dim=(2, 3, 4), norm="ortho")
    fft_k = torch.fft.rfftn(k_fp16, s=(X, Y, Z), dim=(2, 3, 4), norm="ortho")

    shift_x = -((Kx_ - 1) // 2)
    shift_y = -((Ky_ - 1) // 2)
    shift_z = -((Kz_ - 1) // 2)
    if use_phase_shift and (shift_x != 0 or shift_y != 0 or shift_z != 0):
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
        fft_k = fft_k * phase.to(_PHASE_RAMP_TARGET_DTYPE)

    fft_x.mul_(fft_k)

    y = torch.fft.irfftn(fft_x, s=(X, Y, Z), dim=(2, 3, 4), norm="ortho")

    if not use_phase_shift and (shift_x != 0 or shift_y != 0 or shift_z != 0):
        y = torch.roll(y, shifts=(shift_x, shift_y, shift_z), dims=(2, 3, 4))

    y = y.to(x.dtype)
    y = y * sqrt_N
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (H,)
        y = y + rearrange(shortcut, "h -> 1 h 1 1 1") * x
    return y


def circular_fftconv3d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """3D circular FFT convolution in fp16, for inputs with layout ``[B, X, Y, Z, H]``.

    Wrapper around :func:`circular_fftconv3d_fp16_bhl` that handles the
    BLH <-> BHL reshape.
    """
    x_bhl = rearrange(x, "b x y z h -> b h x y z")
    k_bhl = rearrange(kernel, "b kx ky kz h -> b h kx ky kz")
    y = circular_fftconv3d_fp16_bhl(x_bhl, k_bhl, shortcut, use_phase_shift=use_phase_shift)
    return rearrange(y, "b h x y z -> b x y z h")
