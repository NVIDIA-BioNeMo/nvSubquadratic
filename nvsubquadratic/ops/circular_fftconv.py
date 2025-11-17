# TODO: Add license header here


"""Circular (periodic) FFT-based convolution operators for 2D signals.

Layout
------
- BHL: ``[batch, hidden, X_in, Y_in]``

Behavior
--------
- Computes depthwise circular convolution per channel (groups == H) using
  same-size FFTs (no zero-padding). Kernel is centered via circular roll so the
  result matches "same" alignment.

Shapes
------
- ``x: [B, H, X_in, Y_in]``
- ``kernel: [1|B, H, K_x, K_y]``

Shortcut
--------
- Optional ``shortcut: [H]`` scales input per-channel and is added to output:
  ``y += shortcut * x``.

Caching
-------
- This module keeps a small LRU cache of frequency-domain phase ramps used to
  replace spatial rolls. The cache is a module-level singleton and is therefore
  shared across all layers/callers within the same Python process (e.g., a
  single DDP rank). Different processes maintain their own caches.
"""

from __future__ import annotations


__all__ = [
    "circular_fftconv1d_bhl",
    "circular_fftconv1d_bhl_w_reshape",
    "circular_fftconv2d_bhl",
    "circular_fftconv2d_bhl_w_reshape",
    "circular_fftconv3d_bhl",
    "circular_fftconv3d_bhl_w_reshape",
]

import math
import time
from collections import OrderedDict

import torch
from einops import rearrange


class _PhaseRampCache1D:
    """Tiny LRU cache for 1D frequency-domain phase ramps used to replace spatial rolls.

    Purpose
    -------
    - Provides precomputed complex exponentials to implement an integer spatial shift
      via multiplication in the frequency domain: exp(-i 2π f · s).

    Keying
    ------
    - Keys are (L, s, device.type, device.index|(-1), complex_dtype) so that
      - Different sequence lengths get distinct ramps
      - Different shifts (kernel sizes/parity) get distinct ramps
      - Each device and dtype keeps its own tensor

    Memory
    ------
    - Each cached ramp has shape [Lf] where Lf = floor(L/2)+1 and complex dtype
      (complex64 for float32 inputs).
    """

    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()

    @staticmethod
    def _key(L: int, s: int, device: torch.device, real_dtype: torch.dtype) -> tuple:
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        dev_type = device.type
        dev_idx = device.index if device.index is not None else -1
        return (L, s, dev_type, dev_idx, complex_dtype)

    def get(self, L: int, s: int, device: torch.device, real_dtype: torch.dtype) -> torch.Tensor:
        """Return a 1D phase ramp tensor for given size/shift/device/dtype.

        Args:
            L (int): Sequence length.
            s (int): Integer pixel shift (spatial roll equivalent).
            device (torch.device): Target device for the tensor.
            real_dtype (torch.dtype): Real-valued dtype of inputs (float32/float64).

        Returns:
            Tensor of shape [L//2 + 1] with complex dtype, suitable to multiply
            elementwise with rfft(x/kernel) results (broadcasted over batch/channels).
        """
        key = self._key(L, s, device, real_dtype)
        phase = self._cache.get(key)
        if phase is not None:
            self._cache.move_to_end(key)
            return phase

        with torch.inference_mode(False):
            with torch.no_grad():
                f = torch.fft.rfftfreq(L, d=1.0, device=device, dtype=real_dtype)  # [Lf]
                phases = -2.0 * math.pi * (s * f)  # [Lf]
                phase = torch.complex(torch.cos(phases), torch.sin(phases))
        self._cache[key] = phase
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return phase


_phase_ramp_cache_1d = _PhaseRampCache1D(maxsize=64)


class _PhaseRampCache2D:
    """Tiny LRU (Least Recently Used) cache for frequency-domain phase ramps.

    A phase ramp implements a sub-pixel shift in the spatial domain by multiplying
    spectra with exp(-i 2π (fx * sx + fy * sy)). For circular 'same' alignment,
    we use integer shifts sx = -((Kx - 1)//2), sy = -((Ky - 1)//2).

    Keying
    ------
    Keys are (X, Y, sx, sy, device.type, device.index|(-1), complex_dtype) so that
    - Different spatial sizes build different ramps
    - Different shifts (kernel sizes/parity) get distinct ramps
    - Each device and dtype keeps its own tensor

    Memory
    ------
    Each cached ramp has shape [X, Y//2 + 1] with complex dtype (complex64 for
    float32 inputs). For example, X=1024, Y=1024 ~ 1024*513 complex numbers.
    The cache keeps at most `maxsize` such tensors live.
    """

    def __init__(self, maxsize: int = 64):
        # maxsize: upper bound on number of distinct phase tensors retained.
        self.maxsize = maxsize
        self._cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()

    @staticmethod
    def _key(
        X: int,
        Y: int,
        sx: int,
        sy: int,
        device: torch.device,
        real_dtype: torch.dtype,
    ) -> tuple:
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        dev_type = device.type
        dev_idx = device.index if device.index is not None else -1
        return (X, Y, sx, sy, dev_type, dev_idx, complex_dtype)

    def get(
        self,
        X: int,
        Y: int,
        sx: int,
        sy: int,
        device: torch.device,
        real_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a phase ramp tensor for given size/shift/device/dtype.

        Args:
            X (int): Spatial size along X.
            Y (int): Spatial size along Y.
            sx (int): Integer pixel shift along X (spatial roll equivalent).
            sy (int): Integer pixel shift along Y (spatial roll equivalent).
            device (torch.device): Target device for the tensor.
            real_dtype (torch.dtype): Real-valued dtype of inputs (float32/float64).

        Returns:
            Tensor of shape [X, Y//2 + 1] with complex dtype, suitable to multiply
            elementwise with rfft2(x/kernel) results (broadcasted over batch/channels).
        """
        key = self._key(X, Y, sx, sy, device, real_dtype)
        phase = self._cache.get(key)
        if phase is not None:
            self._cache.move_to_end(key)
            return phase

        with torch.inference_mode(False):
            with torch.no_grad():
                fx = torch.fft.fftfreq(X, d=1.0, device=device, dtype=real_dtype)  # [X]
                fy = torch.fft.rfftfreq(Y, d=1.0, device=device, dtype=real_dtype)  # [Y//2+1]
                phases = -2.0 * math.pi * (sx * fx[:, None] + sy * fy[None, :])  # [X, Yf]
                # Build complex phase e^{i*phases}
                phase = torch.complex(torch.cos(phases), torch.sin(phases))
        # Insert + evict LRU
        self._cache[key] = phase
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return phase


_phase_ramp_cache_2d = _PhaseRampCache2D(maxsize=64)


class _PhaseRampCache3D:
    """Tiny LRU cache for 3D frequency-domain phase ramps used to replace spatial rolls.

    Purpose
    -------
    - Provides precomputed complex exponentials to implement integer spatial shifts
      along three axes via multiplication in frequency domain:
      exp(-i 2π (fx*sx + fy*sy + fz*sz)).

    Keying
    ------
    - Keys are (X, Y, Z, sx, sy, sz, device.type, device.index|(-1), complex_dtype) so that
      - Different spatial sizes build different ramps
      - Different shifts (kernel sizes/parity) get distinct ramps
      - Each device and dtype keeps its own tensor

    Memory
    ------
    - Each cached ramp has shape [X, Y, Z//2 + 1] with complex dtype (complex64 for
      float32 inputs).
    """

    def __init__(self, maxsize: int = 64):
        self.maxsize = maxsize
        self._cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()

    @staticmethod
    def _key(
        X: int,
        Y: int,
        Z: int,
        sx: int,
        sy: int,
        sz: int,
        device: torch.device,
        real_dtype: torch.dtype,
    ) -> tuple:
        complex_dtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128
        dev_type = device.type
        dev_idx = device.index if device.index is not None else -1
        return (X, Y, Z, sx, sy, sz, dev_type, dev_idx, complex_dtype)

    def get(
        self,
        X: int,
        Y: int,
        Z: int,
        sx: int,
        sy: int,
        sz: int,
        device: torch.device,
        real_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return a 3D phase ramp tensor for given size/shift/device/dtype.

        Args:
            X (int): Spatial size along X.
            Y (int): Spatial size along Y.
            Z (int): Spatial size along Z.
            sx (int): Integer pixel shift along X (spatial roll equivalent).
            sy (int): Integer pixel shift along Y (spatial roll equivalent).
            sz (int): Integer pixel shift along Z (spatial roll equivalent).
            device (torch.device): Target device for the tensor.
            real_dtype (torch.dtype): Real-valued dtype of inputs (float32/float64).

        Returns:
            Tensor of shape [X, Y, Z//2 + 1] with complex dtype, suitable to
            multiply elementwise with rfftn(x/kernel) (broadcasted over batch/channels).
        """
        key = self._key(X, Y, Z, sx, sy, sz, device, real_dtype)
        phase = self._cache.get(key)
        if phase is not None:
            self._cache.move_to_end(key)
            return phase

        with torch.inference_mode(False):
            with torch.no_grad():
                fx = torch.fft.fftfreq(X, d=1.0, device=device, dtype=real_dtype)  # [X]
                fy = torch.fft.fftfreq(Y, d=1.0, device=device, dtype=real_dtype)  # [Y]
                fz = torch.fft.rfftfreq(Z, d=1.0, device=device, dtype=real_dtype)  # [Zf]
                phases = -2.0 * math.pi * (
                    sx * fx[:, None, None] + sy * fy[None, :, None] + sz * fz[None, None, :]
                )
                phase = torch.complex(torch.cos(phases), torch.sin(phases))  # [X, Y, Zf]
        self._cache[key] = phase
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return phase


_phase_ramp_cache_3d = _PhaseRampCache3D(maxsize=64)


def circular_fftconv1d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """1D circular FFT convolution with optional shortcut (BHL layout).

    Circular convolution
    --------------------
    - Spatially, circular (periodic) convolution can be written as:
        circular_conv(x, k) = roll(irfft(rfft(x) * rfft(k)))
    - To avoid the explicit spatial roll, we apply an equivalent frequency-domain
      phase ramp:
        circular_conv(x, k) = irfft(rfft(x) * rfft(k) * phase_ramp)
      where ``phase_ramp`` has shape ``[L//2 + 1]`` and encodes the integer shift.

    Layout and shapes
    -----------------
    - Layout: BHL (``[batch, hidden, length]``)
    - Inputs:
      - ``x: [B, H, L]``
      - ``kernel: [1|B, H, K]``
    - Output:
      - ``y: [B, H, L]``

    Alignment and shifts
    --------------------
    - We align to the "same" output by shifting with
      ``shift = -((K - 1) // 2)``.
    - If ``use_phase_shift=True``, we multiply the kernel spectrum by the cached
      phase ramp in frequency domain. Otherwise we roll the spatial output by
      ``shift`` after the inverse transform.

    Shortcut
    --------
    - Optional ``shortcut: [H]`` scales the input per-channel and is added to
      the convolution output: ``y += shortcut * x``.

    Caching
    -------
    - The phase ramp is retrieved from a global, module-level LRU cache shared
      across all layers/callers within the same process.

    Args:
        x (Tensor): ``[B, H, L]``, dtype float32.
        kernel (Tensor): ``[1|B, H, K]``, dtype float32.
        shortcut (Tensor | None): Optional ``[H]`` per-channel residual scale.
        use_phase_shift (bool): Use frequency-domain shift if True; else spatial roll.

    Returns:
        Tensor: ``[B, H, L]``
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    B, H, L = x.shape
    assert len(kernel.shape) == 3, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )
    _, Hk, K = kernel.shape

    # For same-size FFT (circular conv), enforce kernel dims not exceeding input dims
    assert H == Hk, "Input and kernel must have the same number of channels (H)."
    assert K <= L, f"K must be <= L. Got K={K}, L={L}."

    # Same-size real FFTs (circular convolution)
    fft_x = torch.fft.rfft(x, n=L, dim=2)
    fft_kernel = torch.fft.rfft(kernel, n=L, dim=2)

    # Alignment via integer shift:
    # - We want outputs to match a spatial “same” circular convolution with a flipped
    #   kernel (to convert correlation -> convolution). That alignment corresponds to
    #   rolling the spatial output up/left by floor((K-1)/2).
    # - Two equivalent implementations:
    #   (1) frequency-domain phase ramp (preferred: no extra memory move),
    #   (2) spatial torch.roll after iFFT (simple but moves real memory).
    shift = -((K - 1) // 2)
    if use_phase_shift:
        phase = _phase_ramp_cache_1d.get(L, shift, x.device, x.dtype)  # [Lf]
        fft_kernel.mul_(phase)  # broadcast over (B|1, H)

    # Depthwise per-channel multiplication
    fft_x.mul_(fft_kernel)

    y = torch.fft.irfft(fft_x, n=L, dim=2)
    if not use_phase_shift:
        y = torch.roll(y, shifts=(shift,), dims=(2,))

    if shortcut is not None:
        assert shortcut.shape == (H,)
        y.add_(rearrange(shortcut, "h -> 1 h 1") * x)

    return y


def circular_fftconv2d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """2D circular FFT convolution with optional shortcut (BHL layout).

    The circular convolution is computed as:
        circular_conv(x, kernel) = roll(ifft2(fft2(x) * fft2(kernel)))

    However, by using shifting on the frequency-domain phase ramp, we can compute the convolution as:
        circular_conv(x, kernel) = ifft2(fft2(x) * fft2(kernel) * phase_ramp)
    where phase_ramp is a complex tensor of shape (X_in, Y_in//2 + 1) with the phase ramp for the given shift.

    By doing so, this makes the convolution faster and more memory efficient.

    Args:
        x: Tensor of shape (B, H, X_in, Y_in), dtype float32.
        kernel: Tensor of shape (1|B, H, K_x, K_y), dtype float32.
        shortcut: Optional tensor of shape (H,), dtype float32.
        use_phase_shift: If True, apply alignment via frequency-domain phase ramp.
            If False, align via spatial torch.roll after iFFT.

    Returns:
        Tensor of shape (B, H, X_in, Y_in).

    Notes:
        When ``use_phase_shift=True``, the phase ramp is retrieved from a global,
        module-level LRU cache (shared across all layers/callers within the same
        Python process). This avoids recomputing the ramp for repeated sizes and
        shifts on the same device/dtype.
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    B, H, X_in, Y_in = x.shape

    assert len(kernel.shape) == 4, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )

    _, Hk, K_x, K_y = kernel.shape
    assert H == Hk, "Input and kernel must have the same number of channels (H)."

    # For same-size FFT (circular conv), enforce kernel dims not exceeding input dims
    assert K_x <= X_in, f"K_x must be <= X_in. Got K_x={K_x}, X_in={X_in}."
    assert K_y <= Y_in, f"K_y must be <= Y_in. Got K_y={K_y}, Y_in={Y_in}."

    # Same-size real FFTs (circular convolution)
    fft_x = torch.fft.rfft2(x, s=(X_in, Y_in), dim=(2, 3))
    fft_kernel = torch.fft.rfft2(kernel, s=(X_in, Y_in), dim=(2, 3))

    # Alignment via integer pixel shifts:
    # - We want outputs to match a spatial “same” circular convolution with a flipped
    #   kernel (to convert correlation -> convolution). That alignment corresponds to
    #   rolling the spatial output up/left by floor((K-1)/2) on each axis.
    # - Using (K-1)//2 unifies odd/even kernel sizes without branching.
    # - Two equivalent implementations:
    #   (1) frequency-domain phase ramp (preferred: no extra memory move),
    #   (2) spatial torch.roll after iFFT (simple but moves real memory).
    shift_x = -((K_x - 1) // 2)
    shift_y = -((K_y - 1) // 2)
    if use_phase_shift:
        phase = _phase_ramp_cache_2d.get(X_in, Y_in, shift_x, shift_y, x.device, x.dtype)
        fft_kernel.mul_(phase)  # broadcast over (B|1, H)

    # Depthwise per-channel multiplication
    fft_x.mul_(fft_kernel)

    y = torch.fft.irfft2(fft_x, s=(X_in, Y_in), dim=(2, 3))
    if not use_phase_shift:
        y = torch.roll(y, shifts=(shift_x, shift_y), dims=(2, 3))

    if shortcut is not None:
        assert shortcut.shape == (H,)
        y.add_(rearrange(shortcut, "h -> 1 h 1 1") * x)

    return y


def circular_fftconv3d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """3D circular FFT convolution with optional shortcut (BHL layout).

    Circular convolution
    --------------------
    - Spatially, circular (periodic) convolution can be written as:
        circular_conv(x, k) = roll(irfftn(rfftn(x) * rfftn(k)))
    - To avoid the explicit spatial roll, we apply an equivalent frequency-domain
      phase ramp:
        circular_conv(x, k) = irfftn(rfftn(x) * rfftn(k) * phase_ramp)
      where ``phase_ramp`` has shape ``[X, Y, Z//2 + 1]`` and encodes the integer shifts.

    Layout and shapes
    -----------------
    - Layout: BHL (``[batch, hidden, X, Y, Z]``)
    - Inputs:
      - ``x: [B, H, X, Y, Z]``
      - ``kernel: [1|B, H, Kx, Ky, Kz]``
    - Output:
      - ``y: [B, H, X, Y, Z]``

    Alignment and shifts
    --------------------
    - We align to the "same" output by shifting with:
      ``shift_x = -((Kx - 1)//2)``, ``shift_y = -((Ky - 1)//2)``, ``shift_z = -((Kz - 1)//2)``.
    - If ``use_phase_shift=True``, we multiply the kernel spectrum by the cached
      3D phase ramp in frequency domain. Otherwise we roll the spatial output
      after the inverse transform.

    Shortcut
    --------
    - Optional ``shortcut: [H]`` scales the input per-channel and is added to
      the convolution output: ``y += shortcut * x``.

    Caching
    -------
    - The 3D phase ramp is retrieved from a global, module-level LRU cache
      shared across all layers/callers within the same process.

    Args:
        x (Tensor): ``[B, H, X, Y, Z]``, dtype float32.
        kernel (Tensor): ``[1|B, H, Kx, Ky, Kz]``, dtype float32.
        shortcut (Tensor | None): Optional ``[H]`` per-channel residual scale.
        use_phase_shift (bool): Use frequency-domain shift if True; else spatial roll.

    Returns:
        Tensor: ``[B, H, X, Y, Z]``
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    B, H, X, Y, Z = x.shape
    assert len(kernel.shape) == 5, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )
    _, Hk, Kx_, Ky_, Kz_ = kernel.shape

    # For same-size FFT (circular conv), enforce kernel dims not exceeding input dims
    assert H == Hk, "Input and kernel must have the same number of channels (H)."
    assert Kx_ <= X and Ky_ <= Y and Kz_ <= Z, "Kernel must be <= input along each axis."

    # Same-size real FFTs (circular convolution)
    fft_x = torch.fft.rfftn(x, s=(X, Y, Z), dim=(2, 3, 4))
    fft_kernel = torch.fft.rfftn(kernel, s=(X, Y, Z), dim=(2, 3, 4))

    # Alignment via integer shift:
    # - We want outputs to match a spatial “same” circular convolution with a flipped
    #   kernel (to convert correlation -> convolution). That alignment corresponds to
    #   rolling the spatial output up/left by floor((K-1)/2) on each axis.
    # - Using (K-1)//2 unifies odd/even kernel sizes without branching.
    # - Two equivalent implementations:
    #   (1) frequency-domain phase ramp (preferred: no extra memory move),
    #   (2) spatial torch.roll after iFFT (simple but moves real memory).
    shift_x = -((Kx_ - 1) // 2)
    shift_y = -((Ky_ - 1) // 2)
    shift_z = -((Kz_ - 1) // 2)
    if use_phase_shift:
        phase = _phase_ramp_cache_3d.get(X, Y, Z, shift_x, shift_y, shift_z, x.device, x.dtype)  # [X,Y,Zf]
        fft_kernel.mul_(phase)  # broadcast over (B|1, H)

    # Depthwise per-channel multiplication
    fft_x.mul_(fft_kernel)

    y = torch.fft.irfftn(fft_x, s=(X, Y, Z), dim=(2, 3, 4))
    if not use_phase_shift:
        y = torch.roll(y, shifts=(shift_x, shift_y, shift_z), dims=(2, 3, 4))

    if shortcut is not None:
        assert shortcut.shape == (H,)
        y.add_(rearrange(shortcut, "h -> 1 h 1 1 1") * x)

    return y


def circular_fftconv1d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """1D circular FFT conv wrapper for BLH layout (batch, length, hidden).

    This reshapes BLH -> BHL, calls ``circular_fftconv1d_bhl``, and reshapes back.

    Args:
        x (Tensor): ``[B, L, H]``, dtype float32.
        kernel (Tensor): ``[1|B, K, H]``, dtype float32.
        shortcut (Tensor | None): Optional ``[H]`` per-channel residual scale.
        use_phase_shift (bool): Use frequency-domain shift if True; else spatial roll.

    Returns:
        Tensor: ``[B, L, H]``
    """
    x_bhl = rearrange(x, "b l h -> b h l")
    kernel_bhl = rearrange(kernel, "b k h -> b h k")
    y_bhl = circular_fftconv1d_bhl(x_bhl, kernel_bhl, shortcut, use_phase_shift=use_phase_shift)
    return rearrange(y_bhl, "b h l -> b l h")


def circular_fftconv2d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """2D circular FFT conv wrapper for BLH layout (batch, height, width, hidden).

    This reshapes BLH -> BHL, calls ``circular_fftconv2d_bhl``, and reshapes back.

    Args:
        x (Tensor): ``[B, X, Y, H]``, dtype float32.
        kernel (Tensor): ``[1|B, Kx, Ky, H]``, dtype float32.
        shortcut (Tensor | None): Optional ``[H]`` per-channel residual scale.
        use_phase_shift (bool): Use frequency-domain shift if True; else spatial roll.

    Returns:
        Tensor: ``[B, X, Y, H]``
    """
    x_bhl = rearrange(x, "b x y h -> b h x y")
    kernel_bhl = rearrange(kernel, "b kx ky h -> b h kx ky")
    y_bhl = circular_fftconv2d_bhl(x_bhl, kernel_bhl, shortcut, use_phase_shift=use_phase_shift)
    return rearrange(y_bhl, "b h x y -> b x y h")


def circular_fftconv3d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    use_phase_shift: bool = True,
) -> torch.Tensor:
    """3D circular FFT conv wrapper for BLH layout (batch, depth, height, width, hidden).

    This reshapes BLH -> BHL, calls ``circular_fftconv3d_bhl``, and reshapes back.

    Args:
        x (Tensor): ``[B, X, Y, Z, H]``, dtype float32.
        kernel (Tensor): ``[1|B, Kx, Ky, Kz, H]``, dtype float32.
        shortcut (Tensor | None): Optional ``[H]`` per-channel residual scale.
        use_phase_shift (bool): Use frequency-domain shift if True; else spatial roll.

    Returns:
        Tensor: ``[B, X, Y, Z, H]``
    """
    x_bhl = rearrange(x, "b x y z h -> b h x y z")
    kernel_bhl = rearrange(kernel, "b kx ky kz h -> b h kx ky kz")
    y_bhl = circular_fftconv3d_bhl(x_bhl, kernel_bhl, shortcut, use_phase_shift=use_phase_shift)
    return rearrange(y_bhl, "b h x y z -> b x y z h")


if __name__ == "__main__":
    # Minimal quick correctness + speed comparison for 1D, 2D, and 3D on H100
    """
    Scaling benchmark 1D (lengths, kernel == input), B=8, H=64:
    N=64: phase 0.000068s | roll 0.000070s | diff 7.08e-05 | phase|max 3.345e+01 mean 6.294e+00 | roll|max 3.345e+01 mean 6.294e+00
    N=128: phase 0.000063s | roll 0.000064s | diff 2.55e-04 | phase|max 5.120e+01 mean 9.072e+00 | roll|max 5.120e+01 mean 9.072e+00
    N=256: phase 0.000065s | roll 0.000064s | diff 6.36e-04 | phase|max 7.527e+01 mean 1.278e+01 | roll|max 7.527e+01 mean 1.278e+01
    N=512: phase 0.000062s | roll 0.000067s | diff 1.87e-03 | phase|max 1.237e+02 mean 1.807e+01 | roll|max 1.237e+02 mean 1.807e+01
    N=1024: phase 0.000065s | roll 0.000061s | diff 5.43e-03 | phase|max 1.540e+02 mean 2.546e+01 | roll|max 1.540e+02 mean 2.546e+01
    N=2048: phase 0.000066s | roll 0.000064s | diff 1.56e-02 | phase|max 2.322e+02 mean 3.620e+01 | roll|max 2.321e+02 mean 3.620e+01
    N=4096: phase 0.000066s | roll 0.000067s | diff 4.61e-02 | phase|max 3.152e+02 mean 5.100e+01 | roll|max 3.152e+02 mean 5.100e+01
    N=8192: phase 0.000071s | roll 0.000088s | diff 1.38e-01 | phase|max 5.018e+02 mean 7.238e+01 | roll|max 5.018e+02 mean 7.238e+01
    N=16384: phase 0.000159s | roll 0.000194s | diff 4.80e-01 | phase|max 6.796e+02 mean 1.021e+02 | roll|max 6.795e+02 mean 1.021e+02
    N=32768: phase 0.000345s | roll 0.000420s | diff 1.27e+00 | phase|max 9.762e+02 mean 1.443e+02 | roll|max 9.761e+02 mean 1.443e+02
    N=65536: phase 0.000744s | roll 0.000896s | diff 3.69e+00 | phase|max 1.434e+03 mean 2.043e+02 | roll|max 1.434e+03 mean 2.043e+02

    Scaling benchmark (input sizes, kernel == input), B=8, H=64:
    N=64x64: phase 0.000110s | roll 0.000125s | diff 9.69e-04 | phase|max 3.138e+02 mean 5.105e+01 | roll|max 3.138e+02 mean 5.105e+01
    N=128x128: phase 0.000202s | roll 0.000283s | diff 4.39e-03 | phase|max 7.020e+02 mean 1.022e+02 | roll|max 7.020e+02 mean 1.022e+02
    N=256x256: phase 0.000778s | roll 0.001089s | diff 1.75e-02 | phase|max 1.390e+03 mean 2.042e+02 | roll|max 1.390e+03 mean 2.042e+02
    N=512x512: phase 0.003067s | roll 0.004264s | diff 7.34e-02 | phase|max 2.969e+03 mean 4.086e+02 | roll|max 2.969e+03 mean 4.086e+02
    N=1024x1024: phase 0.011948s | roll 0.016724s | diff 3.20e-01 | phase|max 6.198e+03 mean 8.170e+02 | roll|max 6.198e+03 mean 8.170e+02

    Scaling benchmark 3D (cubes, kernel == input), B=2, H=16:
    N=16^3: phase 0.000085s | roll 0.000112s | diff 3.47e-04 | phase|max 3.085e+02 mean 5.100e+01 | roll|max 3.085e+02 mean 5.100e+01
    N=24^3: phase 0.000087s | roll 0.000096s | diff 1.27e-03 | phase|max 6.091e+02 mean 9.399e+01 | roll|max 6.091e+02 mean 9.399e+01
    N=32^3: phase 0.000085s | roll 0.000096s | diff 1.43e-03 | phase|max 9.444e+02 mean 1.444e+02 | roll|max 9.444e+02 mean 1.444e+02
    N=48^3: phase 0.000142s | roll 0.000188s | diff 9.03e-03 | phase|max 1.658e+03 mean 2.650e+02 | roll|max 1.658e+03 mean 2.650e+02
    N=64^3: phase 0.000293s | roll 0.000410s | diff 9.40e-03 | phase|max 2.940e+03 mean 4.085e+02 | roll|max 2.940e+03 mean 4.085e+02
    """
    torch.manual_seed(0)

    # 1D quick correctness
    B, H, L = 2, 8, 256
    K = 65
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    x = torch.randn(B, H, L, device=device, dtype=dtype)
    k = torch.randn(H, K, device=device, dtype=dtype)

    # Reference using circular padding and flipped kernel (to get convolution)
    k_for_conv = k.unsqueeze(1)  # [H,1,K]
    k_flipped = torch.flip(k_for_conv, dims=[-1])
    pad_left = K // 2
    pad_right = K - 1 - K // 2
    padded = torch.nn.functional.pad(x, (pad_left, pad_right), mode="circular")
    ref = torch.nn.functional.conv1d(padded, k_flipped, groups=H, padding=0)

    # Correctness check (small size to keep it quick)
    x_small = torch.randn(1, 2, 256, device=device, dtype=dtype)
    k_small = torch.randn(2, 31, device=device, dtype=dtype)
    k_small_for_conv = k_small.unsqueeze(1)  # [H,1,K]
    k_small_flip = torch.flip(k_small_for_conv, dims=[-1])
    pad_left_s = k_small.shape[-1] // 2
    pad_right_s = k_small.shape[-1] - 1 - k_small.shape[-1] // 2
    padded_small = torch.nn.functional.pad(x_small, (pad_left_s, pad_right_s), mode="circular")
    ref_small = torch.nn.functional.conv1d(padded_small, k_small_flip, groups=2, padding=0)
    y_phase_small = circular_fftconv1d_bhl(x_small, rearrange(k_small, "h k -> 1 h k"), use_phase_shift=True)
    y_roll_small = circular_fftconv1d_bhl(x_small, rearrange(k_small, "h k -> 1 h k"), use_phase_shift=False)
    print("max abs diff (phase vs ref):", (y_phase_small - ref_small).abs().max().item())
    print("max abs diff (roll  vs ref):", (y_roll_small - ref_small).abs().max().item())
    print("max abs diff (phase vs roll):", (y_phase_small - y_roll_small).abs().max().item())

    # Speed test across increasing input lengths (kernel size == input length)
    B, H = 8, 64
    print(f"\nScaling benchmark 1D (lengths, kernel == input), B={B}, H={H}:")
    lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    for N in lengths:
        xN = torch.randn(B, H, N, device=device, dtype=dtype)
        kN = torch.randn(H, N, device=device, dtype=dtype)  # full-size kernel
        kN_ = rearrange(kN, "h k -> 1 h k")
        # Warmup
        for _ in range(10):
            _ = circular_fftconv1d_bhl(xN, kN_, use_phase_shift=True)
            _ = circular_fftconv1d_bhl(xN, kN_, use_phase_shift=False)
        reps = 20
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            y_phase = circular_fftconv1d_bhl(xN, kN_, use_phase_shift=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(reps):
            y_roll = circular_fftconv1d_bhl(xN, kN_, use_phase_shift=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        avg_phase = (t1 - t0) / reps
        avg_roll = (t2 - t1) / reps
        max_diff = (y_phase - y_roll).abs().max().item()
        # Magnitudes (to see if larger outputs correlate with larger absolute diffs)
        y_phase_abs_max = y_phase.abs().max().item()
        y_phase_abs_mean = y_phase.abs().mean().item()
        y_roll_abs_max = y_roll.abs().max().item()
        y_roll_abs_mean = y_roll.abs().mean().item()
        print(
            f"  N={N}: phase {avg_phase:.6f}s | roll {avg_roll:.6f}s | diff {max_diff:.2e} | "
            f"phase|max {y_phase_abs_max:.3e} mean {y_phase_abs_mean:.3e} | "
            f"roll|max {y_roll_abs_max:.3e} mean {y_roll_abs_mean:.3e}"
        )

    # 2D quick correctness
    print("\n2D quick correctness check...")
    B, H, X2, Y2 = 2, 8, 256, 256
    Kx2, Ky2 = 65, 65
    x2 = torch.randn(B, H, X2, Y2, device=device, dtype=dtype)
    k2 = torch.randn(H, Kx2, Ky2, device=device, dtype=dtype)
    k2_for_conv = k2.unsqueeze(1)  # [H,1,Kx,Ky]
    k2_flip = torch.flip(k2_for_conv, dims=[-1, -2])
    pad_w_left = Ky2 // 2
    pad_w_right = Ky2 - 1 - Ky2 // 2
    pad_h_top = Kx2 // 2
    pad_h_bottom = Kx2 - 1 - Kx2 // 2
    padded2 = torch.nn.functional.pad(x2, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom), mode="circular")
    ref2 = torch.nn.functional.conv2d(padded2, k2_flip, groups=H, padding=0)
    y2_phase = circular_fftconv2d_bhl(x2, rearrange(k2, "h kx ky -> 1 h kx ky"), use_phase_shift=True)
    y2_roll = circular_fftconv2d_bhl(x2, rearrange(k2, "h kx ky -> 1 h kx ky"), use_phase_shift=False)
    print("2D max abs diff (phase vs ref):", (y2_phase - ref2).abs().max().item())
    print("2D max abs diff (roll  vs ref):", (y2_roll - ref2).abs().max().item())
    print("2D max abs diff (phase vs roll):", (y2_phase - y2_roll).abs().max().item())

    # 2D scaling benchmark (kernel == input size)
    B, H = 8, 64
    print(f"\nScaling benchmark 2D (squares, kernel == input), B={B}, H={H}:")
    squares = [64, 128, 256, 512, 1024]
    for N in squares:
        XN, YN = N, N
        xN = torch.randn(B, H, XN, YN, device=device, dtype=dtype)
        kN = torch.randn(H, XN, YN, device=device, dtype=dtype)
        kN_ = rearrange(kN, "h kx ky -> 1 h kx ky")
        # Warmup
        for _ in range(10):
            _ = circular_fftconv2d_bhl(xN, kN_, use_phase_shift=True)
            _ = circular_fftconv2d_bhl(xN, kN_, use_phase_shift=False)
        reps = 20
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            y_phase = circular_fftconv2d_bhl(xN, kN_, use_phase_shift=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(reps):
            y_roll = circular_fftconv2d_bhl(xN, kN_, use_phase_shift=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        avg_phase = (t1 - t0) / reps
        avg_roll = (t2 - t1) / reps
        max_diff = (y_phase - y_roll).abs().max().item()
        # Magnitudes
        y_phase_abs_max = y_phase.abs().max().item()
        y_phase_abs_mean = y_phase.abs().mean().item()
        y_roll_abs_max = y_roll.abs().max().item()
        y_roll_abs_mean = y_roll.abs().mean().item()
        print(
            f"  N={N}x{N}: phase {avg_phase:.6f}s | roll {avg_roll:.6f}s | diff {max_diff:.2e} | "
            f"phase|max {y_phase_abs_max:.3e} mean {y_phase_abs_mean:.3e} | "
            f"roll|max {y_roll_abs_max:.3e} mean {y_roll_abs_mean:.3e}"
        )

    # 3D quick correctness
    print("\n3D quick correctness check...")
    B, H, X, Y, Z = 2, 4, 32, 32, 32
    Kx, Ky, Kz = 9, 9, 9
    x3 = torch.randn(B, H, X, Y, Z, device=device, dtype=dtype)
    k3 = torch.randn(H, Kx, Ky, Kz, device=device, dtype=dtype)
    k3_for_conv = k3.unsqueeze(1)  # [H,1,Kx,Ky,Kz]
    k3_flip = torch.flip(k3_for_conv, dims=[-1, -2, -3])
    pad_w_left = Kz // 2
    pad_w_right = Kz - 1 - Kz // 2
    pad_h_top = Ky // 2
    pad_h_bottom = Ky - 1 - Ky // 2
    pad_d_front = Kx // 2
    pad_d_back = Kx - 1 - Kx // 2
    padded3 = torch.nn.functional.pad(
        x3, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_front, pad_d_back), mode="circular"
    )
    ref3 = torch.nn.functional.conv3d(padded3, k3_flip, groups=H, padding=0)
    y3_phase = circular_fftconv3d_bhl(x3, rearrange(k3, "h kx ky kz -> 1 h kx ky kz"), use_phase_shift=True)
    y3_roll = circular_fftconv3d_bhl(x3, rearrange(k3, "h kx ky kz -> 1 h kx ky kz"), use_phase_shift=False)
    print("3D max abs diff (phase vs ref):", (y3_phase - ref3).abs().max().item())
    print("3D max abs diff (roll  vs ref):", (y3_roll - ref3).abs().max().item())
    print("3D max abs diff (phase vs roll):", (y3_phase - y3_roll).abs().max().item())

    # 3D scaling benchmark (kernel == input). Choose conservative sizes for memory.
    B, H = 2, 16
    print(f"\nScaling benchmark 3D (cubes, kernel == input), B={B}, H={H}:")
    cube_sizes = [16, 24, 32, 48, 64]
    for N in cube_sizes:
        XN, YN, ZN = N, N, N
        xN = torch.randn(B, H, XN, YN, ZN, device=device, dtype=dtype)
        kN = torch.randn(H, XN, YN, ZN, device=device, dtype=dtype)  # full-size kernel
        kN_ = rearrange(kN, "h kx ky kz -> 1 h kx ky kz")
        # Warmup
        for _ in range(10):
            _ = circular_fftconv3d_bhl(xN, kN_, use_phase_shift=True)
            _ = circular_fftconv3d_bhl(xN, kN_, use_phase_shift=False)
        reps = 20
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            y_phase = circular_fftconv3d_bhl(xN, kN_, use_phase_shift=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(reps):
            y_roll = circular_fftconv3d_bhl(xN, kN_, use_phase_shift=False)
        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        avg_phase = (t1 - t0) / reps
        avg_roll = (t2 - t1) / reps
        max_diff = (y_phase - y_roll).abs().max().item()
        # Magnitudes
        y_phase_abs_max = y_phase.abs().max().item()
        y_phase_abs_mean = y_phase.abs().mean().item()
        y_roll_abs_max = y_roll.abs().max().item()
        y_roll_abs_mean = y_roll.abs().mean().item()
        print(
            f"  N={N}^3: phase {avg_phase:.6f}s | roll {avg_roll:.6f}s | diff {max_diff:.2e} | "
            f"phase|max {y_phase_abs_max:.3e} mean {y_phase_abs_mean:.3e} | "
            f"roll|max {y_roll_abs_max:.3e} mean {y_roll_abs_mean:.3e}"
        )
