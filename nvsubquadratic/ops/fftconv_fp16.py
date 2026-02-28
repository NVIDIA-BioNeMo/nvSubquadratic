"""FP16 FFT-based convolution for 2D signals.

Leverages PyTorch's native fp16 FFT support (cuFFT C2C half-precision under the hood).
Two constraints must be satisfied:

1. cuFFT requires **power-of-2 sizes** for half-precision transforms.
2. The frequency-domain products can exceed fp16's 65504 limit with real model weights.

To avoid overflow we use ``norm="ortho"`` which divides both the forward and inverse
FFT by ``sqrt(N)``.  This keeps all intermediate complex values within fp16 range.
The ortho pair computes ``conv / sqrt(N)`` instead of ``conv``, so we multiply by
``sqrt(N)`` after the inverse FFT to restore the correct scale.
"""

import math

import torch
from einops import rearrange


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def fftconv2d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution in fp16 with power-of-2 padding and ortho normalization.

    Drop-in replacement for ``fftconv2d_bhl``.  Inputs may be any dtype; they are
    cast to fp16 internally.  The output dtype matches *x*.

    Args:
        x: Input tensor ``[B, H, X_in, Y_in]``.
        kernel: Kernel tensor ``[1, H, K_x, K_y]`` or ``[B, H, K_x, K_y]``.
        shortcut: Optional per-channel shortcut ``[H]``.

    Returns:
        Tensor ``[B, H, X_in, Y_in]`` in the **original** dtype of *x*.
    """
    x_dtype = x.dtype
    B, hidden_dim, X_in, Y_in = x.shape
    _, _, K_x, K_y = kernel.shape

    fft_shape = (
        _next_power_of_2(min(X_in + (K_x + 1) // 2, 2 * X_in)),
        _next_power_of_2(min(Y_in + (K_y + 1) // 2, 2 * Y_in)),
    )

    # sqrt(N) correction: ortho computes conv/sqrt(N), we need conv
    sqrt_N = math.sqrt(fft_shape[0] * fft_shape[1])

    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)

    fft_x = torch.fft.rfft2(x_fp16, s=fft_shape, dim=(2, 3), norm="ortho")
    fft_k = torch.fft.rfft2(k_fp16, s=fft_shape, dim=(2, 3), norm="ortho")

    fft_x.mul_(fft_k)

    crop_start_x = K_x // 2
    crop_start_y = K_y // 2

    y = torch.fft.irfft2(fft_x, s=fft_shape, dim=(2, 3), norm="ortho")[
        ..., crop_start_x: crop_start_x + X_in, crop_start_y: crop_start_y + Y_in
    ]

    # Restore correct convolution scale (ortho gives conv / sqrt(N))
    y = y * sqrt_N

    if shortcut is not None:
        y = y + rearrange(shortcut.to(torch.float16), "h -> 1 h 1 1") * x_fp16

    return y.to(x_dtype)


def fftconv2d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution in fp16, for inputs with layout ``[B, X, Y, H]``.

    Wrapper around :func:`fftconv2d_fp16_bhl` that handles the BLH <-> BHL reshape.
    """
    x = rearrange(x, "b x y h -> b h x y")
    kernel = rearrange(kernel, "b x y h -> b h x y")
    y = fftconv2d_fp16_bhl(x, kernel, shortcut)
    return rearrange(y, "b h x y -> b x y h")
