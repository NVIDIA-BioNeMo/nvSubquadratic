"""FP16 FFT-based convolution for 1D, 2D, and 3D signals.

Leverages PyTorch's native fp16 FFT support (cuFFT C2C half-precision under the hood).
Two constraints must be satisfied:

1. cuFFT requires **power-of-2 sizes** for half-precision transforms.
2. The frequency-domain products can exceed fp16's 65504 limit with real model weights.

To avoid overflow we use ``norm="ortho"`` which divides both the forward and inverse
FFT by ``sqrt(N)``.  This keeps all intermediate complex values within fp16 range.
The ortho pair computes ``conv / sqrt(N)`` instead of ``conv``, so we multiply by
``sqrt(N)`` after the inverse FFT to restore the correct scale.

Families provided
-----------------
Standard:
  - 1D: ``fftconv1d_fp16_bhl``, ``causal_fftconv1d_fp16_bhl`` (+``_w_reshape`` variants)
  - 2D: ``fftconv2d_fp16_bhl`` (+``_w_reshape``)
  - 3D: ``fftconv3d_fp16_bhl`` (+``_w_reshape``)

Chunked (memory-efficient, processes channels in chunks):
  - 1D: ``fftconv1d_fp16_bhl_chunked``, ``causal_fftconv1d_fp16_bhl_chunked``
        (+``_w_reshape`` variants)
  - 2D: ``fftconv2d_fp16_bhl_chunked`` (+``_w_reshape``)
  - 3D: ``fftconv3d_fp16_bhl_chunked`` (+``_w_reshape``)
"""

import math

import torch
from einops import rearrange


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


###############################################################################
# 1D — non-causal
###############################################################################


def fftconv1d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution in fp16 with power-of-2 padding and ortho normalization.

    Drop-in replacement for ``fftconv1d_fp32_bhl``.  Casts *x* and *kernel* to fp16
    internally; shortcut is never cast.  Returns in the **original** dtype of *x*.

    Args:
        x: Input tensor ``[B, H, L]`` (any dtype).
        kernel: Kernel tensor ``[1, H, K]`` or ``[B, H, K]`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (any dtype, not cast).

    Returns:
        Tensor ``[B, H, L]`` in the original dtype of *x*.
    """
    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)

    _, hidden_dim, seq_len = x.shape
    _, _, kernel_len = kernel.shape

    fft_len = _next_power_of_2(min(seq_len + (kernel_len + 1) // 2, 2 * seq_len))
    sqrt_N = math.sqrt(fft_len)

    fft_x = torch.fft.rfft(x_fp16, n=fft_len, dim=2, norm="ortho")
    fft_k = torch.fft.rfft(k_fp16, n=fft_len, dim=2, norm="ortho")
    fft_x.mul_(fft_k)

    crop_start = kernel_len // 2
    y = torch.fft.irfft(fft_x, n=fft_len, dim=2, norm="ortho")[..., crop_start : crop_start + seq_len]

    # Upcast before scaling to avoid fp16 overflow (sqrt_N can be large)
    y = y.to(x.dtype)
    y = y * sqrt_N
    if shortcut is not None:
        y = y + rearrange(shortcut, "h -> 1 h 1") * x

    return y.to(x.dtype)


def fftconv1d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution in fp16, for inputs with layout ``[B, L, H]``.

    Wrapper around :func:`fftconv1d_fp16_bhl` that handles the BLH <-> BHL reshape.
    """
    x = rearrange(x, "b l h -> b h l")
    kernel = rearrange(kernel, "b l h -> b h l")
    y = fftconv1d_fp16_bhl(x, kernel, shortcut)
    return rearrange(y, "b h l -> b l h")


###############################################################################
# 1D — causal
###############################################################################


def causal_fftconv1d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal 1D FFT convolution in fp16 with power-of-2 padding and ortho normalization.

    Drop-in replacement for ``causal_fftconv1d_fp32_bhl``.  Casts *x* and *kernel* to
    fp16 internally; shortcut is never cast.  Returns in the **original** dtype of *x*.

    Args:
        x: Input tensor ``[B, H, L]`` (any dtype).
        kernel: Kernel tensor ``[1, H, K]`` or ``[B, H, K]`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (any dtype, not cast).

    Returns:
        Tensor ``[B, H, L]`` in the original dtype of *x*.
    """
    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)

    _, hidden_dim, seq_len = x.shape
    _, _, kernel_len = kernel.shape

    # Causal: need seq_len + kernel_len to avoid wraparound
    fft_len = _next_power_of_2(min(seq_len + kernel_len, 2 * seq_len))
    sqrt_N = math.sqrt(fft_len)

    fft_x = torch.fft.rfft(x_fp16, n=fft_len, dim=2, norm="ortho")
    fft_k = torch.fft.rfft(k_fp16, n=fft_len, dim=2, norm="ortho")
    fft_x.mul_(fft_k)

    y = torch.fft.irfft(fft_x, n=fft_len, dim=2, norm="ortho")[..., :seq_len]

    # Upcast before scaling to avoid fp16 overflow (sqrt_N can be large)
    y = y.to(x.dtype)
    y = y * sqrt_N
    if shortcut is not None:
        y = y + rearrange(shortcut, "h -> 1 h 1") * x

    return y.to(x.dtype)


def causal_fftconv1d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal 1D FFT convolution in fp16, for inputs with layout ``[B, L, H]``.

    Wrapper around :func:`causal_fftconv1d_fp16_bhl` that handles the BLH <-> BHL reshape.
    """
    x = rearrange(x, "b l h -> b h l")
    kernel = rearrange(kernel, "b l h -> b h l")
    y = causal_fftconv1d_fp16_bhl(x, kernel, shortcut)
    return rearrange(y, "b h l -> b l h")


###############################################################################
# 2D
###############################################################################


def fftconv2d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution in fp16 with power-of-2 padding and ortho normalization.

    Drop-in replacement for ``fftconv2d_fp32_bhl``.  Casts *x* and *kernel* to fp16
    internally; shortcut is never cast.  Returns in the **original** dtype of *x*.

    Args:
        x: Input tensor ``[B, H, X_in, Y_in]`` (any dtype).
        kernel: Kernel tensor ``[1, H, K_x, K_y]`` or ``[B, H, K_x, K_y]`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (any dtype, not cast).

    Returns:
        Tensor ``[B, H, X_in, Y_in]`` in the original dtype of *x*.
    """
    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)

    B, hidden_dim, X_in, Y_in = x.shape
    _, _, K_x, K_y = kernel.shape

    fft_shape = (
        _next_power_of_2(min(X_in + (K_x + 1) // 2, 2 * X_in)),
        _next_power_of_2(min(Y_in + (K_y + 1) // 2, 2 * Y_in)),
    )

    # sqrt(N) correction: ortho computes conv/sqrt(N), we need conv
    sqrt_N = math.sqrt(fft_shape[0] * fft_shape[1])

    fft_x = torch.fft.rfft2(x_fp16, s=fft_shape, dim=(2, 3), norm="ortho")
    fft_k = torch.fft.rfft2(k_fp16, s=fft_shape, dim=(2, 3), norm="ortho")

    fft_x.mul_(fft_k)

    crop_start_x = K_x // 2
    crop_start_y = K_y // 2

    y = torch.fft.irfft2(fft_x, s=fft_shape, dim=(2, 3), norm="ortho")[
        ..., crop_start_x : crop_start_x + X_in, crop_start_y : crop_start_y + Y_in
    ]

    # Upcast before scaling to avoid fp16 overflow (sqrt_N can be large)
    y = y.to(x.dtype)
    y = y * sqrt_N
    if shortcut is not None:
        y = y + rearrange(shortcut, "h -> 1 h 1 1") * x

    return y.to(x.dtype)


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


###############################################################################
# 3D
###############################################################################


def fftconv3d_fp16_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution in fp16 with power-of-2 padding and ortho normalization.

    Drop-in replacement for ``fftconv3d_fp32_bhl``.  Casts *x* and *kernel* to fp16
    internally; shortcut is never cast.  Returns in the **original** dtype of *x*.

    Args:
        x: Input tensor ``[B, H, X_in, Y_in, Z_in]`` (any dtype).
        kernel: Kernel tensor ``[1, H, K_x, K_y, K_z]`` or ``[B, H, K_x, K_y, K_z]`` (any dtype).
        shortcut: Optional per-channel shortcut ``[H]`` (any dtype, not cast).

    Returns:
        Tensor ``[B, H, X_in, Y_in, Z_in]`` in the original dtype of *x*.
    """
    x_fp16 = x.to(torch.float16)
    k_fp16 = kernel.to(torch.float16)

    B, hidden_dim, X_in, Y_in, Z_in = x.shape
    _, _, K_x, K_y, K_z = kernel.shape

    fft_shape = (
        _next_power_of_2(min(X_in + (K_x + 1) // 2, 2 * X_in)),
        _next_power_of_2(min(Y_in + (K_y + 1) // 2, 2 * Y_in)),
        _next_power_of_2(min(Z_in + (K_z + 1) // 2, 2 * Z_in)),
    )

    sqrt_N = math.sqrt(fft_shape[0] * fft_shape[1] * fft_shape[2])

    fft_x = torch.fft.rfftn(x_fp16, s=fft_shape, dim=(2, 3, 4), norm="ortho")
    fft_k = torch.fft.rfftn(k_fp16, s=fft_shape, dim=(2, 3, 4), norm="ortho")
    fft_x.mul_(fft_k)

    crop_start_x = K_x // 2
    crop_start_y = K_y // 2
    crop_start_z = K_z // 2

    y = torch.fft.irfftn(fft_x, s=fft_shape, dim=(2, 3, 4), norm="ortho")[
        :,
        :,
        crop_start_x : crop_start_x + X_in,
        crop_start_y : crop_start_y + Y_in,
        crop_start_z : crop_start_z + Z_in,
    ]

    # Upcast before scaling to avoid fp16 overflow (sqrt_N can be large)
    y = y.to(x.dtype)
    y = y * sqrt_N
    if shortcut is not None:
        y = y + rearrange(shortcut, "h -> 1 h 1 1 1") * x

    return y.to(x.dtype)


def fftconv3d_fp16_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution in fp16, for inputs with layout ``[B, X, Y, Z, H]``.

    Wrapper around :func:`fftconv3d_fp16_bhl` that handles the BLH <-> BHL reshape.
    """
    x = rearrange(x, "b x y z h -> b h x y z")
    kernel = rearrange(kernel, "b x y z h -> b h x y z")
    y = fftconv3d_fp16_bhl(x, kernel, shortcut)
    return rearrange(y, "b h x y z -> b x y z h")


###############################################################################
# Chunked (memory-efficient) variants
#
# Same channel-chunking strategy as fftconv_chunked.py but calling the fp16
# base functions.  This combines the ~36% fp16 memory savings with the ~26%
# chunking savings for maximum memory reduction.
###############################################################################

_DEFAULT_CHUNK_SIZE = 128


def fftconv1d_fp16_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """1D FFT convolution in fp16 with channel chunking (BHL layout).

    Args:
        x: Input tensor ``[B, H, L]``.
        kernel: Kernel tensor ``[1|B, H, K]``.
        shortcut: Optional per-channel shortcut ``[H]``.
        chunk_size: Channels per chunk (None = default 128).

    Returns:
        Tensor ``[B, H, L]`` in the original dtype of *x*.
    """
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE
    H = x.shape[1]
    if H <= chunk_size:
        return fftconv1d_fp16_bhl(x, kernel, shortcut)
    outputs = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        outputs.append(
            fftconv1d_fp16_bhl(
                x[:, start:end],
                kernel[:, start:end],
                shortcut[start:end] if shortcut is not None else None,
            )
        )
    return torch.cat(outputs, dim=1)


def fftconv1d_fp16_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """1D FFT convolution in fp16 with chunking, for ``[B, L, H]`` inputs."""
    x = rearrange(x, "b l h -> b h l")
    kernel = rearrange(kernel, "b l h -> b h l")
    y = fftconv1d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size)
    return rearrange(y, "b h l -> b l h")


def causal_fftconv1d_fp16_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Causal 1D FFT convolution in fp16 with channel chunking (BHL layout).

    Args:
        x: Input tensor ``[B, H, L]``.
        kernel: Kernel tensor ``[1|B, H, K]``.
        shortcut: Optional per-channel shortcut ``[H]``.
        chunk_size: Channels per chunk (None = default 128).

    Returns:
        Tensor ``[B, H, L]`` in the original dtype of *x*.
    """
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE
    H = x.shape[1]
    if H <= chunk_size:
        return causal_fftconv1d_fp16_bhl(x, kernel, shortcut)
    outputs = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        outputs.append(
            causal_fftconv1d_fp16_bhl(
                x[:, start:end],
                kernel[:, start:end],
                shortcut[start:end] if shortcut is not None else None,
            )
        )
    return torch.cat(outputs, dim=1)


def causal_fftconv1d_fp16_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Causal 1D FFT convolution in fp16 with chunking, for ``[B, L, H]`` inputs."""
    x = rearrange(x, "b l h -> b h l")
    kernel = rearrange(kernel, "b l h -> b h l")
    y = causal_fftconv1d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size)
    return rearrange(y, "b h l -> b l h")


def fftconv2d_fp16_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """2D FFT convolution in fp16 with channel chunking (BHL layout).

    Args:
        x: Input tensor ``[B, H, X_in, Y_in]``.
        kernel: Kernel tensor ``[1|B, H, K_x, K_y]``.
        shortcut: Optional per-channel shortcut ``[H]``.
        chunk_size: Channels per chunk (None = default 128).

    Returns:
        Tensor ``[B, H, X_in, Y_in]`` in the original dtype of *x*.
    """
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE
    H = x.shape[1]
    if H <= chunk_size:
        return fftconv2d_fp16_bhl(x, kernel, shortcut)
    outputs = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        outputs.append(
            fftconv2d_fp16_bhl(
                x[:, start:end],
                kernel[:, start:end],
                shortcut[start:end] if shortcut is not None else None,
            )
        )
    return torch.cat(outputs, dim=1)


def fftconv2d_fp16_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """2D FFT convolution in fp16 with chunking, for ``[B, X, Y, H]`` inputs."""
    x = rearrange(x, "b x y h -> b h x y")
    kernel = rearrange(kernel, "b x y h -> b h x y")
    y = fftconv2d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size)
    return rearrange(y, "b h x y -> b x y h")


def fftconv3d_fp16_bhl_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """3D FFT convolution in fp16 with channel chunking (BHL layout).

    Args:
        x: Input tensor ``[B, H, X_in, Y_in, Z_in]``.
        kernel: Kernel tensor ``[1|B, H, K_x, K_y, K_z]``.
        shortcut: Optional per-channel shortcut ``[H]``.
        chunk_size: Channels per chunk (None = default 128).

    Returns:
        Tensor ``[B, H, X_in, Y_in, Z_in]`` in the original dtype of *x*.
    """
    if chunk_size is None:
        chunk_size = _DEFAULT_CHUNK_SIZE
    H = x.shape[1]
    if H <= chunk_size:
        return fftconv3d_fp16_bhl(x, kernel, shortcut)
    outputs = []
    for start in range(0, H, chunk_size):
        end = min(start + chunk_size, H)
        outputs.append(
            fftconv3d_fp16_bhl(
                x[:, start:end],
                kernel[:, start:end],
                shortcut[start:end] if shortcut is not None else None,
            )
        )
    return torch.cat(outputs, dim=1)


def fftconv3d_fp16_bhl_w_reshape_chunked(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """3D FFT convolution in fp16 with chunking, for ``[B, X, Y, Z, H]`` inputs."""
    x = rearrange(x, "b x y z h -> b h x y z")
    kernel = rearrange(kernel, "b x y z h -> b h x y z")
    y = fftconv3d_fp16_bhl_chunked(x, kernel, shortcut, chunk_size)
    return rearrange(y, "b h x y z -> b x y z h")
