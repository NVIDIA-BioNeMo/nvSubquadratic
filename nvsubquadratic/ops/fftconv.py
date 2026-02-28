# TODO: Add license header here


"""FFT-based convolution operators for 1D, 2D, and 3D signals.

This module provides fast FFT convolutions for both common memory layouts:

- BLH: ``[batch, * spatial_dims, hidden]``
- BHL: ``[batch, hidden, * spatial_dims]``

Families provided
-----------------
- 1D convolutions (causal and non-causal) with optional per-channel shortcut
  - BLH: ``causal_fftconv1d_blh``, ``fftconv1d_blh``
  - BHL: ``causal_fftconv1d_bhl``, ``fftconv1d_bhl``
- 2D convolutions with optional per-channel shortcut
  - BLH: ``fftconv2d_blh``
  - BHL: ``fftconv2d_bhl``
- 3D convolutions with optional per-channel shortcut
  - BLH: ``fftconv3d_blh``
  - BHL: ``fftconv3d_bhl``

Wrapper variants (recommended for BLH inputs)
--------------------------------------------
- ``*_bhl_w_reshape`` wrappers accept BLH inputs, internally reshape to BHL for
  faster execution, apply the BHL operator, and then reshape back. They return
  tensors in the same layout they received (BLH).

Shapes and conventions
----------------------
- BLH inputs and kernels:
  - 1D: ``x: [B, L, H]``, ``kernel: [1|B, K, H]``
  - 2D: ``x: [B, X_in, Y_in, H]``, ``kernel: [1|B, K_x, K_y, H]``
  - 3D: ``x: [B, X_in, Y_in, Z_in, H]``, ``kernel: [1|B, K_x, K_y, K_z, H]``
- BHL inputs and kernels:
  - 1D: ``x: [B, H, L]``, ``kernel: [1|B, H, K]``
  - 2D: ``x: [B, H, X_in, Y_in]``, ``kernel: [1|B, H, K_x, K_y]``
  - 3D: ``x: [B, H, X_in, Y_in, Z_in]``, ``kernel: [1|B, H, K_x, K_y, K_z]``

Cropping and causality
----------------------
- Non-causal variants produce "same" outputs by cropping the linear convolution
  result centered on the input. Crop offsets are ``K//2`` (1D) or per-axis.
- Causal 1D uses ``fft_len = min(L + K, 2L)`` and crops the tail to length ``L``.
  Non-causal 1D uses ``fft_len = min(L + ceil(K/2), 2L)`` and centers the crop.
- Non-causal variants are faster and more memory efficient, as they require
  less padding.

Shortcuts and dtype
-------------------
- Optional ``shortcut: [H]`` scales the input per-channel and is added to the
  convolution output: ``y += shortcut * x`` (broadcasted along spatial dims).
- All operators expect ``float32`` inputs, kernels, and shortcut.

Performance
-----------
- For BLH inputs, prefer the ``*_bhl_w_reshape`` wrappers; benchmarks show they
  are faster than operating directly in BLH layout.
"""

__all__ = [
    "causal_fftconv1d_bhl",
    "causal_fftconv1d_bhl_w_reshape",
    "causal_fftconv1d_blh",
    "fftconv1d_bhl",
    "fftconv1d_bhl_w_reshape",
    "fftconv1d_blh",
    "fftconv2d_bhl",
    "fftconv2d_bhl_w_reshape",
    "fftconv2d_blh",
    "fftconv3d_bhl",
    "fftconv3d_bhl_w_reshape",
    "fftconv3d_blh",
]

import torch
from einops import rearrange


###############################################################################
# BLH variants
###############################################################################


def causal_fftconv1d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut. When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, kernel_len, hidden_dim).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    batch_size, seq_len, hidden_dim = x.shape

    assert len(kernel.shape) == 3, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, batch_size), (
        f"Leading dimension must be 1 or batch_size ({batch_size}). Got kernel.shape={kernel.shape}."
    )

    _, kernel_len, _ = kernel.shape
    assert kernel_len <= 2 * seq_len, f"Kernel length must be less than or equal to 2 * seq_len. Got {kernel_len}."

    # IMPORTANT: The main difference between causal and non-causal FFT convolutions is the FFT length.
    # For causal FFT convolutions, we use fft_len = seq_len + kernel_len.
    # For non-causal FFT convolutions, we use fft_len = 2 * seq_len.
    fft_len = min(seq_len + kernel_len, 2 * seq_len)

    fft_x, fft_kernel = (
        torch.fft.rfft(x, n=fft_len, dim=1),
        torch.fft.rfft(kernel, n=fft_len, dim=1),
    )

    # 3. Apply the Convolution Theorem
    fft_x = fft_x * fft_kernel

    y = torch.fft.irfft(fft_x, n=fft_len, dim=1)[:, :seq_len, :]

    if shortcut is not None:
        assert shortcut.shape == (hidden_dim,)
        y.add_(rearrange(shortcut, "h -> 1 1 h") * x)
    return y


def fftconv1d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut. When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, kernel_len, hidden_dim).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    batch_size, seq_len, hidden_dim = x.shape

    assert len(kernel.shape) == 3, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, batch_size), (
        f"Leading dimension must be 1 or batch_size ({batch_size}). Got kernel.shape={kernel.shape}."
    )

    _, kernel_len, _ = kernel.shape
    assert kernel_len <= 2 * seq_len, f"Kernel length must be less than or equal to 2 * seq_len. Got {kernel_len}."

    # If the kernel is bigger than the input sequence, use fft_len = 2 * seq_len
    fft_len = min(seq_len + (kernel_len + 1) // 2, 2 * seq_len)

    fft_x, fft_kernel = (
        torch.fft.rfft(x, n=fft_len, dim=1),
        torch.fft.rfft(kernel, n=fft_len, dim=1),
    )

    # 3. Apply the Convolution Theorem
    fft_x = fft_x * fft_kernel

    crop_start = (kernel_len) // 2
    y = torch.fft.irfft(fft_x, n=fft_len, dim=1)[:, crop_start : crop_start + seq_len, :]

    if shortcut is not None:
        assert shortcut.shape == (hidden_dim,)
        y.add_(rearrange(shortcut, "h -> 1 1 h") * x)
    return y


def fftconv2d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut. When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, X_in, Y_in, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, K_x, K_y, hidden_dim).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, X_in, Y_in, hidden_dim).
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    B, X_in, Y_in, hidden_dim = x.shape

    assert len(kernel.shape) == 4, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )

    _, K_x, K_y, _ = kernel.shape

    assert K_x <= X_in * 2, f"Kernel size must be less than 2 * X_in. Got {K_x}."
    assert K_y <= Y_in * 2, f"Kernel size must be less than 2 * Y_in. Got {K_y}."
    assert hidden_dim == kernel.shape[-1], "Input and kernel must have the same number of channels (H)."

    # 1. Determine FFT size for linear convolution (same as 'same' version)
    fft_shape = (
        min(X_in + (K_x + 1) // 2, 2 * X_in),
        min(Y_in + (K_y + 1) // 2, 2 * Y_in),
    )

    # 2. Compute 2D FFT of the input and kernel
    fft_x = torch.fft.rfft2(x, s=fft_shape, dim=(1, 2))
    fft_kernel = torch.fft.rfft2(kernel, s=fft_shape, dim=(1, 2))

    # 3. Apply the Convolution Theorem
    fft_x = fft_x * fft_kernel

    crop_start_x = (K_x) // 2
    crop_start_y = (K_y) // 2

    # 4. Compute the inverse FFT to get the full convolution result
    # 5. Crop the result to the 'same' size
    # The output should have the same size as the input: (X_in, Y_in)
    # To achieve this, we crop from the full convolution result,
    # starting at an offset that centers the output.
    y = torch.fft.irfft2(fft_x, s=fft_shape, dim=(1, 2))[
        :, crop_start_x : crop_start_x + X_in, crop_start_y : crop_start_y + Y_in, :
    ]

    if shortcut is not None:
        assert shortcut.shape == (hidden_dim,)
        y.add_(rearrange(shortcut, "h -> 1 1 1 h") * x)

    return y


def fftconv3d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution with optional shortcut. When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, X_in, Y_in, Z_in, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, K_x, K_y, K_z, hidden_dim).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, X_in, Y_in, Z_in, hidden_dim).
    """
    B, X_in, Y_in, Z_in, hidden_dim = x.shape

    assert len(kernel.shape) == 5, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )

    _, K_x, K_y, K_z, _ = kernel.shape

    assert K_x <= X_in * 2, f"Kernel size must be less than 2 * X_in. Got {K_x}."
    assert K_y <= Y_in * 2, f"Kernel size must be less than 2 * Y_in. Got {K_y}."
    assert K_z <= Z_in * 2, f"Kernel size must be less than 2 * Z_in. Got {K_z}."
    assert hidden_dim == kernel.shape[-1], "Input and kernel must have the same number of channels (H)."

    # 1. Determine FFT size for linear convolution (same as 'same' version)
    fft_shape = (
        min(X_in + (K_x + 1) // 2, 2 * X_in),
        min(Y_in + (K_y + 1) // 2, 2 * Y_in),
        min(Z_in + (K_z + 1) // 2, 2 * Z_in),
    )

    # 2. Compute 3D FFT of the input and kernel
    fft_x = torch.fft.rfftn(x, s=fft_shape, dim=(1, 2, 3))
    fft_kernel = torch.fft.rfftn(kernel, s=fft_shape, dim=(1, 2, 3))

    # 3. Apply the Convolution Theorem
    fft_x = fft_x * fft_kernel

    crop_start_x = (K_x) // 2
    crop_start_y = (K_y) // 2
    crop_start_z = (K_z) // 2

    # 4. Compute the inverse FFT to get the full convolution result &
    # 5. Crop the result to the 'same' size
    # The output should have the same size as the input: (X_in, Y_in)
    # To achieve this, we crop from the full convolution result,
    # starting at an offset that centers the output.
    y = torch.fft.irfftn(fft_x, s=fft_shape, dim=(1, 2, 3))[
        :,
        crop_start_x : crop_start_x + X_in,
        crop_start_y : crop_start_y + Y_in,
        crop_start_z : crop_start_z + Z_in,
        :,
    ]

    if shortcut is not None:
        assert shortcut.shape == (hidden_dim,)
        y.add_(rearrange(shortcut, "h -> 1 1 1 1 h") * x)

    return y


def causal_fftconv1d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut, for inputs with layout (batch, length, hidden).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    This is a wrapper around fftconv1d_bhl that reshapes the input and kernel to (batch, hidden, length)
    and (1, hidden, kernel_len) respectively as our benchmarking results show that this is faster than processing
    with the original layout (batch, length, hidden) and (1, kernel_len, hidden) directly.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, kernel_len, hidden_dim).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    x = rearrange(x, "b l h -> b h l")
    kernel = rearrange(kernel, "b l h -> b h l")
    y = causal_fftconv1d_bhl(x, kernel, shortcut)
    return rearrange(y, "b h l -> b l h")


def fftconv1d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut, for inputs with layout (batch, length, hidden).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    This is a wrapper around fftconv1d_bhl that reshapes the input and kernel to (batch, hidden, length)
    and (1, hidden, kernel_len) respectively as our benchmarking results show that this is faster than processing
    with the original layout (batch, length, hidden) and (1, kernel_len, hidden) directly.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, kernel_len, hidden_dim).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
    """
    x = rearrange(x, "b l h -> b h l")
    kernel = rearrange(kernel, "b l h -> b h l")
    y = fftconv1d_bhl(x, kernel, shortcut)
    return rearrange(y, "b h l -> b l h")


def fftconv2d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut, for inputs with layout (batch, height, width, hidden).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    This is a wrapper around fftconv2d_bhl that reshapes the input and kernel to (batch, hidden, height, width)
    and (1, hidden, K_x, K_y) respectively as our benchmarking results show that this is faster than processing
    with the original layout (batch, height, width, hidden) and (1, K_x, K_y, hidden) directly.
    """
    x = rearrange(x, "b x y h -> b h x y")
    kernel = rearrange(kernel, "b x y h -> b h x y")
    y = fftconv2d_bhl(x, kernel, shortcut)
    return rearrange(y, "b h x y -> b x y h")


def fftconv3d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution with optional shortcut, for inputs with layout (batch, depth, height, width, hidden).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    This is a wrapper around fftconv3d_bhl that reshapes the input and kernel to (batch, hidden, depth, height, width)
    and (1, hidden, K_x, K_y, K_z) respectively as our benchmarking results show that this is faster than processing
    with the original layout (batch, depth, height, width, hidden) and (1, K_x, K_y, K_z, hidden) directly.
    """
    x = rearrange(x, "b x y z h -> b h x y z")
    kernel = rearrange(kernel, "b x y z h -> b h x y z")
    y = fftconv3d_bhl(x, kernel, shortcut)
    return rearrange(y, "b h x y z -> b x y z h")


###############################################################################
# BHL variants
###############################################################################


def causal_fftconv1d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut, for inputs with layout (batch, hidden, length).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, seq_len).
        kernel (torch.Tensor): Kernel tensor of shape (1, hidden_dim, kernel_len).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim, seq_len).
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    batch_size, hidden_dim, seq_len = x.shape

    assert len(kernel.shape) == 3, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, batch_size), (
        f"Leading dimension must be 1 or batch_size ({batch_size}). Got kernel.shape={kernel.shape}."
    )

    _, _, kernel_len = kernel.shape
    assert kernel_len <= 2 * seq_len, f"Kernel length must be less than or equal to 2 * seq_len. Got {kernel_len}."

    # If the kernel is bigger than the input sequence, use fft_len = 2 * seq_len
    fft_len = min(seq_len + kernel_len, 2 * seq_len)

    fft_x, fft_kernel = (
        torch.fft.rfft(x, n=fft_len, dim=2),
        torch.fft.rfft(kernel, n=fft_len, dim=2),
    )

    # Apply the Convolution Theorem
    fft_x = fft_x * fft_kernel

    y = torch.fft.irfft(fft_x, n=fft_len, dim=2)[..., :seq_len]

    if shortcut is not None:
        assert shortcut.shape == (hidden_dim,)
        y.add_(rearrange(shortcut, "h -> 1 h 1") * x)
    return y


def fftconv1d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut, for inputs with layout (batch, hidden, length).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, seq_len).
        kernel (torch.Tensor): Kernel tensor of shape (1, hidden_dim, kernel_len).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim, seq_len).
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    batch_size, hidden_dim, seq_len = x.shape

    assert len(kernel.shape) == 3, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, batch_size), (
        f"Leading dimension must be 1 or batch_size ({batch_size}). Got kernel.shape={kernel.shape}."
    )

    _, _, kernel_len = kernel.shape
    assert kernel_len <= 2 * seq_len, f"Kernel length must be less than or equal to 2 * seq_len. Got {kernel_len}."

    # If the kernel is bigger than the input sequence, use fft_len = 2 * seq_len
    fft_len = min(seq_len + (kernel_len + 1) // 2, 2 * seq_len)

    fft_x, fft_kernel = (
        torch.fft.rfft(x, n=fft_len, dim=2),
        torch.fft.rfft(kernel, n=fft_len, dim=2),
    )

    # Apply the Convolution Theorem
    fft_x = fft_x * fft_kernel

    crop_start = (kernel_len) // 2

    y = torch.fft.irfft(fft_x, n=fft_len, dim=2)[..., crop_start : crop_start + seq_len]

    if shortcut is not None:
        assert shortcut.shape == (hidden_dim,)
        y.add_(rearrange(shortcut, "h -> 1 h 1") * x)
    return y


def fftconv2d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut, for inputs with layout (batch, hidden, height, width).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, X_in, Y_in).
        kernel (torch.Tensor): Kernel tensor of shape (1, hidden_dim, K_x, K_y).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim, X_in, Y_in).
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    B, hidden_dim, X_in, Y_in = x.shape

    assert len(kernel.shape) == 4, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )

    _, _, K_x, K_y = kernel.shape

    assert K_x <= X_in * 2, f"Kernel size must be less than 2 * X_in. Got {K_x}."
    assert K_y <= Y_in * 2, f"Kernel size must be less than 2 * Y_in. Got {K_y}."
    assert hidden_dim == kernel.shape[1], "Input and kernel must have the same number of channels (H)."

    # 1. Determine FFT size for linear convolution (same as 'same' version)
    fft_shape = (
        min(X_in + (K_x + 1) // 2, 2 * X_in),
        min(Y_in + (K_y + 1) // 2, 2 * Y_in),
    )

    # 2. Compute 2D FFT of the input and kernel
    fft_x = torch.fft.rfft2(x, s=fft_shape, dim=(2, 3))
    fft_kernel = torch.fft.rfft2(kernel, s=fft_shape, dim=(2, 3))

    # 3. Apply the Convolution Theorem
    fft_x = fft_x * fft_kernel

    crop_start_x = (K_x) // 2
    crop_start_y = (K_y) // 2

    # 4. Compute the inverse FFT to get the full convolution result &
    # 5. Crop the result to the 'same' size
    # The output should have the same size as the input: (X_in, Y_in)
    # To achieve this, we crop from the full convolution result,
    # starting at an offset that centers the output.

    y = torch.fft.irfft2(fft_x, s=fft_shape, dim=(2, 3))[
        ..., crop_start_x : crop_start_x + X_in, crop_start_y : crop_start_y + Y_in
    ]

    if shortcut is not None:
        assert shortcut.shape == (hidden_dim,)
        y.add_(rearrange(shortcut, "h -> 1 h 1 1") * x)

    return y


def fftconv3d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution with optional shortcut, for inputs with layout (batch, hidden, depth, height, width).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, X_in, Y_in, Z_in).
        kernel (torch.Tensor): Kernel tensor of shape (1, hidden_dim, K_x, K_y, K_z).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim, X_in, Y_in, Z_in).
    """
    B, hidden_dim, X_in, Y_in, Z_in = x.shape

    assert len(kernel.shape) == 5, f"Unexpected kernel shape: {kernel.shape}."
    assert kernel.shape[0] in (1, B), (
        f"Leading dimension must be 1 or batch_size ({B}). Got kernel.shape={kernel.shape}."
    )

    _, _, K_x, K_y, K_z = kernel.shape

    assert K_x <= X_in * 2, f"Kernel size must be less than 2 * X_in. Got {K_x}."
    assert K_y <= Y_in * 2, f"Kernel size must be less than 2 * Y_in. Got {K_y}."
    assert K_z <= Z_in * 2, f"Kernel size must be less than 2 * Z_in. Got {K_z}."
    assert hidden_dim == kernel.shape[1], "Input and kernel must have the same number of channels (H)."

    # 1. Determine FFT size for linear convolution (same as 'same' version)
    fft_shape = (
        min(X_in + (K_x + 1) // 2, 2 * X_in),
        min(Y_in + (K_y + 1) // 2, 2 * Y_in),
        min(Z_in + (K_z + 1) // 2, 2 * Z_in),
    )

    # 2. Compute 3D FFT of the input and kernel
    fft_x = torch.fft.rfftn(x, s=fft_shape, dim=(2, 3, 4))
    fft_kernel = torch.fft.rfftn(kernel, s=fft_shape, dim=(2, 3, 4))

    # 3. Apply the Convolution Theorem
    fft_x = fft_x * fft_kernel

    crop_start_x = (K_x) // 2
    crop_start_y = (K_y) // 2
    crop_start_z = (K_z) // 2

    # 4. Compute the inverse FFT to get the full convolution result &
    # 5. Crop the result to the 'same' size
    # The output should have the same size as the input: (X_in, Y_in)
    # To achieve this, we crop from the full convolution result,
    # starting at an offset that centers the output.
    y = torch.fft.irfftn(fft_x, s=fft_shape, dim=(2, 3, 4))[
        :,
        :,
        crop_start_x : crop_start_x + X_in,
        crop_start_y : crop_start_y + Y_in,
        crop_start_z : crop_start_z + Z_in,
    ]

    if shortcut is not None:
        assert shortcut.shape == (hidden_dim,)
        y.add_(rearrange(shortcut, "h -> 1 h 1 1 1") * x)
    return y
