# TODO: Add license header here


"""Custom CUDA FFT convolution operators for 2D signals.

This module provides wrapper functions around the optimized CUDA kernels from
:mod:`subquadratic_ops_torch`. It mirrors the API of :mod:`nvsubquadratic.ops.fftconv`
for 2D operators while delegating the heavy lifting to the custom kernel.

The custom kernel performs LINEAR convolution (not circular), equivalent to:
    xf = torch.fft.rfft2(x, s=(2*X, 2*Y))
    wf = torch.fft.rfft2(weight, s=(2*X, 2*Y))
    y = irfft2(xf * wf)[..., X//2:X//2+X, Y//2:Y//2+Y]

Families provided
-----------------
- 2D convolutions with optional per-channel shortcut
  - BLH: ``fftconv2d_blh``
  - BHL: ``fftconv2d_bhl``
  - Wrapper: ``fftconv2d_bhl_w_reshape``

Shapes and conventions
----------------------
- BLH inputs and kernels:
  - 2D: ``x: [B, X_in, Y_in, H]``, ``kernel: [1, K_x, K_y, H]``
- BHL inputs and kernels:
  - 2D: ``x: [B, H, X_in, Y_in]``, ``kernel: [1, H, K_x, K_y]``

Shortcuts and dtype
-------------------
- Optional ``shortcut: [H]`` scales the input per-channel and is added to the
  convolution output: ``y += shortcut * x`` (broadcasted along spatial dims).
- All operators expect ``float32`` inputs, kernels, and shortcut.

Limitations
-----------
- Requires CUDA tensors.
- Only supports shared kernels (kernel.shape[0] == 1).
- **Kernel spatial dimensions must equal input spatial dimensions** (full-size kernels).
"""

__all__ = [
    "fftconv2d_bhl",
    "fftconv2d_bhl_w_reshape",
    "fftconv2d_blh",
]

import torch
from einops import rearrange
from subquadratic_ops_torch.fft_conv2d import fft_conv2d


###############################################################################
# BHL variants
###############################################################################


def fftconv2d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut, for inputs with layout (batch, hidden, height, width).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Note: The custom CUDA kernel requires kernel spatial dimensions to match the input
    spatial dimensions exactly. Use `grid_type='single'` when generating kernels.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, X_in, Y_in).
        kernel (torch.Tensor): Kernel tensor of shape (1, hidden_dim, X_in, Y_in).
            Note: K_x must equal X_in and K_y must equal Y_in.
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
    assert kernel.shape[0] == 1, "Custom CUDA kernel only supports shared kernels (kernel.shape[0] == 1)."

    _, _, K_x, K_y = kernel.shape

    # Custom kernel requires kernel spatial dims to match input spatial dims
    assert K_x == X_in, f"Kernel K_x must equal X_in. Got K_x={K_x}, X_in={X_in}."
    assert K_y == Y_in, f"Kernel K_y must equal Y_in. Got K_y={K_y}, Y_in={Y_in}."
    assert hidden_dim == kernel.shape[1], "Input and kernel must have the same number of channels (H)."
    assert x.is_cuda, "Custom CUDA kernel requires CUDA tensors."

    # 1. Extract weight tensor with shape expected by custom kernel: (hidden_dim, K_x, K_y)
    weight = kernel[0]  # [hidden_dim, K_x, K_y]

    # 2. Apply the custom CUDA FFT convolution kernel
    y = fft_conv2d(x.contiguous(), weight.contiguous())
    assert y.shape == x.shape, f"Kernel returned shape {y.shape}, expected {x.shape}."

    # 3. Add shortcut if provided
    if shortcut is not None:
        assert shortcut.shape == (hidden_dim,)
        y = y + rearrange(shortcut, "h -> 1 h 1 1") * x

    return y


###############################################################################
# BLH variants and wrappers
###############################################################################


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

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, X_in, Y_in, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, X_in, Y_in, hidden_dim).
            Note: K_x must equal X_in and K_y must equal Y_in.
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, X_in, Y_in, hidden_dim).
    """
    x = rearrange(x, "b x y h -> b h x y")
    kernel = rearrange(kernel, "b x y h -> b h x y")
    y = fftconv2d_bhl(x, kernel, shortcut)
    return rearrange(y, "b h x y -> b x y h")


def fftconv2d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut, for inputs with layout (batch, height, width, hidden).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, X_in, Y_in, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, X_in, Y_in, hidden_dim).
            Note: K_x must equal X_in and K_y must equal Y_in.
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, X_in, Y_in, hidden_dim).
    """
    return fftconv2d_bhl_w_reshape(x, kernel, shortcut)
