# TODO: Add license header here

"""Wrappers around the custom CUDA FFT convolution kernels.

This module mirrors the API of :mod:`nvsubquadratic.ops.fftconv` for the 2D
operators while delegating the heavy lifting to the optimized kernel provided
by :mod:`subquadratic_ops_torch`. The intent is to be a drop-in replacement
that preserves shapes, dtype checks, and shortcut semantics.
"""

from __future__ import annotations


__all__ = [
    "fftconv1d_bhl",
    "fftconv1d_bhl_w_reshape",
    "fftconv1d_blh",
    "fftconv2d_bhl",
    "fftconv2d_bhl_w_reshape",
    "fftconv2d_blh",
]

import torch
import torch.nn.functional as F
from einops import rearrange


try:
    from subquadratic_ops_torch.fft_causal_conv1d import fft_causal_conv1d
    from subquadratic_ops_torch.fft_conv2d import fft_conv2d
except ImportError as _custom_import_error:
    fft_causal_conv1d = None  # type: ignore[assignment]
    fft_conv2d = None  # type: ignore[assignment]
else:
    _custom_import_error = None


def _validate_float32_tensor(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None:
        return
    assert tensor.dtype == torch.float32, f"{name} must be float32. Current dtype: {tensor.dtype}"


def _ensure_custom_available() -> None:
    if _custom_import_error is not None:
        raise ImportError(
            "subquadratic_ops_torch is required for custom FFT kernels; install the provided wheel to enable this path."
        ) from _custom_import_error


def fftconv1d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut for BHL layout (B, H, L)."""
    _ensure_custom_available()
    _validate_float32_tensor("x", x)
    _validate_float32_tensor("kernel", kernel)
    _validate_float32_tensor("shortcut", shortcut)

    assert x.ndim == 3, f"Expected x with 3 dims (B, H, L). Got {x.shape}."
    assert kernel.ndim == 3, f"Expected kernel with 3 dims (1|B, H, K). Got {kernel.shape}."
    _batch, H, _length = x.shape
    assert x.is_cuda, "Custom CUDA kernel requires CUDA tensors."
    assert kernel.shape[0] == 1, "Custom CUDA kernel only supports shared kernels (kernel.shape[0] == 1)."
    _, H_k, _kernel_len = kernel.shape
    assert H_k == H, "Input and kernel must have the same number of channels (H)."

    weight = kernel[0]  # (H, K)
    y = fft_causal_conv1d(x.contiguous(), weight.contiguous())
    assert y.shape == x.shape, f"Kernel returned shape {y.shape}, expected {x.shape}."

    if shortcut is not None:
        assert shortcut.shape == (H,)
        y = y.add(rearrange(shortcut, "h -> 1 h 1") * x)
    return y


def fftconv1d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut for BLH layout (B, L, H)."""
    x_bhl = rearrange(x, "b l h -> b h l")
    kernel_bhl = rearrange(kernel, "b k h -> b h k")
    y_bhl = fftconv1d_bhl(x_bhl, kernel_bhl, shortcut)
    return rearrange(y_bhl, "b h l -> b l h")


def fftconv1d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Alias for fftconv1d_bhl_w_reshape to match nvsubquadratic.ops.fftconv API."""
    return fftconv1d_bhl_w_reshape(x, kernel, shortcut)


def fftconv2d_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut for BHL layout (B, H, X, Y)."""
    _ensure_custom_available()
    _validate_float32_tensor("x", x)
    _validate_float32_tensor("kernel", kernel)
    _validate_float32_tensor("shortcut", shortcut)

    assert x.ndim == 4, f"Expected x with 4 dims (B, H, X, Y). Got {x.shape}."
    assert kernel.ndim == 4, f"Expected kernel with 4 dims (1|B, H, K_x, K_y). Got {kernel.shape}."
    _batch, H, X_in, Y_in = x.shape
    assert x.is_cuda, "Custom CUDA kernel requires CUDA tensors."
    assert kernel.shape[0] == 1, "Custom CUDA kernel only supports shared kernels (kernel.shape[0] == 1)."
    _, H_k, K_x, K_y = kernel.shape
    assert H_k == H, "Input and kernel must have the same number of channels (H)."
    assert K_x <= X_in and K_y <= Y_in, (
        "Custom CUDA kernel expects kernel spatial dims <= input; use grid_type='single' so they match."
    )
    pad_x = X_in - K_x
    pad_y = Y_in - K_y
    if pad_x or pad_y:
        pad_x_before = pad_x // 2
        pad_x_after = pad_x - pad_x_before
        pad_y_before = pad_y // 2
        pad_y_after = pad_y - pad_y_before
        kernel = F.pad(kernel, (pad_y_before, pad_y_after, pad_x_before, pad_x_after))

    weight = kernel[0]
    y = fft_conv2d(x.contiguous(), weight.contiguous())
    assert y.shape == x.shape, f"Kernel returned shape {y.shape}, expected {x.shape}."

    if shortcut is not None:
        assert shortcut.shape == (H,)
        y = y.add(rearrange(shortcut, "h -> 1 h 1 1") * x)
    return y


def fftconv2d_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut for BLH layout (B, X, Y, H)."""
    x_bhl = rearrange(x, "b x y h -> b h x y")
    kernel_bhl = rearrange(kernel, "b x y h -> b h x y")
    y_bhl = fftconv2d_bhl(x_bhl, kernel_bhl, shortcut)
    return rearrange(y_bhl, "b h x y -> b x y h")


def fftconv2d_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """Alias for fftconv2d_bhl_w_reshape to match nvsubquadratic.ops.fftconv API."""
    return fftconv2d_bhl_w_reshape(x, kernel, shortcut)
