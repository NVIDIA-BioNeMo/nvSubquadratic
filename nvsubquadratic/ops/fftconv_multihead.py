# TODO: Add license header here


"""Multi-head FFT convolution operators for 2D signals.

This module provides FFT convolutions with dense within-head channel mixing,
analogous to multi-head attention but for convolutions.

Key difference from standard (depthwise) FFT conv:
- Depthwise: kernel shape [1, H, K_x, K_y], element-wise multiply in freq domain
- Multi-head: kernel shape [num_heads, head_dim, head_dim, K_x, K_y], dense matmul within heads

Shapes:
- Input x: [B, num_heads, head_dim, H, W]
- Kernel: [num_heads, head_dim_out, head_dim_in, K_x, K_y]
- Output: [B, num_heads, head_dim, H, W]

The convolution applies dense channel mixing within each head independently,
enabling cross-channel learning while maintaining head isolation.
"""

__all__ = [
    "fftconv2d_multihead_bhl",
    "fftconv2d_multihead_lowrank_bhl",
    "fftconv2d_multihead_circular_bhl",
    "fftconv2d_multihead_lowrank_circular_bhl",
]

import torch


def fftconv2d_multihead_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with dense within-head channel mixing.

    Applies a dense [head_dim x head_dim] convolution within each head,
    similar to how attention mixes across positions but here mixing channels.

    Args:
        x: Input tensor [B, num_heads, head_dim, H, W], float32
        kernel: Kernel tensor [num_heads, head_dim_out, head_dim_in, K_x, K_y], float32
        shortcut: Optional per-channel scale [num_heads * head_dim], float32

    Returns:
        Output tensor [B, num_heads, head_dim, H, W]
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"
    if shortcut is not None:
        assert shortcut.dtype == torch.float32, f"shortcut must be float32. Current dtype: {shortcut.dtype}"

    B, num_heads, head_dim, H, W = x.shape
    N, head_dim_out, head_dim_in, K_x, K_y = kernel.shape

    assert N == num_heads, f"Kernel num_heads ({N}) must match input ({num_heads})"
    assert head_dim_out == head_dim_in == head_dim, (
        f"Kernel head_dims ({head_dim_out}, {head_dim_in}) must match input head_dim ({head_dim})"
    )
    assert K_x <= H * 2, f"Kernel size K_x ({K_x}) must be <= 2 * H ({2 * H})"
    assert K_y <= W * 2, f"Kernel size K_y ({K_y}) must be <= 2 * W ({2 * W})"

    # Determine FFT size for linear convolution
    fft_h = min(H + (K_x + 1) // 2, 2 * H)
    fft_w = min(W + (K_y + 1) // 2, 2 * W)

    # FFT of input: [B, num_heads, head_dim, fft_h, fft_w//2+1]
    x_fft = torch.fft.rfft2(x, s=(fft_h, fft_w))

    # FFT of kernel: [num_heads, head_dim_out, head_dim_in, fft_h, fft_w//2+1]
    k_fft = torch.fft.rfft2(kernel, s=(fft_h, fft_w))

    # Dense conv within heads using einsum
    # x_fft: [B, num_heads, head_dim_in, fft_h, fft_w//2+1]
    # k_fft: [num_heads, head_dim_out, head_dim_in, fft_h, fft_w//2+1]
    # out_fft: [B, num_heads, head_dim_out, fft_h, fft_w//2+1]
    out_fft = torch.einsum("bnihw,noihw->bnohw", x_fft, k_fft)

    # Crop after inverse FFT
    crop_h = K_x // 2
    crop_w = K_y // 2

    # Inverse FFT and crop
    out_full = torch.fft.irfft2(out_fft, s=(fft_h, fft_w))
    out = out_full[..., crop_h : crop_h + H, crop_w : crop_w + W]

    # Add shortcut if provided
    if shortcut is not None:
        # shortcut: [num_heads * head_dim] -> [1, num_heads, head_dim, 1, 1]
        shortcut_reshaped = shortcut.view(1, num_heads, head_dim, 1, 1)
        out = out + x * shortcut_reshaped

    return out


def fftconv2d_multihead_lowrank_bhl(
    x: torch.Tensor,
    kernel_u: torch.Tensor,
    kernel_v: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with low-rank within-head channel mixing.

    Instead of a full [head_dim x head_dim] kernel, uses a rank-r factorization:
        K = U @ V^T
    where U: [num_heads, head_dim, rank, K_x, K_y] and
          V: [num_heads, rank, head_dim, K_x, K_y].

    The convolution becomes two smaller matmuls in the frequency domain:
        z = V @ x  (sum over head_dim_in, output rank)
        y = U @ z  (sum over rank, output head_dim_out)

    Args:
        x: Input tensor [B, num_heads, head_dim, H, W], float32
        kernel_u: U factor [num_heads, head_dim, rank, K_x, K_y], float32
        kernel_v: V factor [num_heads, rank, head_dim, K_x, K_y], float32
        shortcut: Optional per-channel scale [num_heads * head_dim], float32

    Returns:
        Output tensor [B, num_heads, head_dim, H, W]
    """
    B, num_heads, head_dim, H, W = x.shape
    _, _, rank, K_x, K_y = kernel_u.shape

    # Determine FFT size for linear convolution
    fft_h = min(H + (K_x + 1) // 2, 2 * H)
    fft_w = min(W + (K_y + 1) // 2, 2 * W)

    # FFT of input and kernel factors
    x_fft = torch.fft.rfft2(x, s=(fft_h, fft_w))
    u_fft = torch.fft.rfft2(kernel_u, s=(fft_h, fft_w))
    v_fft = torch.fft.rfft2(kernel_v, s=(fft_h, fft_w))

    # Two-step low-rank conv (equivalent to K_fft @ x_fft where K_fft = U_fft @ V_fft,
    # but avoids materializing the full [head_dim x head_dim] K_fft).
    # Step 1: z = V @ x — contract over head_dim_in
    # x_fft: [B, n, head_dim, fft_h, fft_w'], v_fft: [n, rank, head_dim, fft_h, fft_w']
    z_fft = torch.einsum("bnihw,nrihw->bnrhw", x_fft, v_fft)

    # Step 2: y = U @ z — contract over rank
    # z_fft: [B, n, rank, fft_h, fft_w'], u_fft: [n, head_dim, rank, fft_h, fft_w']
    out_fft = torch.einsum("bnrhw,norhw->bnohw", z_fft, u_fft)

    # Crop after inverse FFT
    crop_h = K_x // 2
    crop_w = K_y // 2
    out_full = torch.fft.irfft2(out_fft, s=(fft_h, fft_w))
    out = out_full[..., crop_h : crop_h + H, crop_w : crop_w + W]

    if shortcut is not None:
        shortcut_reshaped = shortcut.view(1, num_heads, head_dim, 1, 1)
        out = out + x * shortcut_reshaped

    return out


def fftconv2d_multihead_circular_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D circular FFT convolution with dense within-head channel mixing.

    Same as fftconv2d_multihead_bhl but uses circular (periodic) convolution.
    Kernel size should equal input size for circular conv.

    Args:
        x: Input tensor [B, num_heads, head_dim, H, W], float32
        kernel: Kernel tensor [num_heads, head_dim_out, head_dim_in, H, W], float32
        shortcut: Optional per-channel scale [num_heads * head_dim], float32

    Returns:
        Output tensor [B, num_heads, head_dim, H, W]
    """
    assert x.dtype == torch.float32, f"x must be float32. Current dtype: {x.dtype}"
    assert kernel.dtype == torch.float32, f"kernel must be float32. Current dtype: {kernel.dtype}"

    B, num_heads, head_dim, H, W = x.shape

    # FFT of input and kernel (circular: no padding needed)
    x_fft = torch.fft.rfft2(x)
    k_fft = torch.fft.rfft2(kernel, s=(H, W))

    # Dense conv within heads
    out_fft = torch.einsum("bnihw,noihw->bnohw", x_fft, k_fft)

    # Inverse FFT (no cropping for circular)
    out = torch.fft.irfft2(out_fft, s=(H, W))

    # Add shortcut if provided
    if shortcut is not None:
        shortcut_reshaped = shortcut.view(1, num_heads, head_dim, 1, 1)
        out = out + x * shortcut_reshaped

    return out


def fftconv2d_multihead_lowrank_circular_bhl(
    x: torch.Tensor,
    kernel_u: torch.Tensor,
    kernel_v: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D circular FFT convolution with low-rank within-head channel mixing.

    Circular (periodic) variant of fftconv2d_multihead_lowrank_bhl.

    Args:
        x: Input tensor [B, num_heads, head_dim, H, W], float32
        kernel_u: U factor [num_heads, head_dim, rank, H, W], float32
        kernel_v: V factor [num_heads, rank, head_dim, H, W], float32
        shortcut: Optional per-channel scale [num_heads * head_dim], float32

    Returns:
        Output tensor [B, num_heads, head_dim, H, W]
    """
    B, num_heads, head_dim, H, W = x.shape

    x_fft = torch.fft.rfft2(x)
    u_fft = torch.fft.rfft2(kernel_u, s=(H, W))
    v_fft = torch.fft.rfft2(kernel_v, s=(H, W))

    # Two-step low-rank conv (avoids materializing full [head_dim x head_dim] K_fft)
    z_fft = torch.einsum("bnihw,nrihw->bnrhw", x_fft, v_fft)
    out_fft = torch.einsum("bnrhw,norhw->bnohw", z_fft, u_fft)

    out = torch.fft.irfft2(out_fft, s=(H, W))

    if shortcut is not None:
        shortcut_reshaped = shortcut.view(1, num_heads, head_dim, 1, 1)
        out = out + x * shortcut_reshaped

    return out
