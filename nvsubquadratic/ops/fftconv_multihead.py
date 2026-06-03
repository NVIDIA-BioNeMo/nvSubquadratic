# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Multi-head 2D FFT convolution operators with dense within-head channel mixing.

Motivation
----------
The standard FFT convolutions in :mod:`nvsubquadratic.ops.fftconv` are
*depthwise*: each output channel is the convolution of a single input
channel with its own kernel, i.e. the kernel has shape ``[H, K_x, K_y]`` and
the frequency-domain product is element-wise across channels. This is fast
but lets no information flow between channels — channel mixing has to be
done by a separate pointwise layer (typically a 1x1 conv or MLP).

The **multi-head** variant in this module bundles depthwise spatial mixing
and dense channel mixing into a single op, in the spirit of multi-head
attention:

- Channels are split into ``num_heads`` groups of ``head_dim`` each.
- Within a head, the convolution is *dense* across channels: every output
  channel sees every input channel through its own learned 2D kernel.
- Across heads, channels remain isolated (no cross-head mixing).

Frequency-domain operation
--------------------------
Concretely, if :math:`\hat{x} \in \mathbb{C}^{B \times n \times d \times F_x \times F_y}`
is the rFFT of the input (with :math:`d = \text{head\_dim}`,
:math:`n = \text{num\_heads}`), and
:math:`\hat{K} \in \mathbb{C}^{n \times d_o \times d_i \times F_x \times F_y}`
is the rFFT of the kernel, then per spatial frequency bin :math:`(f_x, f_y)`
each head applies a dense :math:`d_o \times d_i` matrix to the input vector:

.. math::
    \hat{y}_{b, n, o, f_x, f_y}
        = \sum_{i} \hat{K}_{n, o, i, f_x, f_y} \, \hat{x}_{b, n, i, f_x, f_y}

Implemented as a single :func:`torch.einsum` over the frequency-domain
tensors. The inverse rFFT then materialises the spatial output.

Low-rank variant
----------------
For large ``head_dim``, the dense kernel ``[d_o, d_i, K_x, K_y]`` has
:math:`d^2 K_x K_y` parameters and a matching memory/compute cost in
frequency domain. The low-rank functions factor the kernel as
:math:`K = U V` with rank ``r < d``:

.. math::
    \hat{y} = \hat{U} \,(\hat{V} \hat{x})

This drops the per-frequency-bin cost from :math:`O(d^2)` to :math:`O(2 d r)`
and the parameter count from :math:`d^2 K_x K_y` to :math:`2 d r K_x K_y`,
without materialising the full :math:`d \times d` kernel spectrum.

Shape conventions
-----------------
- Input: ``x: [B, num_heads, head_dim, H, W]``
- Dense kernel: ``[num_heads, head_dim_out, head_dim_in, K_x, K_y]``
- Low-rank factors: ``U: [n, d, r, K_x, K_y]``, ``V: [n, r, d, K_x, K_y]``
- Output: ``[B, num_heads, head_dim, H, W]``

Linear vs. circular
-------------------
Functions without the ``_circular`` suffix produce same-aligned *linear*
convolutions (zero-padded FFTs with crop). The ``_circular_*`` variants
compute the periodic convolution at the same spatial size (no padding,
no crop) and expect ``K_x == H``, ``K_y == W``.
"""

__all__ = [
    "fftconv2d_multihead_bhl",
    "fftconv2d_multihead_circular_bhl",
    "fftconv2d_multihead_lowrank_bhl",
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

    _B, num_heads, head_dim, H, W = x.shape
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
    _B, num_heads, head_dim, H, W = x.shape
    _, _, _rank, K_x, K_y = kernel_u.shape

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

    _B, num_heads, head_dim, H, W = x.shape

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
    _B, num_heads, head_dim, H, W = x.shape

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
