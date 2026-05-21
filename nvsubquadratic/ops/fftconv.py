# TODO: Add license header here


r"""FFT-based convolution operators (fp32) for 1D, 2D, and 3D signals.

Mathematical background
-----------------------
The discrete convolution of an input :math:`x` with a kernel :math:`k` is

.. math::
    y[n] = \sum_{m} x[n - m] \, k[m]

Computing this directly costs :math:`O(N \cdot K)` per channel (and per
spatial dimension), which becomes the bottleneck once the kernel grows large
relative to the input — the regime needed by Hyena-style models with global
("input-length") kernels. The **convolution theorem** lets us replace the
spatial product with an element-wise frequency-domain product:

.. math::
    y = \mathcal{F}^{-1}\bigl( \mathcal{F}(x) \odot \mathcal{F}(k) \bigr)

so the cost drops to :math:`O(N \log N)` per channel, independent of kernel
size. This is what makes Hyena and related sequence/spatial mixers
*subquadratic* even when their effective receptive field spans the whole
input.

These ops implement that convolution via PyTorch's real-input FFT
(:func:`torch.fft.rfft` / :func:`torch.fft.rfftn`), which exploits the
real-valued input to halve the memory of the frequency-domain tensors.

Linear vs. circular convolution
-------------------------------
A naive same-size FFT product gives *circular* convolution (kernel wraps
around the input boundary). To get the standard *linear* "same" output you
zero-pad both signals to a length :math:`F \ge N + K - 1`, multiply in
frequency, invert, and crop. The functions in this module use the smallest
such :math:`F`:

- Causal 1D: ``F = min(L + K, 2L)`` and crop the trailing ``L`` samples.
- Non-causal nD: ``F_d = min(N_d + ceil(K_d / 2), 2 N_d)`` per axis and crop
  the centered ``N_d`` samples.

The non-causal variant is cheaper because it needs less padding (the
"same" output only needs enough headroom for *half* the kernel on each side).

For *circular* convolutions (no wrap-around removal), see
:mod:`nvsubquadratic.ops.circular_fftconv`.

Layouts
-------
Two channel orderings are supported. Pick whichever matches your model:

- **BHL** (channels-first, ``[batch, hidden, * spatial_dims]``): standard for
  ``torch.nn.ConvNd``-style modules. **Faster** under the hood because the
  FFT runs on contiguous spatial axes without a transpose.
- **BLH** (channels-last, ``[batch, * spatial_dims, hidden]``): common in
  transformer-style code. The ``*_fp32_bhl_w_reshape`` wrappers transparently
  reshape BLH -> BHL -> BLH and are the recommended entry point for
  channels-last callers.

Families provided
-----------------
- 1D, causal and non-causal: ``[causal_]fftconv1d_fp32_{blh,bhl}[_w_reshape]``
- 2D, non-causal: ``fftconv2d_fp32_{blh,bhl}[_w_reshape]``
- 3D, non-causal: ``fftconv3d_fp32_{blh,bhl}[_w_reshape]``

Shape conventions
-----------------
- BHL kernels are ``[1|B, H, * K_dims]`` (channels first); BLH kernels are
  ``[1|B, * K_dims, H]``.
- A leading dim of ``1`` indicates a *shared* kernel across the batch (the
  standard depthwise case); a leading dim of ``B`` indicates a *per-sample*
  kernel (e.g. FiLM-conditioned Hyena, where each sample gets its own kernel).

Shortcut term
-------------
Optional ``shortcut: [H]`` adds a per-channel residual scale of the input:

.. math::
    y \leftarrow y + \text{shortcut} \odot x

broadcast along the spatial dimensions. This fuses the residual into the same
kernel launch and matches the algebra used by the multi-head FFT conv (see
:mod:`nvsubquadratic.ops.fftconv_multihead`) and by Hyena gating.

Precision
---------
All operators accept any input dtype. Internally ``x`` and ``kernel`` are
cast to ``float32`` for numerical stability (the frequency-domain product
amplifies the dynamic range of intermediate values); the output is returned
in the original dtype of ``x``. For aggressive memory/compute savings on
power-of-two spatial dims, see the fp16 counterparts in
:mod:`nvsubquadratic.ops.fftconv_fp16` and
:mod:`nvsubquadratic.ops.circular_fftconv_fp16`.

Performance
-----------
- For BLH inputs, prefer the ``*_fp32_bhl_w_reshape`` wrappers; benchmarks
  show they are consistently faster than operating directly in BLH layout
  because the FFT then runs on contiguous spatial axes.
- For memory-constrained workloads, see
  :mod:`nvsubquadratic.ops.fftconv_chunked` for channel-chunked variants.
"""

__all__ = [
    "causal_fftconv1d_fp32_bhl",
    "causal_fftconv1d_fp32_bhl_w_reshape",
    "causal_fftconv1d_fp32_blh",
    "fftconv1d_fp32_bhl",
    "fftconv1d_fp32_bhl_w_reshape",
    "fftconv1d_fp32_blh",
    "fftconv2d_fp32_bhl",
    "fftconv2d_fp32_bhl_w_reshape",
    "fftconv2d_fp32_blh",
    "fftconv3d_fp32_bhl",
    "fftconv3d_fp32_bhl_w_reshape",
    "fftconv3d_fp32_blh",
]

import torch
from einops import rearrange


# When True, use real-arithmetic complex multiply that is compilable by
# torch.compile / Inductor (Triton cannot codegen complex64 kernels).
# When False (default), use the faster in-place fft_x.mul_(fft_kernel).
COMPILE_COMPATIBLE = False


def _complex_mul_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise complex multiplication using only real arithmetic.

    Avoids complex64 intermediate tensors so ``torch.compile`` / Inductor can
    generate Triton kernels (Triton has no complex dtype support).

    Args:
        a: Complex tensor (output of rfft / rfftn).
        b: Complex tensor (output of rfft / rfftn), broadcastable with *a*.

    Returns:
        Complex tensor ``a * b`` computed via real-valued ops.
    """
    ar = torch.view_as_real(a)  # [..., 2]
    br = torch.view_as_real(b)  # [..., 2]
    real = ar[..., 0] * br[..., 0] - ar[..., 1] * br[..., 1]
    imag = ar[..., 0] * br[..., 1] + ar[..., 1] * br[..., 0]
    return torch.view_as_complex(torch.stack([real, imag], dim=-1))


###############################################################################
# BLH variants
###############################################################################


def causal_fftconv1d_fp32_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Causal 1D FFT convolution (BLH layout, channels-last) with optional shortcut.

    Computes :math:`y[n] = \sum_{m=0}^{n} x[n-m]\, k[m]` per channel via the
    FFT path:

    .. math::
        y = \mathcal{F}^{-1}\bigl(\mathcal{F}_F(x) \odot \mathcal{F}_F(k)\bigr)[\,:L\,]

    where :math:`F = \min(L + K, 2L)` is the zero-pad length that prevents
    wrap-around. Causality is enforced implicitly by keeping only the leading
    ``L`` samples of the inverse FFT (no future taps leak into position ``n``).

    When ``shortcut`` is provided, the per-channel residual is added:

    .. math::
        y \leftarrow y + \text{shortcut} \odot x

    Args:
        x: Input tensor of shape ``[batch_size, seq_len, hidden_dim]``.
        kernel: Kernel tensor of shape ``[1|B, kernel_len, hidden_dim]``. The
            leading dim is ``1`` for a shared kernel or ``B`` for FiLM-style
            per-sample kernels.
        shortcut: Optional ``[hidden_dim]`` per-channel residual scale.

    Returns:
        Output tensor of shape ``[batch_size, seq_len, hidden_dim]`` in the
        original dtype of ``x``.
    """
    x_fp32 = x.to(torch.float32)
    k_fp32 = kernel.to(torch.float32)

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
        torch.fft.rfft(x_fp32, n=fft_len, dim=1),
        torch.fft.rfft(k_fp32, n=fft_len, dim=1),
    )

    # 3. Apply the Convolution Theorem
    if COMPILE_COMPATIBLE:
        fft_x = _complex_mul_real(fft_x, fft_kernel)
    else:
        fft_x.mul_(fft_kernel)

    y = torch.fft.irfft(fft_x, n=fft_len, dim=1)[:, :seq_len, :]

    y = y.to(x.dtype)
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (hidden_dim,)
        y = y + rearrange(shortcut, "h -> 1 1 h") * x
    return y


def fftconv1d_fp32_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Non-causal 1D FFT convolution (BLH layout, channels-last) with optional shortcut.

    Computes a "same"-aligned linear convolution per channel by zero-padding
    to :math:`F = \min(L + \lceil K/2 \rceil, 2L)`, multiplying in the
    frequency domain, inverting, and cropping centered with offset ``K // 2``.
    The non-causal variant only needs enough headroom for half the kernel on
    each side, so it is cheaper than the causal variant.

    When ``shortcut`` is provided, the per-channel residual is added:

    .. math::
        y \leftarrow y + \text{shortcut} \odot x

    Args:
        x: Input tensor of shape ``[batch_size, seq_len, hidden_dim]``.
        kernel: Kernel tensor of shape ``[1|B, kernel_len, hidden_dim]``.
        shortcut: Optional ``[hidden_dim]`` per-channel residual scale.

    Returns:
        Output tensor of shape ``[batch_size, seq_len, hidden_dim]`` in the
        original dtype of ``x``.
    """
    x_fp32 = x.to(torch.float32)
    k_fp32 = kernel.to(torch.float32)

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
        torch.fft.rfft(x_fp32, n=fft_len, dim=1),
        torch.fft.rfft(k_fp32, n=fft_len, dim=1),
    )

    # 3. Apply the Convolution Theorem
    if COMPILE_COMPATIBLE:
        fft_x = _complex_mul_real(fft_x, fft_kernel)
    else:
        fft_x.mul_(fft_kernel)

    crop_start = (kernel_len) // 2
    y = torch.fft.irfft(fft_x, n=fft_len, dim=1)[:, crop_start : crop_start + seq_len, :]

    y = y.to(x.dtype)
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (hidden_dim,)
        y = y + rearrange(shortcut, "h -> 1 1 h") * x
    return y


def fftconv2d_fp32_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut. When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Accepts any input dtype. Internally casts ``x`` and ``kernel`` to float32 for
    numerical stability and returns the result in the original dtype of ``x``.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, X_in, Y_in, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, K_x, K_y, hidden_dim).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, X_in, Y_in, hidden_dim), in the original dtype of ``x``.
    """
    x_fp32 = x.to(torch.float32)
    k_fp32 = kernel.to(torch.float32)

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
    fft_x = torch.fft.rfft2(x_fp32, s=fft_shape, dim=(1, 2))
    fft_kernel = torch.fft.rfft2(k_fp32, s=fft_shape, dim=(1, 2))

    # 3. Apply the Convolution Theorem
    if COMPILE_COMPATIBLE:
        fft_x = _complex_mul_real(fft_x, fft_kernel)
    else:
        fft_x.mul_(fft_kernel)

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

    y = y.to(x.dtype)
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (hidden_dim,)
        y = y + rearrange(shortcut, "h -> 1 1 1 h") * x

    return y


def fftconv3d_fp32_blh(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution with optional shortcut. When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Accepts any input dtype. Internally casts ``x`` and ``kernel`` to float32 for
    numerical stability and returns the result in the original dtype of ``x``.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, X_in, Y_in, Z_in, hidden_dim).
        kernel (torch.Tensor): Kernel tensor of shape (1, K_x, K_y, K_z, hidden_dim).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, X_in, Y_in, Z_in, hidden_dim), in the original dtype of ``x``.
    """
    x_fp32 = x.to(torch.float32)
    k_fp32 = kernel.to(torch.float32)

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
    fft_x = torch.fft.rfftn(x_fp32, s=fft_shape, dim=(1, 2, 3))
    fft_kernel = torch.fft.rfftn(k_fp32, s=fft_shape, dim=(1, 2, 3))

    # 3. Apply the Convolution Theorem
    if COMPILE_COMPATIBLE:
        fft_x = _complex_mul_real(fft_x, fft_kernel)
    else:
        fft_x.mul_(fft_kernel)

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

    y = y.to(x.dtype)
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (hidden_dim,)
        y = y + rearrange(shortcut, "h -> 1 1 1 1 h") * x

    return y


def causal_fftconv1d_fp32_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut, for inputs with layout (batch, length, hidden).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    This is a wrapper around causal_fftconv1d_fp32_bhl that reshapes the input and kernel to (batch, hidden, length)
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
    y = causal_fftconv1d_fp32_bhl(x, kernel, shortcut)
    return rearrange(y, "b h l -> b l h")


def fftconv1d_fp32_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut, for inputs with layout (batch, length, hidden).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    This is a wrapper around fftconv1d_fp32_bhl that reshapes the input and kernel to (batch, hidden, length)
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
    y = fftconv1d_fp32_bhl(x, kernel, shortcut)
    return rearrange(y, "b h l -> b l h")


def fftconv2d_fp32_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut, for inputs with layout (batch, height, width, hidden).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    This is a wrapper around fftconv2d_fp32_bhl that reshapes the input and kernel to (batch, hidden, height, width)
    and (1, hidden, K_x, K_y) respectively as our benchmarking results show that this is faster than processing
    with the original layout (batch, height, width, hidden) and (1, K_x, K_y, hidden) directly.
    """
    x = rearrange(x, "b x y h -> b h x y")
    kernel = rearrange(kernel, "b x y h -> b h x y")
    y = fftconv2d_fp32_bhl(x, kernel, shortcut)
    return rearrange(y, "b h x y -> b x y h")


def fftconv3d_fp32_bhl_w_reshape(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution with optional shortcut, for inputs with layout (batch, depth, height, width, hidden).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    This is a wrapper around fftconv3d_fp32_bhl that reshapes the input and kernel to (batch, hidden, depth, height, width)
    and (1, hidden, K_x, K_y, K_z) respectively as our benchmarking results show that this is faster than processing
    with the original layout (batch, depth, height, width, hidden) and (1, K_x, K_y, K_z, hidden) directly.
    """
    x = rearrange(x, "b x y z h -> b h x y z")
    kernel = rearrange(kernel, "b x y z h -> b h x y z")
    y = fftconv3d_fp32_bhl(x, kernel, shortcut)
    return rearrange(y, "b h x y z -> b x y z h")


###############################################################################
# BHL variants
###############################################################################


def causal_fftconv1d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut, for inputs with layout (batch, hidden, length).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Accepts any input dtype. Internally casts ``x`` and ``kernel`` to float32 for
    numerical stability and returns the result in the original dtype of ``x``.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, seq_len).
        kernel (torch.Tensor): Kernel tensor of shape (1, hidden_dim, kernel_len).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim, seq_len), in the original dtype of ``x``.
    """
    x_fp32 = x.to(torch.float32)
    k_fp32 = kernel.to(torch.float32)

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
        torch.fft.rfft(x_fp32, n=fft_len, dim=2),
        torch.fft.rfft(k_fp32, n=fft_len, dim=2),
    )

    # Apply the Convolution Theorem
    if COMPILE_COMPATIBLE:
        fft_x = _complex_mul_real(fft_x, fft_kernel)
    else:
        fft_x.mul_(fft_kernel)

    y = torch.fft.irfft(fft_x, n=fft_len, dim=2)[..., :seq_len]

    y = y.to(x.dtype)
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (hidden_dim,)
        y = y + rearrange(shortcut, "h -> 1 h 1") * x
    return y


def fftconv1d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """1D FFT convolution with optional shortcut, for inputs with layout (batch, hidden, length).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Accepts any input dtype. Internally casts ``x`` and ``kernel`` to float32 for
    numerical stability and returns the result in the original dtype of ``x``.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, seq_len).
        kernel (torch.Tensor): Kernel tensor of shape (1, hidden_dim, kernel_len).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim, seq_len), in the original dtype of ``x``.
    """
    x_fp32 = x.to(torch.float32)
    k_fp32 = kernel.to(torch.float32)

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
        torch.fft.rfft(x_fp32, n=fft_len, dim=2),
        torch.fft.rfft(k_fp32, n=fft_len, dim=2),
    )

    # Apply the Convolution Theorem
    if COMPILE_COMPATIBLE:
        fft_x = _complex_mul_real(fft_x, fft_kernel)
    else:
        fft_x.mul_(fft_kernel)

    crop_start = (kernel_len) // 2

    y = torch.fft.irfft(fft_x, n=fft_len, dim=2)[..., crop_start : crop_start + seq_len]

    y = y.to(x.dtype)
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (hidden_dim,)
        y = y + rearrange(shortcut, "h -> 1 h 1") * x
    return y


def fftconv2d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """2D FFT convolution with optional shortcut, for inputs with layout (batch, hidden, height, width).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Accepts any input dtype. Internally casts ``x`` and ``kernel`` to float32 for
    numerical stability and returns the result in the original dtype of ``x``.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, X_in, Y_in).
        kernel (torch.Tensor): Kernel tensor of shape (1, hidden_dim, K_x, K_y).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim, X_in, Y_in), in the original dtype of ``x``.
    """
    x_fp32 = x.to(torch.float32)
    k_fp32 = kernel.to(torch.float32)

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
    fft_x = torch.fft.rfft2(x_fp32, s=fft_shape, dim=(2, 3))
    fft_kernel = torch.fft.rfft2(k_fp32, s=fft_shape, dim=(2, 3))

    # 3. Apply the Convolution Theorem
    if COMPILE_COMPATIBLE:
        fft_x = _complex_mul_real(fft_x, fft_kernel)
    else:
        fft_x.mul_(fft_kernel)

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

    y = y.to(x.dtype)
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (hidden_dim,)
        y = y + rearrange(shortcut, "h -> 1 h 1 1") * x

    return y


def fftconv3d_fp32_bhl(
    x: torch.Tensor,
    kernel: torch.Tensor,
    shortcut: torch.Tensor | None = None,
) -> torch.Tensor:
    """3D FFT convolution with optional shortcut, for inputs with layout (batch, hidden, depth, height, width).

    When shortcut provided, then the output is given by shortcut(x) + conv(x, kernel).

    Accepts any input dtype. Internally casts ``x`` and ``kernel`` to float32 for
    numerical stability and returns the result in the original dtype of ``x``.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim, X_in, Y_in, Z_in).
        kernel (torch.Tensor): Kernel tensor of shape (1, hidden_dim, K_x, K_y, K_z).
        shortcut (torch.Tensor | None, optional): Optional shortcut tensor of shape (hidden_dim). Defaults to None.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim, X_in, Y_in, Z_in), in the original dtype of ``x``.
    """
    x_fp32 = x.to(torch.float32)
    k_fp32 = kernel.to(torch.float32)

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
    fft_x = torch.fft.rfftn(x_fp32, s=fft_shape, dim=(2, 3, 4))
    fft_kernel = torch.fft.rfftn(k_fp32, s=fft_shape, dim=(2, 3, 4))

    # 3. Apply the Convolution Theorem
    if COMPILE_COMPATIBLE:
        fft_x = _complex_mul_real(fft_x, fft_kernel)
    else:
        fft_x.mul_(fft_kernel)

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

    y = y.to(x.dtype)
    if shortcut is not None:
        assert shortcut.dtype == x.dtype, f"shortcut.dtype ({shortcut.dtype}) must match x.dtype ({x.dtype})"
        assert shortcut.shape == (hidden_dim,)
        y = y + rearrange(shortcut, "h -> 1 h 1 1 1") * x
    return y
