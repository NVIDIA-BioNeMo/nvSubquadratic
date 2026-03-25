# TODO: Add license header here


"""CKConv (long-convolution) implementation for ND signals."""

import math
from typing import Literal

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate

# Standard FFT convolutions
from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_bhl,
    circular_fftconv1d_bhl_w_reshape,
    circular_fftconv2d_bhl,
    circular_fftconv2d_bhl_w_reshape,
    circular_fftconv3d_bhl,
    circular_fftconv3d_bhl_w_reshape,
)
from nvsubquadratic.ops.fftconv import (
    causal_fftconv1d_fp32_bhl,
    causal_fftconv1d_fp32_bhl_w_reshape,
    fftconv1d_fp32_bhl,
    fftconv1d_fp32_bhl_w_reshape,
    fftconv2d_fp32_bhl,
    fftconv2d_fp32_bhl_w_reshape,
    fftconv3d_fp32_bhl,
    fftconv3d_fp32_bhl_w_reshape,
)

# Chunked (memory-efficient) variants for zero-padded and causal convolutions
# Note: circular convolutions don't have chunked variants (lower memory overhead already)
from nvsubquadratic.ops.fftconv_chunked import (
    causal_fftconv1d_fp32_bhl as causal_fftconv1d_fp32_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    causal_fftconv1d_fp32_bhl_w_reshape as causal_fftconv1d_fp32_bhl_w_reshape_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv1d_fp32_bhl as fftconv1d_fp32_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv1d_fp32_bhl_w_reshape as fftconv1d_fp32_bhl_w_reshape_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv2d_fp32_bhl as fftconv2d_fp32_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv2d_fp32_bhl_w_reshape as fftconv2d_fp32_bhl_w_reshape_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv3d_fp32_bhl as fftconv3d_fp32_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv3d_fp32_bhl_w_reshape as fftconv3d_fp32_bhl_w_reshape_chunked,
)

# FP16 FFT convolutions (power-of-2 padding + ortho normalization)
from nvsubquadratic.ops.fftconv_fp16 import (
    causal_fftconv1d_fp16_bhl,
    causal_fftconv1d_fp16_bhl_chunked,
    causal_fftconv1d_fp16_bhl_w_reshape,
    causal_fftconv1d_fp16_bhl_w_reshape_chunked,
    fftconv1d_fp16_bhl,
    fftconv1d_fp16_bhl_chunked,
    fftconv1d_fp16_bhl_w_reshape,
    fftconv1d_fp16_bhl_w_reshape_chunked,
    fftconv2d_fp16_bhl,
    fftconv2d_fp16_bhl_chunked,
    fftconv2d_fp16_bhl_w_reshape,
    fftconv2d_fp16_bhl_w_reshape_chunked,
    fftconv3d_fp16_bhl,
    fftconv3d_fp16_bhl_chunked,
    fftconv3d_fp16_bhl_w_reshape,
    fftconv3d_fp16_bhl_w_reshape_chunked,
)


# Mapping from padding mode and data dimensionality to FFT convolution functions.
# Each entry is a tuple: (fn_for_BLH_input (bhl + reshape), fn_for_BHL_input)
FFT_FUNCTIONS = {
    "circular": {
        1: (circular_fftconv1d_bhl_w_reshape, circular_fftconv1d_bhl),
        2: (circular_fftconv2d_bhl_w_reshape, circular_fftconv2d_bhl),
        3: (circular_fftconv3d_bhl_w_reshape, circular_fftconv3d_bhl),
    },
    "zero": {
        1: (fftconv1d_fp32_bhl_w_reshape, fftconv1d_fp32_bhl),
        2: (fftconv2d_fp32_bhl_w_reshape, fftconv2d_fp32_bhl),
        3: (fftconv3d_fp32_bhl_w_reshape, fftconv3d_fp32_bhl),
    },
    "causal": {
        1: (causal_fftconv1d_fp32_bhl_w_reshape, causal_fftconv1d_fp32_bhl),
        # Causal is only supported for 1D (sequences)
    },
}

# Chunked versions (memory-efficient, trades compute for lower peak memory)
# Note: circular convolutions don't have chunked variants - they already have lower
# memory overhead since they don't require padding.
FFT_FUNCTIONS_CHUNKED = {
    "zero": {
        1: (fftconv1d_fp32_bhl_w_reshape_chunked, fftconv1d_fp32_bhl_chunked),
        2: (fftconv2d_fp32_bhl_w_reshape_chunked, fftconv2d_fp32_bhl_chunked),
        3: (fftconv3d_fp32_bhl_w_reshape_chunked, fftconv3d_fp32_bhl_chunked),
    },
    "causal": {
        1: (causal_fftconv1d_fp32_bhl_w_reshape_chunked, causal_fftconv1d_fp32_bhl_chunked),
        # Causal is only supported for 1D (sequences)
    },
}

# FP16 versions (power-of-2 padding + ortho normalization to prevent overflow)
# Note: circular convolutions are not supported in fp16 — cuFFT half-precision
# requires power-of-2 sizes which circular padding cannot guarantee.
FFT_FUNCTIONS_FP16 = {
    "zero": {
        1: (fftconv1d_fp16_bhl_w_reshape, fftconv1d_fp16_bhl),
        2: (fftconv2d_fp16_bhl_w_reshape, fftconv2d_fp16_bhl),
        3: (fftconv3d_fp16_bhl_w_reshape, fftconv3d_fp16_bhl),
    },
    "causal": {
        1: (causal_fftconv1d_fp16_bhl_w_reshape, causal_fftconv1d_fp16_bhl),
        # Causal is only supported for 1D (sequences)
    },
}

# FP16 + chunked: combines fp16 memory savings with channel-chunking savings
FFT_FUNCTIONS_FP16_CHUNKED = {
    "zero": {
        1: (fftconv1d_fp16_bhl_w_reshape_chunked, fftconv1d_fp16_bhl_chunked),
        2: (fftconv2d_fp16_bhl_w_reshape_chunked, fftconv2d_fp16_bhl_chunked),
        3: (fftconv3d_fp16_bhl_w_reshape_chunked, fftconv3d_fp16_bhl_chunked),
    },
    "causal": {
        1: (causal_fftconv1d_fp16_bhl_w_reshape_chunked, causal_fftconv1d_fp16_bhl_chunked),
        # Causal is only supported for 1D (sequences)
    },
}


class CKConvND(torch.nn.Module):
    """CKConv (long-convolution) implementation for ND signals."""

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        kernel_cfg: LazyConfig,
        mask_cfg: LazyConfig,
        grid_type: Literal["double", "single"],
        fft_padding: Literal["zero", "circular"],
        is_causal: bool = False,
        use_chunked_fftconv: bool = False,
        use_fp16_fft: bool = False,
        fft_backend: Literal["torch_fft", "subq_ops"] = "torch_fft",
    ):
        """Initialize the CKConvND.

        Args:
            data_dim: Dimension of input data (1D for sequences, 2D for images, 3D for videos, etc.).
            hidden_dim: Hidden dimension.
            kernel_cfg: LazyConfig for the kernel.
            mask_cfg: LazyConfig for the mask.
            grid_type: Type of grid to use.
            fft_padding: Boundary behavior of the FFT convolution. 'zero' uses zero-padding with
                cropping (conventional FFT-based conv). 'circular' uses periodic
                (wrap-around) convolution implemented via frequency-domain phase ramps.
                Must be 'zero' when is_causal=True.
            is_causal: If True, use causal (left-only) convolution where output at position i
                only depends on inputs at positions 0, 1, ..., i. Only supported for 1D data.
            use_chunked_fftconv: If True, use memory-efficient chunked FFT convolutions.
                Processes channels in chunks to reduce peak memory from complex FFT
                intermediates. Typical savings: ~26% memory with ~11% compute overhead.
                Useful for memory-constrained training with large spatial dimensions
                in 2D/3D. Default is False.
            use_fp16_fft: If True, use fp16 FFT convolutions. Pads to power-of-2
                sizes (cuFFT requirement) and uses ortho normalization to prevent
                overflow. Saves ~36% peak memory per convolution with ~0.8% mean
                relative error vs f32. Supported for 1D/2D/3D with zero or causal
                padding (not circular). Default is False.
            fft_backend: FFT convolution backend to use. ``'torch_fft'`` (default)
                uses the torch.fft-based implementations. ``'subq_ops'`` uses the
                optimized CUDA kernels from ``subquadratic_ops_torch``. The subq_ops
                backend currently only supports 2D, zero-padded, non-causal
                convolutions and does not support fp16 FFT. It supports chunked
                convolutions via channel-wise chunking.
        """
        assert grid_type in ["double", "single"], f"Invalid grid type: {grid_type}. Must be 'double' or 'single'."
        assert fft_padding in ["zero", "circular"], (
            f"Invalid FFT padding: {fft_padding}. Must be 'zero' or 'circular'."
        )
        assert fft_backend in ["torch_fft", "subq_ops"], (
            f"Invalid fft_backend: {fft_backend!r}. Must be 'torch_fft' or 'subq_ops'."
        )
        if is_causal:
            assert data_dim == 1, f"Causal CKConvND only supports 1D inputs. Got {data_dim}D."
            assert fft_padding == "zero", f"Causal CKConvND requires fft_padding='zero'. Got '{fft_padding}'."
        if fft_padding == "circular":
            # Circular (periodic) convolution only makes sense with kernel size == input size,
            # which corresponds to 'single' grid type in this CKConv setup.
            assert grid_type == "single", (
                "fft_padding='circular' requires grid_type='single' (kernel size equals input size)."
            )
            # Chunked FFT conv is only implemented for zero-padded and causal convolutions.
            # Circular convolutions have lower memory overhead (no padding) so chunking
            # provides less benefit. The circular functions are re-exported unchanged.
            assert not use_chunked_fftconv, (
                "use_chunked_fftconv=True is not supported with fft_padding='circular'. "
                "Chunked FFT convolutions are only implemented for 'zero' padding (and 'causal' 1D). "
                "Circular convolutions already have lower memory overhead due to no padding."
            )

        if use_fp16_fft:
            assert fft_padding != "circular", (
                "use_fp16_fft does not support circular padding — cuFFT half-precision "
                "requires power-of-2 sizes which circular padding cannot guarantee."
            )

        # subq_ops backend constraints
        if fft_backend == "subq_ops":
            assert data_dim == 2, f"fft_backend='subq_ops' only supports 2D convolutions. Got data_dim={data_dim}."
            assert fft_padding == "zero", (
                f"fft_backend='subq_ops' only supports zero-padded convolutions. Got fft_padding='{fft_padding}'."
            )
            assert not is_causal, "fft_backend='subq_ops' does not support causal convolutions (causal is 1D only)."
            assert not use_fp16_fft, (
                "fft_backend='subq_ops' does not support fp16 FFT — the CUDA kernel "
                "manages its own precision internally. Use use_fp16_fft=False."
            )

        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.fft_padding = fft_padding
        self.is_causal = is_causal
        self.use_chunked_fftconv = use_chunked_fftconv
        self.use_fp16_fft = use_fp16_fft
        self.fft_backend = fft_backend

        # Construct kernel and mask
        self.kernel = instantiate(kernel_cfg)
        self.mask = instantiate(mask_cfg)

        # Construct shortcut projection
        self.shortcut = torch.nn.Parameter(torch.empty(hidden_dim))
        bounds = math.sqrt(1.0 / hidden_dim)
        self.shortcut.data.uniform_(-bounds, bounds)

        # Select FFT convolution functions based on backend
        if fft_backend == "subq_ops":
            from nvsubquadratic.ops.fftconv_custom import (
                fftconv2d_bhl,
                fftconv2d_bhl_chunked,
                fftconv2d_bhl_w_reshape,
                fftconv2d_bhl_w_reshape_chunked,
            )

            if use_chunked_fftconv:
                self.fftconv_fn = fftconv2d_bhl_w_reshape_chunked
                self.fftconv_fn_bhl_input = fftconv2d_bhl_chunked
            else:
                self.fftconv_fn = fftconv2d_bhl_w_reshape
                self.fftconv_fn_bhl_input = fftconv2d_bhl
        else:
            # torch_fft backend: use lookup tables
            # Causal mode overrides fft_padding for 1D
            effective_padding = "causal" if is_causal else self.fft_padding

            # Choose FFT functions: fp16+chunked > fp16 > chunked > standard
            if use_fp16_fft and use_chunked_fftconv:
                fft_fn_table = FFT_FUNCTIONS_FP16_CHUNKED
            elif use_fp16_fft:
                fft_fn_table = FFT_FUNCTIONS_FP16
            elif use_chunked_fftconv:
                fft_fn_table = FFT_FUNCTIONS_CHUNKED
            else:
                fft_fn_table = FFT_FUNCTIONS
            try:
                self.fftconv_fn, self.fftconv_fn_bhl_input = fft_fn_table[effective_padding][self.data_dim]
            except KeyError:
                valid_dims = sorted(fft_fn_table.get(effective_padding, {}).keys())
                raise ValueError(
                    f"Unsupported configuration: fft_padding='{effective_padding}', data_dim={self.data_dim}. "
                    f"Valid dimensions for '{effective_padding}': {valid_dims}"
                )

        # Define the grid type
        self.grid_type = grid_type

    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return (
            f"data_dim={self.data_dim}, hidden_dim={self.hidden_dim}, "
            f"fft_padding={self.fft_padding!r}, grid_type={self.grid_type!r}, is_causal={self.is_causal}, "
            f"use_chunked_fftconv={self.use_chunked_fftconv}, use_fp16_fft={self.use_fp16_fft}, "
            f"fft_backend={self.fft_backend!r}"
        )

    def flop_count(self, spatial_dims: tuple[int, ...], inference: bool = False) -> int:
        """Count FLOPs for CKConv: kernel generation + FFT convolution.

        Two phases:

        **Phase 1 — Kernel generation** (via SIREN MLP):
          Delegated to ``self.kernel.flop_count(grid_lens, inference)``.
          At ``inference=True`` without FiLM, the kernel is input-independent
          and can be precomputed, so this returns 0.

        **Phase 2 — FFT-based depthwise convolution** (C = ``self.hidden_dim``):
          The convolution is computed in the frequency domain.  Padded signal
          sizes Np_i depend on the padding mode:
            - ``"zero"`` non-causal ("same"-mode):
                Np_i = min(s_i + (k_i + 1) // 2,  2 * s_i)
              Only half the kernel width of extra padding is needed beyond
              the input size, because the output is cropped back to input
              size (centered crop).  Matches ``fftconv.py`` line 624-628.
            - ``"zero"`` causal (1D only):
                Np_i = min(s_i + k_i,  2 * s_i)
              Full linear convolution length; output is tail-cropped.
            - ``"circular"``: Np_i = s_i  (wrap-around, no extra padding)

          A separable N-D FFT on a grid of size (Np_1, ..., Np_d) costs:
            5 * prod(Np) * sum(log2(Np_i))  real FLOPs per channel,
          based on the radix-2 Cooley-Tukey decomposition where each butterfly
          operation costs ~5 real FLOPs (1 complex multiply ≈ 4 real muls +
          2 real adds, minus shared twiddle-factor optimizations → ~5 ops).
          Note: the implementation uses ``rfft`` (real-to-complex), which is
          ~2x cheaper than a full complex FFT; the 5N log N formula is a
          conservative (upper-bound) estimate consistent with standard
          vision-paper conventions.

          Three FFTs are needed: forward FFT of input, forward FFT of kernel,
          and inverse FFT of the product.  At ``inference=True`` without FiLM,
          the kernel FFT is precomputed and cached, reducing to 2 FFTs.

          Pointwise complex multiply in the frequency domain:
            6 * C * prod(Np)  (4 real muls + 2 real adds for (a+bi)(c+di)).

          Shortcut (skip connection): C * prod(spatial_dims)  (elementwise).

        Args:
            spatial_dims: Spatial dimensions of the input signal, e.g. (H, W).
            inference: If True and kernel has no FiLM, skip kernel generation
                and kernel FFT (both are precomputable and cached).

        Returns:
            Total FLOPs as an integer.
        """
        C = self.hidden_dim
        has_film = getattr(self.kernel, "film_generator", None) is not None

        # Determine kernel grid_lens (same logic as forward)
        if self.grid_type == "single":
            grid_lens = tuple((s + 1) // 2 for s in spatial_dims)
        else:
            grid_lens = tuple(spatial_dims)

        # Kernel spatial sizes: the SIREN generates on a (2*L - 1) grid per dim
        kernel_sizes = tuple(2 * gl - 1 for gl in grid_lens)

        # For causal 1D, kernel is cropped to second half
        if self.is_causal:
            kernel_sizes = tuple(ks // 2 + 1 for ks in kernel_sizes)

        flops = 0

        # Phase 1: Kernel generation
        flops += self.kernel.flop_count(grid_lens, inference=inference)

        # Phase 2: FFT convolution
        # Padded sizes match the actual fftconv implementations (fftconv.py):
        #   non-causal "same": min(s + (k+1)//2, 2*s)
        #   causal:            min(s + k, 2*s)
        #   circular:          s  (no extra padding)
        if self.fft_padding == "circular":
            padded_dims = tuple(spatial_dims)
        elif self.is_causal:
            padded_dims = tuple(min(s + k, 2 * s) for s, k in zip(spatial_dims, kernel_sizes))
        else:
            padded_dims = tuple(min(s + (k + 1) // 2, 2 * s) for s, k in zip(spatial_dims, kernel_sizes))

        prod_padded = 1
        for p in padded_dims:
            prod_padded *= p
        log2_sum = sum(math.log2(max(p, 1)) for p in padded_dims)

        # 3 FFTs (input, kernel, inverse) normally;
        # 2 FFTs (input, inverse) at inference without FiLM (kernel FFT cached).
        num_ffts = 2 if (inference and not has_film) else 3
        fft_flops = num_ffts * 5 * C * prod_padded * log2_sum

        # Pointwise complex multiply in frequency domain
        cmul_flops = 6 * C * prod_padded

        # Shortcut (elementwise multiply: input * shortcut_weight)
        prod_spatial = 1
        for s in spatial_dims:
            prod_spatial *= s
        shortcut_flops = C * prod_spatial

        flops += int(fft_flops) + cmul_flops + shortcut_flops

        return flops

    def apply_convolution(
        self, x: torch.Tensor, conv_kernel: torch.Tensor, shortcut: torch.Tensor, is_bhl_input: bool
    ) -> torch.Tensor:
        """Apply the convolution operation using the FFT-based convolution function.

        Args:
            x (torch.Tensor): Input tensor.
            conv_kernel (torch.Tensor): Convolution kernel tensor.
            shortcut (torch.Tensor): Shortcut tensor.
            is_bhl_input (bool): Whether the input is in BHL format.

        Returns:
            torch.Tensor: Output tensor after applying convolution.
        """
        if is_bhl_input:
            conv_kernel = rearrange(
                conv_kernel, "b ... c -> b c ..."
            )  # Reshape kernel to [B, C, * spatial_dims] (Kernels are always in BLH format)
            _conv_fn = self.fftconv_fn_bhl_input
        else:
            _conv_fn = self.fftconv_fn

        return _conv_fn(x, conv_kernel, shortcut)

    def forward(
        self,
        x: torch.Tensor,
        is_bhl_input: bool = False,
        cp_group: torch.distributed.ProcessGroup = None,
        **mixer_kwargs,
    ) -> torch.Tensor:
        """Forward pass of the CKConvND.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, * spatial_dims, hidden_dim) or (batch_size, hidden_dim, * spatial_dims)
            is_bhl_input (bool): Whether the input is in BHL format, i.e., (batch_size, hidden_dim, * spatial_dims).
                Default is False.
            cp_group (torch.distributed.ProcessGroup): Context parallel process group.
                Default is None.
            **mixer_kwargs: Additional keyword arguments forwarded to the kernel generator
                (e.g. ``conditioning`` for FiLM-enabled SIRENKernelND).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, * spatial_dims, hidden_dim) or (batch_size, hidden_dim, * spatial_dims)
        """
        # Get the spatial dimensions from the input tensor
        if is_bhl_input:
            spatial_dims = x.shape[2:]  # [* spatial_dims]
        else:
            spatial_dims = x.shape[1:-1]  # [* spatial_dims]

        if self.grid_type == "single":
            # Take half the spatial dimensions for the grid cache. Since the grid cache is
            # double the size of the input, when we take half the spatial dimensions, we get
            # a convolutional kernel with size equal to the input.
            grid_lens = [(seq_len + 1) // 2 for seq_len in spatial_dims]
        else:  # "double"
            # Take the full spatial dimensions for the grid cache. Since the grid cache is
            # double the size of the input, when we take the full spatial dimensions, we get
            # a convolutional kernel with size equal to twice the input.
            grid_lens = spatial_dims

        # Compute kernel (pass conditioning if available for FiLM-enabled kernels)
        conditioning = mixer_kwargs.get("conditioning", None)
        conv_kernel, grid = self.kernel(grid_lens, conditioning=conditioning)

        # Apply mask to kernel
        if not isinstance(self.mask, torch.nn.Identity):
            conv_kernel = self.mask(grid=grid, x=conv_kernel)

        # For causal convolution, crop the kernel to use only the "positive" half
        # (i.e., the part that looks backward in time). The kernel is in BLH format: [1, L, H].
        # We keep positions from L//2 to L-1, which after the FFT flip becomes causal.
        if self.is_causal:
            # Kernel shape is [1, kernel_len, hidden_dim] for 1D
            # Crop to [1, kernel_len // 2, hidden_dim] keeping the second half
            kernel_len = conv_kernel.shape[-2]
            conv_kernel = conv_kernel[..., kernel_len // 2 :, :]

        # Handle context parallelism by slicing the kernel to match input channel dimensions
        if cp_group is not None and cp_group.size() > 1:
            if self.is_causal:
                raise ValueError("Causal CKConvND has not been verified to work with context parallelism.")
            cp_world_size = cp_group.size()
            cp_rank = cp_group.rank()

            # Get the channel dimension (last dimension in BLH format)
            kernel_channels = conv_kernel.shape[-1]
            channels_per_rank = kernel_channels // cp_world_size

            # Slice the kernel along the channel dimension for this CP rank
            start_idx = cp_rank * channels_per_rank
            end_idx = start_idx + channels_per_rank
            conv_kernel = conv_kernel[..., start_idx:end_idx]

            # Also slice the shortcut parameter
            shortcut = self.shortcut[start_idx:end_idx]
        else:
            shortcut = self.shortcut

        # Apply convolution
        out = self.apply_convolution(x, conv_kernel, shortcut, is_bhl_input)

        return out
