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
    causal_fftconv1d_bhl,
    causal_fftconv1d_bhl_w_reshape,
    fftconv1d_bhl,
    fftconv1d_bhl_w_reshape,
    fftconv2d_bhl,
    fftconv2d_bhl_w_reshape,
    fftconv3d_bhl,
    fftconv3d_bhl_w_reshape,
)

# Chunked (memory-efficient) variants for zero-padded and causal convolutions
# Note: circular convolutions don't have chunked variants (lower memory overhead already)
from nvsubquadratic.ops.fftconv_chunked import (
    causal_fftconv1d_bhl as causal_fftconv1d_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    causal_fftconv1d_bhl_w_reshape as causal_fftconv1d_bhl_w_reshape_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv1d_bhl as fftconv1d_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv1d_bhl_w_reshape as fftconv1d_bhl_w_reshape_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv2d_bhl as fftconv2d_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv2d_bhl_w_reshape as fftconv2d_bhl_w_reshape_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv3d_bhl as fftconv3d_bhl_chunked,
)
from nvsubquadratic.ops.fftconv_chunked import (
    fftconv3d_bhl_w_reshape as fftconv3d_bhl_w_reshape_chunked,
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
        1: (fftconv1d_bhl_w_reshape, fftconv1d_bhl),
        2: (fftconv2d_bhl_w_reshape, fftconv2d_bhl),
        3: (fftconv3d_bhl_w_reshape, fftconv3d_bhl),
    },
    "causal": {
        1: (causal_fftconv1d_bhl_w_reshape, causal_fftconv1d_bhl),
        # Causal is only supported for 1D (sequences)
    },
}

# Chunked versions (memory-efficient, trades compute for lower peak memory)
# Note: circular convolutions don't have chunked variants - they already have lower
# memory overhead since they don't require padding.
FFT_FUNCTIONS_CHUNKED = {
    "zero": {
        1: (fftconv1d_bhl_w_reshape_chunked, fftconv1d_bhl_chunked),
        2: (fftconv2d_bhl_w_reshape_chunked, fftconv2d_bhl_chunked),
        3: (fftconv3d_bhl_w_reshape_chunked, fftconv3d_bhl_chunked),
    },
    "causal": {
        1: (causal_fftconv1d_bhl_w_reshape_chunked, causal_fftconv1d_bhl_chunked),
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
        """
        assert grid_type in ["double", "single"], f"Invalid grid type: {grid_type}. Must be 'double' or 'single'."
        assert fft_padding in ["zero", "circular"], (
            f"Invalid FFT padding: {fft_padding}. Must be 'zero' or 'circular'."
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

        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.fft_padding = fft_padding
        self.is_causal = is_causal
        self.use_chunked_fftconv = use_chunked_fftconv

        # Construct kernel and mask
        self.kernel = instantiate(kernel_cfg)
        self.mask = instantiate(mask_cfg)

        # Construct shortcut projection
        self.shortcut = torch.nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
        bounds = math.sqrt(1.0 / hidden_dim)
        self.shortcut.data.uniform_(-bounds, bounds)

        # Define FFT operation depending on padding and dimensionality
        # Causal mode overrides fft_padding for 1D
        effective_padding = "causal" if is_causal else self.fft_padding

        # Choose between standard and chunked FFT functions
        fft_fn_table = FFT_FUNCTIONS_CHUNKED if use_chunked_fftconv else FFT_FUNCTIONS
        try:
            self.fftconv_fn, self.fftconv_fn_bhl_input = fft_fn_table[effective_padding][self.data_dim]
        except KeyError:
            valid_dims = sorted(FFT_FUNCTIONS.get(effective_padding, {}).keys())
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
            f"use_chunked_fftconv={self.use_chunked_fftconv}"
        )

    @torch.compiler.disable
    def apply_convolution(
        self, x: torch.Tensor, conv_kernel: torch.Tensor, shortcut: torch.Tensor, is_bhl_input: bool
    ) -> torch.Tensor:
        """Apply the convolution operation using the FFT-based convolution function.

        Excluded from torch.compile: Triton codegen does not support complex64
        element-wise ops with batch-dependent kernels (e.g. FiLM conditioning).

        Args:
            x (torch.Tensor): Input tensor.
            conv_kernel (torch.Tensor): Convolution kernel tensor.
            shortcut (torch.Tensor): Shortcut tensor.
            is_bhl_input (bool): Whether the input is in BHL format.

        Returns:
            torch.Tensor: Output tensor after applying convolution.
        """
        if is_bhl_input:
            # Apply kernel
            conv_kernel = rearrange(
                conv_kernel, "b ... c -> b c ..."
            )  # Reshape kernel to [B, C, * spatial_dims] (Kernels are always in BLH format)
            x_dtype = x.dtype
            x = self.fftconv_fn_bhl_input(
                x.to(torch.float32),
                conv_kernel.to(torch.float32),
                shortcut.to(torch.float32),
            )
            return x.to(x_dtype)
        else:
            # Apply kernel
            x_dtype = x.dtype
            x = self.fftconv_fn(
                x.to(torch.float32),
                conv_kernel.to(torch.float32),
                shortcut.to(torch.float32),
            )
            return x.to(x_dtype)

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
