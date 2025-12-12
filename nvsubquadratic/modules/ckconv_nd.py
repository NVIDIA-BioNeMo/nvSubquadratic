# TODO: Add license header here


"""CKConv (long-convolution) implementation for ND signals.

For debugging, please run:
    PYTHONPATH=. python nvsubquadratic/modules/ckconv_nd.py
"""

import math
from typing import Literal

import torch
from einops import rearrange

from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.ops.circular_fftconv import (
    circular_fftconv1d_bhl,
    circular_fftconv1d_bhl_w_reshape,
    circular_fftconv2d_bhl,
    circular_fftconv2d_bhl_w_reshape,
    circular_fftconv3d_bhl,
    circular_fftconv3d_bhl_w_reshape,
)
from nvsubquadratic.ops.fftconv import (
    fftconv1d_bhl,
    fftconv1d_bhl_w_reshape,
    fftconv2d_bhl,
    fftconv2d_bhl_w_reshape,
    fftconv3d_bhl,
    fftconv3d_bhl_w_reshape,
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
        use_shortcut: bool = True,
        spectral_mask_cfg: LazyConfig = LazyConfig(torch.nn.Identity)(),
        is_depthwise: bool = True,
    ):
        """Initialize the CKConvND.

        Args:
            data_dim: Dimension of input data (1D for sequences, 2D for images, 3D for videos, etc.).
            hidden_dim: Hidden dimension.
            kernel_cfg: LazyConfig for the kernel.
            mask_cfg: LazyConfig for the mask (spatial mask).
            grid_type: Type of grid to use.
            fft_padding: Boundary behavior of the FFT convolution. 'zero' uses zero-padding with
                cropping (conventional FFT-based conv). 'circular' uses periodic
                (wrap-around) convolution implemented via frequency-domain phase ramps.
            use_shortcut: Whether to use shortcut. Defaults to True.
            spectral_mask_cfg: LazyConfig for the spectral mask (used for learnable stride). Defaults to torch.nn.Identity.
            is_depthwise: Whether the CKConvND is a depthwise convolution. Defaults to True.
        """
        assert grid_type in ["double", "single"], f"Invalid grid type: {grid_type}. Must be 'double' or 'single'."
        assert fft_padding in ["zero", "circular"], (
            f"Invalid FFT padding: {fft_padding}. Must be 'zero' or 'circular'."
        )
        if fft_padding == "circular":
            # Circular (periodic) convolution only makes sense with kernel size == input size,
            # which corresponds to 'single' grid type in this CKConv setup.
            assert grid_type == "single", (
                "fft_padding='circular' requires grid_type='single' (kernel size equals input size)."
            )

        super().__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.fft_padding = fft_padding
        self.is_depthwise = is_depthwise

        # Construct kernel
        self.kernel = instantiate(kernel_cfg)
        # Check that the kernel output dimension is correct for depthwise or non-depthwise convolution

        assert self.kernel.out_dim == self.hidden_dim, "Kernel output dimension must be equal to hidden dimension."

        # Construct spatial & spectral masks
        self.spatial_mask = instantiate(mask_cfg)
        self.spectral_mask = instantiate(spectral_mask_cfg)

        # Construct shortcut projection
        self.use_shortcut = use_shortcut
        if self.use_shortcut:
            self.shortcut = torch.nn.Parameter(torch.empty(hidden_dim, dtype=torch.float32))
            bounds = math.sqrt(1.0 / hidden_dim)
            self.shortcut.data.uniform_(-bounds, bounds)
        else:
            self.shortcut = None

        # Define FFT operation depending on padding and dimensionality
        try:
            self.fftconv_fn, self.fftconv_fn_bhl_input = FFT_FUNCTIONS[self.fft_padding][self.data_dim]
        except KeyError:
            valid_dims = sorted(FFT_FUNCTIONS.get(self.fft_padding, {}).keys())
            raise ValueError(
                f"Unsupported configuration: fft_padding='{self.fft_padding}', data_dim={self.data_dim}. "
                f"Valid dimensions for '{self.fft_padding}': {valid_dims}"
            )

        # Define the grid type
        self.grid_type = grid_type

    def get_stride(self) -> torch.Tensor | None:
        """Get current stride from spectral mask (if applicable).

        Returns:
            torch.Tensor | None: The effective stride per dimension, or None if no
                spectral mask with learnable stride is used.
        """
        if hasattr(self.spectral_mask, "get_stride"):
            return self.spectral_mask.get_stride()
        return None

    @torch.compiler.disable()
    def apply_convolution(
        self,
        x: torch.Tensor,
        conv_kernel: torch.Tensor,
        spectral_mask: torch.Tensor | None,
        shortcut: torch.Tensor,
        is_bhl_input: bool,
    ) -> torch.Tensor:
        """Apply the convolution operation using the FFT-based convolution function.

        Uses separate function to avoid torch.compile issues with complex numbers.

        Args:
            x (torch.Tensor): Input tensor.
            conv_kernel (torch.Tensor): Convolution kernel tensor.
            spectral_mask (torch.Tensor | None): Spectral mask tensor.
            shortcut (torch.Tensor): Shortcut tensor.
            is_bhl_input (bool): Whether the input is in BHL format.

        Returns:
            torch.Tensor: Output tensor after applying convolution.
        """
        if not self.is_depthwise:
            c_in = x.shape[1] if is_bhl_input else x.shape[-1]  # [B, C, * spatial_dims] or [B, * spatial_dims, C]
            # Scale by 1/sqrt(c_in) to preserve variance (analogous to Kaiming initialization)
            conv_kernel = conv_kernel * (c_in**-0.5)

        # Add channel dimension to spectral mask if needed
        # The mask from SpectralGaussianMaskND has shape [1, *cropped_dims], e.g., [1, sM_x, sM_y]
        # Convolution functions expect [1, C, *cropped_dims] (BHL) or [1, *cropped_dims, C] (BLH)
        # Adding a singleton channel dim allows broadcasting across all channels
        if spectral_mask is not None and spectral_mask.dim() == self.data_dim + 1:
            # Shape is [1, *cropped_dims], add channel dim at end for BLH format
            spectral_mask = spectral_mask.unsqueeze(-1)  # [1, sM_x, sM_y] -> [1, sM_x, sM_y, 1]

        if is_bhl_input:
            # Apply kernel
            if self.is_depthwise:
                conv_kernel = rearrange(
                    conv_kernel, "b ... c -> b c ..."
                )  # Reshape kernel to [B, C, * spatial_dims] (Kernels are always in BLH format)
            else:
                conv_kernel = rearrange(
                    conv_kernel,
                    "1 ... (c_out c_in) -> c_out c_in ...",
                    c_in=x.shape[1],
                )  # Reshape kernel to [B, C, * spatial_dims] (Kernels are always in BLH format)
            if spectral_mask is not None:
                spectral_mask = rearrange(
                    spectral_mask, "b ... c -> b c ..."
                )  # Reshape spectral mask to [B, C, * spatial_dims]
            x_dtype = x.dtype
            x = self.fftconv_fn_bhl_input(
                x.to(torch.float32),
                conv_kernel.to(torch.float32),
                is_depthwise=self.is_depthwise,
                shortcut=shortcut.to(torch.float32) if shortcut is not None else None,
                spectral_mask=spectral_mask.to(torch.float32) if spectral_mask is not None else None,
            )
            return x.to(x_dtype)
        else:
            # Apply kernel
            x_dtype = x.dtype
            x = self.fftconv_fn(
                x.to(torch.float32),
                conv_kernel.to(torch.float32),
                is_depthwise=self.is_depthwise,
                shortcut=shortcut.to(torch.float32) if shortcut is not None else None,
                spectral_mask=spectral_mask.to(torch.float32) if spectral_mask is not None else None,
            )
            return x.to(x_dtype)

    def forward(
        self, x: torch.Tensor, is_bhl_input: bool = False, cp_group: torch.distributed.ProcessGroup = None
    ) -> torch.Tensor:
        """Forward pass of the CKConvND.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, * spatial_dims, hidden_dim) or (batch_size, hidden_dim, * spatial_dims)
            is_bhl_input (bool): Whether the input is in BHL format, i.e., (batch_size, hidden_dim, * spatial_dims).
                Default is False.
            cp_group (torch.distributed.ProcessGroup): Context parallel process group.
                Default is None.

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

        # Compute kernel
        conv_kernel, grid = self.kernel(grid_lens)

        # Apply mask to kernel
        if not isinstance(self.spatial_mask, torch.nn.Identity):
            conv_kernel = self.spatial_mask(grid=grid, x=conv_kernel)

        # Generate spectral mask and crop function
        if not isinstance(self.spectral_mask, torch.nn.Identity):
            spectral_mask = self.spectral_mask(spatial_dims=spatial_dims)
        else:
            spectral_mask = None

        shortcut = self.shortcut

        # TODO(@dwromero): I needed to deprecate here to be able to do full-convolutions. Need to de-deprecate this.
        # # Handle context parallelism by slicing the kernel to match input channel dimensions
        # if cp_group is not None and cp_group.size() > 1:
        #     cp_world_size = cp_group.size()
        #     cp_rank = cp_group.rank()

        #     # Get the channel dimension (last dimension in BLH format)
        #     kernel_channels = conv_kernel.shape[-1]
        #     channels_per_rank = kernel_channels // cp_world_size

        #     # Slice the kernel along the channel dimension for this CP rank
        #     start_idx = cp_rank * channels_per_rank
        #     end_idx = start_idx + channels_per_rank
        #     conv_kernel = conv_kernel[..., start_idx:end_idx]

        #     # Also slice the shortcut parameter
        #     if shortcut is not None:
        #         shortcut = shortcut[start_idx:end_idx]

        # Apply convolution
        out = self.apply_convolution(
            x=x, conv_kernel=conv_kernel, spectral_mask=spectral_mask, shortcut=shortcut, is_bhl_input=is_bhl_input
        )

        return out


if __name__ == "__main__":
    """Debug instance for non-depthwise CKConvND."""
    from nvsubquadratic.lazy_config import LazyConfig
    from nvsubquadratic.modules.kernels_nd import RandomFourierKernelND
    from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND

    # Configuration
    data_dim = 2
    hidden_dim = 64
    L_cache = 128  # Will create kernels of size 2*L_cache - 1 = 255

    # For non-depthwise: kernel.out_dim = hidden_dim * hidden_dim (h_out * h_in)
    # The kernel output will be reshaped to (h_out, h_in, K_x, K_y) for einsum
    kernel_out_dim = hidden_dim * hidden_dim

    # Kernel config
    kernel_cfg = LazyConfig(RandomFourierKernelND)(
        out_dim=kernel_out_dim,
        data_dim=data_dim,
        mlp_hidden_dim=128,
        num_layers=3,
        embedding_dim=64,
        omega_0=1.0,
        L_cache=L_cache,
        use_bias=True,
        nonlinear_cfg=LazyConfig(torch.nn.GELU)(),
    )

    # Mask config (spatial modulation)
    mask_cfg = LazyConfig(torch.nn.Identity)()

    # Spectral mask config
    spectral_mask_cfg = LazyConfig(SpectralGaussianMaskND)(
        data_dim=data_dim,
        clip_value=0.5,
        init_stride_value=4.0,
        min_stride_value=1.0,
        max_stride_value=None,
        parametrization="direct",
    )

    # Create the non-depthwise CKConvND
    ckconv = CKConvND(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        kernel_cfg=kernel_cfg,
        mask_cfg=mask_cfg,
        spectral_mask_cfg=spectral_mask_cfg,
        grid_type="single",
        fft_padding="zero",
        use_shortcut=True,
        is_depthwise=False,
    )

    print("Created non-depthwise CKConvND:")
    print(f"  data_dim: {data_dim}")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  kernel.out_dim: {kernel_out_dim}")
    print("  is_depthwise: False")

    # Test forward pass with is_bhl_input=True (B, C, H, W)
    B, H, W = 2, 64, 64
    x_bhl = torch.randn(B, hidden_dim, H, W)  # BHL format: (B, C, H, W)
    print("\n[is_bhl_input=True]")
    print(f"  Input shape: {x_bhl.shape}")

    with torch.no_grad():
        y_bhl = ckconv(x_bhl, is_bhl_input=True)
    print(f"  Output shape: {y_bhl.shape}")

    # Test forward pass with is_bhl_input=False (B, H, W, C)
    x_blh = torch.randn(B, H, W, hidden_dim)  # BLH format: (B, H, W, C)
    print("\n[is_bhl_input=False]")
    print(f"  Input shape: {x_blh.shape}")

    with torch.no_grad():
        y_blh = ckconv(x_blh, is_bhl_input=False)
    print(f"  Output shape: {y_blh.shape}")

    # Demonstrate get_stride()
    stride = ckconv.get_stride()
    print(f"\nSpectral mask stride: {stride.tolist() if stride is not None else None}")
