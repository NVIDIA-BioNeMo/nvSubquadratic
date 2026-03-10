# David W. Romero, 2025-09-09

"""Patchify and Unpatchify layers as ConvND and ConvTransposeND layers.

Usage test:
    PYTHONPATH=. python nvsubquadratic/modules/patchify.py
"""

from typing import Tuple

import torch
from einops import rearrange


# Mapping from data_dim to Conv and ConvTranspose classes
_CONV_CLASSES = {
    1: torch.nn.Conv1d,
    2: torch.nn.Conv2d,
    3: torch.nn.Conv3d,
}

_CONV_TRANSPOSE_CLASSES = {
    1: torch.nn.ConvTranspose1d,
    2: torch.nn.ConvTranspose2d,
    3: torch.nn.ConvTranspose3d,
}


class Patchify(torch.nn.Module):
    """Conv-based image patchification (channels-last input).

    This mirrors the ViT/timm approach where a Conv with kernel_size=stride=patch_size
    produces one embedding per patch location (non-overlapping patches).

    Input shape:  [B, *spatial_dims, in_features] (channels-last, e.g., BHWC)
    Output shape: [B, *spatial_dims // patch_size, out_features]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        patch_size: int,
        stride: int | None = None,
    ):
        """Initialize the Patchify layer.

        Args:
            in_features: The number of input channels.
            out_features: The number of output channels (embedding dimension).
            data_dim: The spatial dimensionality (1, 2, or 3).
            patch_size: The size of each patch (kernel_size for the conv).
            stride: The stride for the conv. Defaults to patch_size (non-overlapping).
        """
        super().__init__()
        if data_dim not in _CONV_CLASSES:
            raise ValueError(f"data_dim must be 1, 2, or 3, got {data_dim}")

        if stride is None:
            stride = patch_size  # Default: non-overlapping patches (ViT-style)

        self.data_dim = data_dim
        self.patch_size = patch_size
        self.stride = stride

        conv_class = _CONV_CLASSES[data_dim]
        self.conv = conv_class(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Patchify layer.

        Args:
            x: The input tensor of shape [B, *spatial_dims, in_features].

        Returns:
            The output tensor of shape [B, *spatial_dims // stride, out_features].
        """
        # Channels-last -> channels-first for ConvNd
        x = rearrange(x, "b ... c -> b c ...")

        # Apply conv
        y = self.conv(x)

        # Channels-first -> channels-last
        y = rearrange(y, "b c ... -> b ... c")
        return y


class Unpatchify(torch.nn.Module):
    """Inverse of Patchify for channels-last inputs (supports 1D/2D/3D).

    Uses ConvTranspose to upsample from patch resolution back to original resolution.

    Input shape:  [B, *spatial_dims, in_features] (channels-last)
    Output shape: [B, *spatial_dims * stride, out_features]

    If exact spatial size control is required, pass output_spatial_shape to forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        patch_size: int,
        stride: int | None = None,
    ):
        """Initialize the Unpatchify layer.

        Args:
            in_features: The number of input channels (embedding dimension).
            out_features: The number of output channels.
            data_dim: The spatial dimensionality (1, 2, or 3).
            patch_size: The size of each patch (kernel_size for the deconv).
            stride: The stride for the deconv. Defaults to patch_size (inverse of non-overlapping).
        """
        super().__init__()
        if data_dim not in _CONV_TRANSPOSE_CLASSES:
            raise ValueError(f"data_dim must be 1, 2, or 3, got {data_dim}")

        if stride is None:
            stride = patch_size  # Default: inverse of non-overlapping patches

        self.data_dim = data_dim
        self.patch_size = patch_size
        self.stride = stride

        deconv_class = _CONV_TRANSPOSE_CLASSES[data_dim]
        self.deconv = deconv_class(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=patch_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor, output_spatial_shape: Tuple[int, ...] | None = None) -> torch.Tensor:
        """Forward pass of the Unpatchify layer.

        Args:
            x: The input tensor of shape [B, *spatial_dims, in_features].
            output_spatial_shape: The desired output spatial shape (optional).

        Returns:
            The output tensor of shape [B, *spatial_dims * stride, out_features].
        """
        expected_dims = self.data_dim + 2  # batch + spatial_dims + channels
        assert x.dim() == expected_dims, (
            f"Expected {expected_dims}D tensor for data_dim={self.data_dim}, got {x.dim()}D tensor with shape {tuple(x.shape)}"
        )

        # Channels-last -> channels-first for ConvTransposeNd
        x_bc = rearrange(x, "b ... c -> b c ...")

        # Apply deconvolution (optionally with target output size)
        if output_spatial_shape is None:
            y_bc = self.deconv(x_bc)
        else:
            y_bc = self.deconv(x_bc, output_size=tuple(int(v) for v in output_spatial_shape))

        # Channels-first -> channels-last
        y = rearrange(y_bc, "b c ... -> b ... c")
        return y


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)

    # Example: 2D image B x 64 x 64 x 3 (channels-last)
    B, H, W, hidden_dim = 2, 64, 64, 3
    embedding_dim = 32
    patch_size = 8
    x = torch.randn(B, H, W, hidden_dim)

    # Patchify layer (ViT-style: stride = patch_size)
    patchify_layer = Patchify(
        in_features=hidden_dim,
        out_features=embedding_dim,
        data_dim=2,
        patch_size=patch_size,
        # stride defaults to patch_size (non-overlapping)
    )

    # Unpatchify layer (inverse of patchify)
    unpatchify_layer = Unpatchify(
        in_features=embedding_dim,
        out_features=hidden_dim,
        data_dim=2,
        patch_size=patch_size,
        # stride defaults to patch_size
    )

    # Run layers
    patchify_layer.cuda()
    unpatchify_layer.cuda()

    y = patchify_layer(x)
    x_rec = unpatchify_layer(y)

    print(f"Input shape:      {tuple(x.shape)}")
    print(f"Patched shape:    {tuple(y.shape)}")
    print(f"Reconstructed:    {tuple(x_rec.shape)}")
    assert x_rec.shape == x.shape, "Reconstructed shape does not match input shape"
    print("Shape check passed.")
