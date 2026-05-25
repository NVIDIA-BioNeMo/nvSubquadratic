# David W. Romero, 2025-09-09

"""Patch embedding and reconstruction layers for ND spatial signals.

Overview
--------
Patchification is the bridge between raw pixel space and the sequence of tokens
that a downstream mixer (Hyena, Attention, CKConv, …) operates on.  Given an
input image (or any ND spatial signal) of shape ``[B, *spatial, C_in]``, the
layer divides the spatial axes into a grid of non-overlapping *patches* of size
``patch_size`` and linearly projects the flattened pixels within each patch into
a ``C_embed``-dimensional token embedding.  The result is a spatially-ordered
grid of tokens ready for sequence mixing.

Implementation approach
-----------------------
Both ``Patchify`` and ``Unpatchify`` are implemented as strided convolutions
(``Conv{1,2,3}d`` / ``ConvTranspose{1,2,3}d``) so that the unfold + linear
projection are fused into a single CUDA kernel.  Setting
``kernel_size == stride == patch_size`` and ``padding == 0`` gives exactly the
ViT non-overlapping patch semantics.

Output layout convention
------------------------
All layers in this module use **channels-last** tensors externally::

    input  : [B, *spatial_dims, C_in]   e.g. [B, H, W, C_in]  for 2D
    output : [B, *patch_grid, C_embed]  e.g. [B, H/P, W/P, C_embed]

Channels are reordered to channels-first only internally before the convolution
and back to channels-last before returning, which matches the layout expected by
``PositionEmbeddingND`` and the subsequent mixer blocks.

Supported dimensionalities
--------------------------
Both classes support ``data_dim ∈ {1, 2, 3}`` (time-series / images / volumes)
via the ``_CONV_CLASSES`` / ``_CONV_TRANSPOSE_CLASSES`` dispatch tables.

Usage test::

    PYTHONPATH=. python nvsubquadratic/modules/patchify.py
"""

import math
from typing import Literal, Tuple

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
    """Conv-based patch embedding for ND spatial signals (channels-last I/O).

    Splits the spatial axes of the input into a regular grid of non-overlapping
    patches and linearly projects each patch into an embedding vector.  The
    operation is equivalent to:

    1. Unfold every ``patch_size^data_dim`` pixel neighbourhood into a vector
       of length ``C_in * patch_size^data_dim``.
    2. Apply a learned linear map from that vector to ``C_out`` dimensions.

    Because the unfold and linear projection can be fused into a single strided
    convolution, this class simply wraps ``torch.nn.Conv{data_dim}d`` with
    ``kernel_size = patch_size``, ``stride = stride``, and ``padding = 0``.

    **Output shape formula** (each spatial axis ``s`` independently)::

        out_s = floor((s - patch_size) / stride) + 1

    For the default non-overlapping case (``stride == patch_size``) this
    reduces to ``s / patch_size`` (assuming ``s`` is divisible by
    ``patch_size``).

    **Layout convention** — inputs and outputs use *channels-last* ordering::

        input  : [B, *spatial_dims, C_in]     (e.g. [B, H, W, C_in] for 2D)
        output : [B, *patch_grid, C_out]      (e.g. [B, H/P, W/P, C_out])

    Internally, the tensor is transposed to channels-first before the Conv and
    back to channels-last before returning, to match the layout expected by
    ``PositionEmbeddingND`` and the mixer blocks.

    **Overlapping patches** — setting ``stride < patch_size`` produces
    overlapping patches with the same formula above.  This is less common in
    ViT-style models but is supported.

    Attributes:
        data_dim (int): Spatial dimensionality (1, 2, or 3).
        patch_size (int): Receptive field size of each patch along every axis.
        stride (int): Step between successive patch origins along every axis.
        conv (torch.nn.Conv{data_dim}d): The underlying strided convolution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        patch_size: int,
        stride: int | None = None,
        bias: bool = True,
    ):
        """Initialise the Patchify layer.

        Args:
            in_features: Number of input channels ``C_in`` (e.g. 3 for RGB).
            out_features: Embedding dimension ``C_out`` of each output token.
            data_dim: Spatial dimensionality of the input signal.  Must be 1
                (sequences), 2 (images), or 3 (volumes).
            patch_size: Side length ``P`` of each patch.  The convolution uses
                ``kernel_size = patch_size`` along every spatial axis, so each
                patch covers ``P^data_dim`` input pixels.
            stride: Step size between consecutive patch origins along every
                spatial axis.  Defaults to ``patch_size``, giving
                non-overlapping ViT-style patches.  Set to a smaller value for
                overlapping patches (denser token grids at the cost of more
                tokens).
            bias: If ``True`` (default), the projection conv includes a
                learnable bias.  Set to ``False`` for bias-free architectures
                (e.g. when a subsequent normalisation layer makes bias
                redundant).

        Raises:
            ValueError: If ``data_dim`` is not 1, 2, or 3.
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
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed the input tensor into a grid of patch tokens.

        Args:
            x: Input tensor in channels-last layout.  Shape:
                ``[B, *spatial_dims, C_in]``, e.g. ``[B, H, W, C_in]`` for
                2D images.

        Returns:
            Patch-embedded tensor in channels-last layout.  Shape:
            ``[B, *patch_grid, C_out]``, where each spatial axis ``s`` is
            reduced to ``floor((s - patch_size) / stride) + 1`` (equal to
            ``s // patch_size`` when ``stride == patch_size`` and ``s`` is
            divisible by ``patch_size``).

        Note:
            ``.contiguous()`` is called after the channels-last → channels-first
            rearrangement to avoid a stride-mismatch error in
            ``torch.compile``'s ``convolution_backward``.
        """
        # Channels-last -> channels-first for ConvNd
        # .contiguous() avoids a stride mismatch in torch.compile's convolution_backward
        x = rearrange(x, "b ... c -> b c ...").contiguous()

        # Apply conv
        y = self.conv(x)

        # Channels-first -> channels-last
        y = rearrange(y, "b c ... -> b ... c")
        return y


class Unpatchify(torch.nn.Module):
    """Inverse patch-embedding layer: reconstruct spatial signal from token grid.

    ``Unpatchify`` is the trainable inverse of ``Patchify``.  Given a grid of
    token embeddings at patch resolution, it reconstructs a signal at the
    original spatial resolution using a transposed convolution
    (``ConvTranspose{data_dim}d``).

    For non-overlapping patches (``stride == patch_size``), the default
    transposed convolution is an exact spatial inverse: each output pixel is
    produced by exactly one input token.  When ``stride < patch_size``
    (overlapping), contributions from overlapping patches are *summed* by the
    transposed convolution — this is the adjoint of the overlapping-patch
    forward pass.

    **Output shape formula** (each spatial axis ``s`` of the patch-grid input)::

        out_s = (s - 1) * stride - 2 * padding + kernel_size
              = (s - 1) * stride + patch_size          (since padding == 0)

    For the non-overlapping case this gives ``s * patch_size``.

    **Layout convention** — inputs and outputs use *channels-last* ordering::

        input  : [B, *patch_grid, C_embed]    (e.g. [B, H/P, W/P, C_embed])
        output : [B, *spatial_dims, C_out]    (e.g. [B, H, W, C_out])

    **Weight initialisation** — PyTorch's default kaiming_uniform for
    ``ConvTranspose`` uses ``fan_out = out_features * patch_size^data_dim``.
    This is incorrect for large embedding dimensions; ``weight_init="fan_in"``
    corrects this by using the true fan-in
    ``in_features * patch_size^data_dim``.

    Attributes:
        data_dim (int): Spatial dimensionality (1, 2, or 3).
        patch_size (int): Kernel size of the transposed convolution.
        stride (int): Stride of the transposed convolution.
        deconv (torch.nn.ConvTranspose{data_dim}d): The underlying deconvolution.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        data_dim: int,
        patch_size: int,
        stride: int | None = None,
        bias: bool = True,
        weight_init: Literal["default", "zeros", "fan_in"] = "default",
    ):
        """Initialise the Unpatchify layer.

        Args:
            in_features: Embedding dimension ``C_embed`` of each input token.
            out_features: Number of output channels ``C_out`` of the
                reconstructed signal (e.g. 3 for RGB images).
            data_dim: Spatial dimensionality.  Must be 1, 2, or 3.
            patch_size: Side length ``P`` of each patch.  The transposed
                convolution uses ``kernel_size = patch_size`` along every
                spatial axis.
            stride: Step between consecutive output patch origins.  Defaults to
                ``patch_size`` (non-overlapping, exact inverse of
                ``Patchify`` with default stride).  Must match the ``stride``
                used in the paired ``Patchify`` layer to recover the original
                spatial resolution.
            bias: If ``True`` (default), the deconvolution includes a learnable
                bias term.
            weight_init: Weight initialisation strategy for the deconv kernel.
                ``"default"`` uses PyTorch's built-in ``kaiming_uniform``
                (fan computed from ``out_features``; can cause output-variance
                blow-up for large ``in_features``).  ``"zeros"`` zero-inits
                weights and bias (DiT-style; output is exactly zero at
                initialisation, safe for residual-stream entry).  ``"fan_in"``
                applies Kaiming-uniform with the corrected fan-in
                ``in_features * patch_size^data_dim``, giving output variance
                O(1) regardless of embedding dimension.

        Raises:
            ValueError: If ``data_dim`` is not 1, 2, or 3.
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
            bias=bias,
        )

        if weight_init == "zeros":
            torch.nn.init.zeros_(self.deconv.weight)
            if self.deconv.bias is not None:
                torch.nn.init.zeros_(self.deconv.bias)
        elif weight_init == "fan_in":
            # True fan-in is in_features * kernel_size^data_dim; PyTorch's default
            # mistakenly uses out_features * kernel_size^data_dim for ConvTranspose.
            true_fan_in = in_features * (patch_size**data_dim)
            std = math.sqrt(2.0 / true_fan_in)
            torch.nn.init.normal_(self.deconv.weight, mean=0.0, std=std)
            if self.deconv.bias is not None:
                torch.nn.init.zeros_(self.deconv.bias)
        # "default": leave PyTorch's kaiming_uniform as-is

    def forward(self, x: torch.Tensor, output_spatial_shape: Tuple[int, ...] | None = None) -> torch.Tensor:
        """Reconstruct the spatial signal from a grid of patch-token embeddings.

        Args:
            x: Token-grid tensor in channels-last layout.  Shape:
                ``[B, *patch_grid, C_embed]``, e.g.
                ``[B, H/P, W/P, C_embed]`` for 2D.  The number of spatial
                dimensions must equal ``data_dim``.
            output_spatial_shape: Optional target spatial shape for the output.
                When provided this is passed as ``output_size`` to the
                transposed convolution, which resolves the output-size ambiguity
                that arises when ``stride > 1`` (multiple input sizes can map to
                the same output size via the floor in the forward formula).
                Must have length ``data_dim``.  When ``None``, PyTorch infers
                the output size (may differ from the original input spatial size
                if ``spatial_dim % patch_size != 0``).

        Returns:
            Reconstructed signal tensor in channels-last layout.  Shape:
            ``[B, *spatial_dims, C_out]``.  Without ``output_spatial_shape``,
            each axis ``s`` of the patch grid expands to
            ``(s - 1) * stride + patch_size``.

        Raises:
            AssertionError: If the rank of ``x`` does not equal
                ``data_dim + 2`` (batch + spatial + channel dims).

        Note:
            ``.contiguous()`` is called after the rearrangement to channels-first
            to avoid a stride-mismatch error in ``torch.compile``'s
            ``convolution_backward``.
        """
        expected_dims = self.data_dim + 2  # batch + spatial_dims + channels
        assert x.dim() == expected_dims, (
            f"Expected {expected_dims}D tensor for data_dim={self.data_dim}, got {x.dim()}D tensor with shape {tuple(x.shape)}"
        )

        # Channels-last -> channels-first for ConvTransposeNd
        # .contiguous() avoids a stride mismatch in torch.compile's convolution_backward
        x_bc = rearrange(x, "b ... c -> b c ...").contiguous()

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
