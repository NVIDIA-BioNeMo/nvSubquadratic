"""Global Response Normalisation (GRN) layer.

GRN is a channel-wise normalisation technique introduced in ConvNeXt V2
(Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
Autoencoders", arXiv:2301.00808, 2023).  It addresses *feature collapse*
observed when training ConvNets with masked-autoencoder pre-training: many
channels converge to identical activation patterns, so the network wastes
capacity.

**Core operation**

For an input ``x`` of shape ``[B, *spatial, C]`` (channels-last):

1. Compute the per-channel global L2 norm across all spatial positions::

       gx[b, c] = ||x[b, :, c]||_2          # shape [B, C]

2. Normalise each channel norm by the *mean* norm across channels
   (divisive / response normalisation step)::

       nx[b, c] = gx[b, c] / (mean_C(gx[b, c]) + eps)   # shape [B, C]

3. Rescale the input with learned scalar parameters ``γ`` (gamma) and
   ``β`` (beta), both zero-initialised so the layer starts as identity::

       y = γ * (x * nx) + β + x

   The ``+ x`` term is a residual connection that preserves the identity
   mapping at initialisation.  ``γ`` and ``β`` are 1-D tensors of size C
   that broadcast over the batch and all spatial dimensions.

   Note: ``x * nx`` is equivalent to weighting each channel of ``x`` by
   how strongly it activates relative to the cross-channel average — *not*
   a simple pointwise division by the spatial L2 norm.  The quantity ``nx``
   encodes *relative channel strength*, not normalisation to unit norm.

**Why GRN instead of LayerNorm inside gated MLPs**

LayerNorm standardises each token's feature vector independently, which
can suppress the *relative* differences between channels.  GRN instead
promotes *inter-channel competition*: channels with strong global
activations are amplified relative to weaker channels, encouraging each
channel to specialise.  This is particularly effective inside gated-linear
units (GLU / SwiGLU) where per-channel magnitude carries semantic weight.

Reference:
    Woo et al., "ConvNeXt V2", arXiv:2301.00808 (CVPR 2023).
"""

import torch
import torch.nn as nn


class GlobalResponseNorm(nn.Module):
    """Global Response Normalisation (GRN) layer (Woo et al., arXiv:2301.00808).

    Computes a per-channel global L2 norm across all spatial positions,
    normalises each channel norm by the cross-channel mean norm, then
    rescales the input with learned ``γ`` / ``β`` parameters plus a
    residual connection:

    .. code-block:: text

        gx  = ||x||_{spatial, L2}             # [B, 1, ..., 1, C]
        nx  = gx / (mean_C(gx) + eps)         # [B, 1, ..., 1, C]
        out = γ * (x * nx)  +  β  +  x        # [B, *spatial, C]

    ``γ`` and ``β`` are 1-D tensors of length ``C`` that broadcast over the
    batch dimension and all spatial dimensions.  They are zero-initialised
    so the layer starts as an identity (``out = x``) and the network can
    learn to activate the normalisation only where it is beneficial.

    **Inter-channel competition**

    The divisive step ``gx / mean_C(gx)`` produces values > 1 for channels
    whose global L2 norm exceeds the cross-channel average, and < 1 for
    weaker channels.  Multiplying the input by ``nx`` therefore amplifies
    dominant channels and suppresses weak ones, enforcing a form of
    *lateral inhibition* across the channel dimension.  This is the key
    mechanism by which GRN combats feature collapse (see module docstring).

    Unlike LayerNorm — which normalises each token's feature vector and
    discards inter-channel magnitude differences — GRN preserves and
    *amplifies* these differences, making it particularly effective inside
    gated MLPs (GLU / SwiGLU) where per-channel activation strength
    carries semantic weight.

    **Broadcast semantics**

    ``keepdim=True`` in the spatial reduction produces ``gx`` of shape
    ``[B, 1, ..., 1, C]`` (one singleton per spatial axis).  The mean is
    then taken along the *channel* axis (``dim=-1, keepdim=True``) to yield
    ``nx`` of the same shape.  The subsequent multiplication ``x * nx``
    broadcasts over all spatial positions without an explicit tile, so GRN
    is memory-efficient and agnostic to the number of spatial dimensions
    (1-D sequences, 2-D images, 3-D volumes, etc.).

    Attributes:
        dim (int): Number of channels C; must equal ``x.shape[-1]`` at
            every forward call.
        gamma (nn.Parameter): Learnable per-channel scale; shape ``(C,)``,
            zero-initialised.
        beta (nn.Parameter): Learnable per-channel bias; shape ``(C,)``,
            zero-initialised.
        eps (float): Small positive constant added to ``mean_C(gx)`` in the
            denominator to prevent division by zero.

    Reference:
        Woo et al., "ConvNeXt V2", arXiv:2301.00808 (CVPR 2023),
        Sec. 3 "Global Response Normalization".
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialise GRN with zero-initialised gamma and beta.

        Args:
            dim: Number of channels C.  Must match the size of the last
                dimension of every input tensor passed to ``forward``.
                Determines the shape of ``gamma`` and ``beta``.
            eps: Small positive constant added to ``mean_C(gx)`` in the
                denominator for numerical stability.  Defaults to ``1e-6``.
        """
        super().__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def flop_count(self, num_tokens: int) -> int:
        """Return the approximate FLOP count for one forward pass.

        Let T = ``num_tokens`` (total number of spatial positions summed
        over the batch, i.e. ``B * prod(spatial_shape)``) and C = ``self.dim``.
        The cost is dominated by element-wise operations over the T × C
        activation grid:

        * **Squared L2 norm per channel** — element-wise square + reduction
          sum over T positions per channel → 2 · T · C FLOPs.
        * **Square root per channel** — C FLOPs (negligible vs T · C,
          included for completeness).
        * **Cross-channel mean** — C additions → C FLOPs.
        * **Division** ``gx / (mean_C(gx) + eps)`` — C FLOPs.
        * **Broadcast multiply** ``x * nx`` — T · C FLOPs.
        * **Scale** ``γ * (x * nx)`` — T · C FLOPs.
        * **Add beta and residual** — 2 · T · C FLOPs.

        Total: approximately **6 · T · C** FLOPs.

        Args:
            num_tokens: Total number of spatial positions in the batch,
                i.e. ``B * prod(spatial_shape)``.  Note this includes the
                batch dimension: for a batch of 8 images of size 32×32 the
                value is ``8 * 32 * 32 = 8192``.

        Returns:
            Estimated integer FLOP count for one forward pass.
        """
        return 6 * num_tokens * self.dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Global Response Normalisation to the input tensor.

        The input must be in **channels-last** layout: the channel axis is
        the *last* dimension and all intermediate dimensions are spatial.
        The batch axis is always the first dimension.

        Args:
            x: Input activation tensor of shape ``[B, *spatial, C]``, where

                * ``B`` — batch size,
                * ``*spatial`` — any number (≥ 1) of spatial dimensions
                  (e.g. ``(L,)`` for 1-D sequences, ``(H, W)`` for 2-D
                  images, ``(D, H, W)`` for 3-D volumes),
                * ``C`` — number of channels; must equal ``self.dim``.

        Returns:
            torch.Tensor: Output tensor of shape ``[B, *spatial, C]``, the
            same dtype and device as ``x``, with GRN applied:
            ``γ * (x * nx) + β + x``.

        Raises:
            RuntimeError: If ``x.shape[-1]`` does not equal ``self.dim``
                (raised implicitly when broadcasting ``self.gamma`` /
                ``self.beta`` against a mismatched channel dimension).
        """
        spatial_dims = tuple(range(1, x.ndim - 1))
        gx = torch.norm(x, p=2, dim=spatial_dims, keepdim=True)  # [B, 1..., C]
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)  # [B, 1..., C]
        return self.gamma * (x * nx) + self.beta + x
