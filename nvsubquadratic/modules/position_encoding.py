# TODO: Add license header here


"""Position encoding layers for ND spatial token grids.

Overview
--------
Sequence mixers (Hyena, Attention, CKConv, …) operate on a flat sequence of
token embeddings and are in principle permutation-equivariant.  *Position
encodings* break that symmetry by injecting spatial-coordinate information
directly into the embedding space so the mixer can reason about where each
token sits in the original spatial grid.

In this codebase position encodings are applied **after** patch embedding (see
``nvsubquadratic.modules.patchify.Patchify``): the ``Patchify`` layer maps the
raw input of shape ``[B, *spatial, C_in]`` to a patch-token grid of shape
``[B, *patch_grid, C_embed]``, and then a position encoding layer adds a
coordinate-dependent bias of the same shape before the tokens are passed into
the first mixer block.

Variants
--------
``PositionEmbeddingND``
    **Fully-learned / lookup-table** encoding.  A separate ``nn.Embedding``
    table is trained for each spatial axis; the per-axis embeddings are
    concatenated channel-wise.  Requires the grid size to be bounded at
    construction time (``max_dim_lengths``) but makes no structural assumption
    about how position information should be represented.  The embedding
    parameters are tagged ``_no_weight_decay = True`` so they are excluded from
    weight-decay optimiser groups.

    Best suited for fixed-resolution training where the spatial grid size does
    not change between train and inference.  For variable-resolution or
    resolution-generalisation use-cases, prefer a sinusoidal or RFF-based
    encoding (not yet included in this module).

ND generalisation strategy
--------------------------
A naive learned positional encoding for an ND grid would store
``prod(max_dim_lengths)`` parameters — exponential in the number of dimensions.
``PositionEmbeddingND`` avoids this by factorising the encoding across axes: a
separate embedding table of length ``max_dim_lengths[d]`` is kept for each
spatial axis ``d``, and the per-axis embeddings of dimension
``embedding_dim // data_dim`` are concatenated to produce the final
``embedding_dim``-dimensional token offset.  The total parameter count is
therefore ``data_dim * max(max_dim_lengths) * (embedding_dim // data_dim) =
embedding_dim * max(max_dim_lengths)`` — linear in the embedding dimension and
the largest axis length.

Cross-references
----------------
* ``nvsubquadratic.modules.patchify.Patchify`` — produces the patch-token grid
  that is the expected input to the ``forward`` method of position-encoding
  layers in this module.
* ``nvsubquadratic.modules.kernels_nd.SIRENKernelND`` — uses a similar spatial
  coordinate grid to parametrise implicit convolutional kernels.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PositionEmbeddingND(nn.Module):
    """Axis-factorised learned positional encoding for ND spatial token grids.

    Each spatial axis has its own ``nn.Embedding`` table of shape
    ``(max_dim_lengths[d], embedding_dim // data_dim)``.  For a grid position
    ``(i_0, i_1, ..., i_{D-1})`` the encoding is formed by looking up the
    per-axis embeddings and *concatenating* them channel-wise::

        PE(i_0, ..., i_{D-1}) = [E_0(i_0) ‖ E_1(i_1) ‖ ... ‖ E_{D-1}(i_{D-1})]

    where ``‖`` denotes concatenation along the channel dimension and each
    ``E_d ∈ R^{max_dim_lengths[d] × (embedding_dim // data_dim)}`` is a
    learned embedding matrix.  The result has length ``embedding_dim``
    (requires ``embedding_dim % data_dim == 0``).

    This factorised form is a *separable* positional encoding: the position
    information for axis ``d`` is captured entirely in the slice of channels
    ``[d * per_dim, (d+1) * per_dim)``.  It is not a *joint* encoding of the
    full ND coordinate (unlike 2D sinusoidal encodings that mix axes), so
    cross-axis interactions must be learned by the mixer layers themselves.

    **Output shape**::

        forward(x) -> Tensor of shape [B, *spatial_dims, embedding_dim]

    The returned tensor is a broadcast-expanded encoding grid with the *same
    shape as the input* ``x``, and is typically **added** to ``x`` before the
    first mixer block::

        x = x + position_embedding(x)

    **Parameter count** — ``data_dim`` embedding tables, each of size
    ``max_dim_lengths[d] × (embedding_dim // data_dim)``::

        total = sum(max_dim_lengths[d] * (embedding_dim // data_dim)
                    for d in range(data_dim))

    **No weight decay** — all embedding parameters are tagged
    ``param._no_weight_decay = True`` so that optimiser builders (e.g. those
    using ``param._no_weight_decay`` to separate param groups) can exclude them
    from L2 regularisation, following the standard ViT practice.

    Attributes:
        embedding_dim (int): Total embedding dimension of the output encoding.
        per_dim_embedding_dim (int): Per-axis slice width,
            ``embedding_dim // data_dim``.
        data_dim (int): Number of spatial axes (1, 2, or 3).
        max_dim_lengths (tuple[int, ...]): Maximum supported grid size for each
            spatial axis.  Length equals ``data_dim``.
        data_embeddings (nn.ModuleDict): Dictionary mapping axis keys
            ``{"x"}`` / ``{"x", "y"}`` / ``{"x", "y", "z"}`` to the
            corresponding ``nn.Embedding`` modules.
    """

    def __init__(self, embedding_dim: int, data_dim: int, max_dim_lengths: Sequence[int]):
        """Initialise per-axis embedding tables.

        Args:
            embedding_dim: Total number of channels in the output position
                encoding.  Must be divisible by ``data_dim`` so that the
                budget can be split evenly across axes.
            data_dim: Number of spatial axes of the input token grid.  Must
                satisfy ``1 <= data_dim <= 3``.
            max_dim_lengths: Maximum grid size for each spatial axis.  Must
                have exactly ``data_dim`` entries.  An ``nn.Embedding`` table
                of length ``max_dim_lengths[d]`` is allocated for axis ``d``;
                inputs whose spatial size along axis ``d`` exceeds this value
                will raise a ``ValueError`` in ``forward``.

        Raises:
            AssertionError: If ``data_dim < 1``.
            AssertionError: If ``len(max_dim_lengths) != data_dim``.
            AssertionError: If ``data_dim > 3``.
            AssertionError: If ``embedding_dim % data_dim != 0``.
        """
        super().__init__()
        assert data_dim >= 1, "data_dim must be >= 1"
        assert len(max_dim_lengths) == data_dim, "max_dim_lengths must have length data_dim"
        assert data_dim <= 3, "data_dim must be <= 3"
        assert embedding_dim % data_dim == 0, "embedding_dim must be divisible by data_dim"

        self.embedding_dim = embedding_dim
        self.per_dim_embedding_dim = embedding_dim // data_dim
        self.data_dim = data_dim
        self.max_dim_lengths = tuple(int(m) for m in max_dim_lengths)

        keys = ["x", "y", "z"][:data_dim]
        self.data_embeddings = nn.ModuleDict(
            {key: nn.Embedding(m, self.per_dim_embedding_dim) for key, m in zip(keys, self.max_dim_lengths)}
        )

        for embedding in self.data_embeddings.values():
            for param in embedding.parameters():
                param._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the position encoding grid for the given input token grid.

        For each spatial axis ``d``, position indices ``0, 1, ..., L_d - 1``
        are looked up in the corresponding ``nn.Embedding`` table to produce a
        1-D embedding slice of shape ``[L_d, per_dim_embedding_dim]``.  That
        slice is broadcast-expanded to the full spatial shape
        ``[B, *spatial_dims, per_dim_embedding_dim]`` and the per-axis tensors
        are concatenated along the last dimension to give the final encoding of
        shape ``[B, *spatial_dims, embedding_dim]``.

        The returned tensor has the same shape as ``x`` and should be **added**
        to ``x``::

            x = x + position_embedding(x)

        Args:
            x: Input token-grid tensor in channels-last layout.  Shape:
                ``[B, *spatial_dims, embedding_dim]``, where the number of
                spatial axes must equal ``data_dim`` and the last dimension
                must equal ``embedding_dim``.  For example:

                - 1-D (sequences):  ``[B, L, C]``
                - 2-D (images):     ``[B, H, W, C]``
                - 3-D (volumes):    ``[B, D, H, W, C]``

        Returns:
            Position encoding tensor of shape
            ``[B, *spatial_dims, embedding_dim]`` — the same shape as ``x``.
            Each spatial location ``(i_0, ..., i_{D-1})`` holds the
            concatenation of the per-axis embedding lookups::

                out[b, i_0, ..., i_{D-1}, :] =
                    [E_0(i_0) ‖ E_1(i_1) ‖ ... ‖ E_{D-1}(i_{D-1})]

        Raises:
            ValueError: If ``x.ndim != data_dim + 2`` (wrong number of
                dimensions — expected batch + ``data_dim`` spatial + channel).
            ValueError: If ``x.shape[-1] != embedding_dim`` (channel dimension
                mismatch).
            ValueError: If any spatial dimension of ``x`` exceeds the
                corresponding entry of ``max_dim_lengths``.
        """
        if x.ndim != self.data_dim + 2:
            raise ValueError(
                f"PositionEmbeddingND expects input of shape [B, *{self.data_dim} dims, C]; got {tuple(x.shape)}"
            )
        if x.shape[-1] != self.embedding_dim:
            raise ValueError(f"Last dim {x.shape[-1]} must match embedding_dim {self.embedding_dim}")

        batch_size = x.shape[0]
        spatial_dims = x.shape[1:-1]
        if any(Ld > Md for Ld, Md in zip(spatial_dims, self.max_dim_lengths)):
            raise ValueError(f"Input spatial dims {tuple(spatial_dims)} exceed max_dim_lengths {self.max_dim_lengths}")

        axis_embs = []
        for axis, embedding in enumerate(self.data_embeddings.values()):
            length_axis = spatial_dims[axis]
            pos_ids = torch.arange(length_axis, device=x.device, dtype=torch.long)
            emb_axis = embedding(pos_ids)
            shape = [1] * self.data_dim
            shape[axis] = length_axis
            emb_axis = emb_axis.view(1, length_axis, self.per_dim_embedding_dim)
            emb_axis = emb_axis.view(1, *shape, self.per_dim_embedding_dim)
            emb_axis = emb_axis.expand(batch_size, *spatial_dims, self.per_dim_embedding_dim)
            axis_embs.append(emb_axis)

        return torch.cat(axis_embs, dim=-1)
