# TODO: Add license header here


"""Learned positional encodings for ND inputs."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PositionEmbeddingND(nn.Module):
    """Learned positional encoding for 1D/2D/3D ... inputs."""

    def __init__(self, embedding_dim: int, data_dim: int, max_dim_lengths: Sequence[int]):
        """Configure positional embeddings for a fixed set of spatial dimensions."""
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
        """Return the concatenated per-axis embeddings for the input grid."""
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
