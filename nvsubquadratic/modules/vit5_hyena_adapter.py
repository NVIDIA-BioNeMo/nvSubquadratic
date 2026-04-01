"""Adapters to plug ND sequence mixers (e.g. Hyena) into the ViT5 token-sequence architecture.

The ViT5 architecture processes [B, T, C] sequences. ND mixers like Hyena expect
[B, *spatial_dims, C] spatial grids. These adapters reshape the flat token sequence
to an ND grid, apply the inner mixer, and reshape back.

All token ordering (CLS position, register placement) is handled upstream by the
network (e.g. ViT5ClassificationNet with prepend_registers=True), so these adapters
treat the entire sequence as a flat spatial grid — they do not know or care about
which tokens are CLS, registers, or patches.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5HyenaAdapter(nn.Module):
    """Bridges ViT5's [B, T, C] token sequences and Hyena's [B, H, W, C] spatial interface.

    Args:
        inner_mixer_cfg: LazyConfig for the 2D sequence mixer (e.g. QKVSequenceMixer wrapping Hyena).
        grid_w: Width of the 2D spatial grid. The height is inferred as T // grid_w.
    """

    def __init__(
        self,
        inner_mixer_cfg: LazyConfig,
        grid_w: int,
    ):
        """Store config and instantiate the inner 2D mixer."""
        super().__init__()
        self.inner_mixer = instantiate(inner_mixer_cfg)
        self.grid_w = grid_w

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        """Count FLOPs for the Hyena adapter (reshape + inner mixer).

        Reshapes the flat token sequence [B, T, C] into a 2D spatial grid
        [B, H, W, C] (where W = ``self.grid_w``, H = T // W) and delegates
        to the inner mixer (QKVSequenceMixer wrapping Hyena).

        The reshape itself is a free metadata operation — no FLOPs.

        Args:
            num_tokens: Total sequence length T.  Must be divisible by grid_w.
                ``spatial_dims`` is computed as ``(T // grid_w, grid_w)``.
            inference: Passed through to the inner mixer.

        Returns:
            Total FLOPs from the inner mixer.
        """
        spatial_dims = (num_tokens // self.grid_w, self.grid_w)
        return self.inner_mixer.flop_count(spatial_dims, inference=inference)

    def forward(self, x: torch.Tensor, **mixer_kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C] token sequence. T must be divisible by grid_w.
                When registers are present, Hyena accomodates them at the first row of the 2D grid,
                making the token at position (0, 0) the CLS token, and the (0, 1), ..., (0, num_registers-1)
                the register tokens.

            IMPORTANT: In the future we can have M < grid_w registers by appending grid_w - M zeros to the row.
            **mixer_kwargs: Forwarded to the inner mixer (e.g. ``conditioning`` for FiLM).

        Returns:
            [B, T, C] with tokens mixed via the 2D inner mixer.
        """
        B, T, C = x.shape
        x = x.reshape(B, T // self.grid_w, self.grid_w, C)
        x = self.inner_mixer(x, **mixer_kwargs)
        x = x.reshape(B, T, C)
        return x

    def extra_repr(self) -> str:
        """Return grid width for repr()."""
        return f"grid_w={self.grid_w}"


class ViT5HyenaAdapterND(nn.Module):
    """Bridges ViT5's [B, T, C] token sequences and Hyena's [B, *spatial_dims, C] spatial interface.

    Generalizes ViT5HyenaAdapter to arbitrary spatial dimensions (1D, 2D, 3D).
    Optionally strips prefix tokens (e.g. registers) before reshaping, applies the
    ND mixer to patch tokens only, then re-prepends the prefix unchanged. This is
    useful when the prefix tokens cannot be cleanly tiled into the ND grid (e.g. 14
    registers with an 8x8x8 3D patch grid).

    Args:
        inner_mixer_cfg: LazyConfig for the ND sequence mixer (e.g. QKVSequenceMixer wrapping Hyena).
        grid_shape: Spatial grid shape as a tuple, e.g. (D, H, W) for 3D or (H, W) for 2D.
            The product must equal the number of patch tokens (T - num_prefix_tokens).
        num_prefix_tokens: Number of tokens at the start of the sequence to exclude
            from the spatial reshape (e.g. CLS + registers). These are re-prepended
            after the mixer. Default: 0 (all tokens are spatial).
    """

    def __init__(
        self,
        inner_mixer_cfg: LazyConfig,
        grid_shape: Sequence[int],
        num_prefix_tokens: int = 0,
    ):
        """Store grid shape and instantiate the inner ND mixer."""
        super().__init__()
        self.inner_mixer = instantiate(inner_mixer_cfg)
        self.grid_shape = tuple(grid_shape)
        self.num_prefix_tokens = num_prefix_tokens

    def forward(self, x: torch.Tensor, **mixer_kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C] token sequence.
            **mixer_kwargs: Forwarded to the inner mixer (e.g. ``conditioning`` for FiLM).

        Returns:
            [B, T, C] with all tokens mixed via the ND inner mixer.
            Prefix tokens are expected to be zero-padded upstream so that they cleanly
            fit into one or more planes along the first spatial dimension.
        """
        B, T, C = x.shape

        if len(self.grid_shape) > 1:
            slice_size = math.prod(self.grid_shape[1:])
            # The first dimension dynamically expands to accommodate prefix tokens padding upstream
            first_dim = T // slice_size
            x = x.reshape(B, first_dim, *self.grid_shape[1:], C)
        else:
            x = x.reshape(B, T, C)

        x = self.inner_mixer(x, **mixer_kwargs)
        x = x.reshape(B, -1, C)

        return x

    def extra_repr(self) -> str:
        """Return grid shape and prefix info for repr()."""
        return f"grid_shape={self.grid_shape}, num_prefix_tokens={self.num_prefix_tokens}"
