"""Adapter to plug 2D sequence mixers (e.g. Hyena) into the ViT5 token-sequence architecture.

The ViT5 architecture processes [B, T, C] sequences. 2D mixers like Hyena expect
[B, H, W, C] spatial grids. This adapter reshapes the flat token sequence to a 2D
grid, applies the inner mixer, and reshapes back.

Token ordering is handled upstream by the network (ViT5ClassificationNet).
The standard layout is [patches, CLS, registers, padding] where padding
ensures T is divisible by grid_w.  This adapter is layout-agnostic: it treats
the entire sequence as a flat spatial grid reshaped to (T // grid_w, grid_w).
"""

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
                Standard layout: [patches (H*W), CLS (1), registers (R), padding (P)].
                The adapter is layout-agnostic and reshapes the full sequence
                to a 2D grid of shape (T // grid_w, grid_w).
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
