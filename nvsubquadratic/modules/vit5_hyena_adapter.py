"""Adapter to plug 2D sequence mixers (e.g. Hyena) into the ViT5 token-sequence architecture.

The ViT5 architecture processes [B, T, C] sequences. 2D mixers like Hyena expect
[B, H, W, C] spatial grids. This adapter reshapes the flat token sequence to a 2D
grid, applies the inner mixer, and reshapes back.

All token ordering (CLS position, register placement) is handled upstream by the
network (e.g. ViT5ClassificationNet with prepend_registers=True), so this adapter
treats the entire sequence as a flat spatial grid — it does not know or care about
which tokens are CLS, registers, or patches.
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
        """Initialize ViT5HyenaAdapter."""
        super().__init__()
        self.inner_mixer = instantiate(inner_mixer_cfg)
        self.grid_w = grid_w

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
        """Return extra representation string."""
        return f"grid_w={self.grid_w}"
