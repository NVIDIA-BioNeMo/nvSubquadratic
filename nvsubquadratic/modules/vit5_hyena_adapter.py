"""Adapter to plug 2D sequence mixers (e.g. Hyena) into the ViT5 token-sequence architecture.

The ViT5 architecture processes [B, T, C] sequences where T includes a CLS token,
flattened patch tokens, and optional register tokens. 2D mixers like Hyena expect
[B, H, W, C] spatial grids.

This adapter:
1. Strips CLS and register tokens (they can't participate in spatial convolutions)
2. Reshapes patch tokens from [B, num_patches, C] to [B, H', W', C]
3. Applies the inner 2D mixer on the spatial grid
4. Reshapes back to [B, num_patches, C]
5. Replaces the CLS token with a mean-pool of the mixed patches so that the
   ViT5ResidualBlock residual connection accumulates spatial information into CLS
6. Re-concatenates [CLS, patches, registers] back to [B, T, C]
"""

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5HyenaAdapter(nn.Module):
    """Bridges ViT5's [B, T, C] token sequences and Hyena's [B, H, W, C] spatial interface.

    Args:
        inner_mixer_cfg: LazyConfig for the 2D sequence mixer (e.g. QKVSequenceMixer wrapping Hyena).
        num_patches_h: Height of the 2D patch grid.
        num_patches_w: Width of the 2D patch grid.
        num_registers: Number of register tokens appended after patches (0 to disable).
    """

    def __init__(
        self,
        inner_mixer_cfg: LazyConfig,
        num_patches_h: int,
        num_patches_w: int,
        num_registers: int = 0,
    ):
        super().__init__()
        self.inner_mixer = instantiate(inner_mixer_cfg)
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.num_registers = num_registers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, C] where T = 1 (CLS) + num_patches_h * num_patches_w + num_registers.

        Returns:
            [B, T, C] with patch tokens mixed via 2D Hyena and CLS updated via mean-pool.
        """
        B, T, C = x.shape
        num_patches = self.num_patches_h * self.num_patches_w

        patch_tokens = x[:, 1 : 1 + num_patches, :]

        # Reshape to 2D spatial grid and apply the inner mixer
        patch_2d = patch_tokens.reshape(B, self.num_patches_h, self.num_patches_w, C)
        patch_2d = self.inner_mixer(patch_2d)
        patch_tokens = patch_2d.reshape(B, num_patches, C)

        # CLS ← mean-pool of mixed patches.
        # The residual connection in ViT5ResidualBlock accumulates:
        #   cls_new = cls_old + DropPath(LayerScale(mean_patches))
        cls_token = patch_tokens.mean(dim=1, keepdim=True)

        parts = [cls_token, patch_tokens]
        if self.num_registers > 0:
            parts.append(x[:, 1 + num_patches :, :])
        return torch.cat(parts, dim=1)

    def extra_repr(self) -> str:
        return (
            f"patches=({self.num_patches_h}x{self.num_patches_w}), "
            f"num_registers={self.num_registers}"
        )
