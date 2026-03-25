"""LayerScale: learnable per-channel scaling of residual branch outputs.

Reference: Touvron et al., "Going deeper with Image Transformers" (CaiT), ICCV 2021.
Used in ViT-5 as a default component for training stability.
"""

import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """Learnable diagonal scaling applied element-wise to the channel dimension.

    Given input x of shape (..., dim), returns x * diag(gamma) where gamma
    is a learnable vector initialized to ``init_value``.

    Args:
        dim: Number of channels.
        init_value: Initial value for the scaling vector (typically 1e-4).
    """

    def __init__(self, dim: int, init_value: float = 1e-4):
        """Initialize learnable scale vector to init_value."""
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
        self.gamma._no_weight_decay = True

    def flop_count(self, num_tokens: int) -> int:
        """Count FLOPs for per-channel scaling of ``num_tokens`` token vectors.

        Each token of dimension D = ``self.gamma.shape[0]`` is multiplied
        element-wise by the learned gamma vector: D FLOPs per token.

        Total: num_tokens * D.

        Args:
            num_tokens: Number of token vectors being scaled.

        Returns:
            Total FLOPs as an integer.
        """
        dim = self.gamma.shape[0]
        return num_tokens * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input by per-channel gamma."""
        return x * self.gamma
