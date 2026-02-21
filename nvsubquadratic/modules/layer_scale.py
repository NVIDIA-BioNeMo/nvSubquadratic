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
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
        self.gamma._no_weight_decay = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
