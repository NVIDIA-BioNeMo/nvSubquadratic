"""DropPath (Stochastic Depth): randomly drop entire residual branches during training.

Reference: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
"""

import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample.

    Args:
        drop_prob: Probability of dropping the path. 0.0 means no drop.
    """

    def __init__(self, drop_prob: float = 0.0):
        """Initialize DropPath with the given drop probability."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"drop_prob={self.drop_prob:.3f}"
