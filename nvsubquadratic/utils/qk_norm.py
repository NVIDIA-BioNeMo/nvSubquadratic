# TODO: Add license header here


"""QK normalization utilities."""

import torch
import torch.nn.functional as F


class L2Norm(torch.nn.Module):
    """L2 normalization as a module, for use as a LazyConfig target.

    Normalizes along the last dimension by default, matching the convention
    of torch.nn.RMSNorm and torch.nn.LayerNorm.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-12):
        """Initialize L2Norm."""
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return f"dim={self.dim}, eps={self.eps}"
