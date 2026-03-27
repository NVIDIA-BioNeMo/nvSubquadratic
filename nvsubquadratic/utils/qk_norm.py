# TODO: Add license header here


"""QK normalization utilities."""

import torch
import torch.nn.functional as F


def apply_qk_norm(query: torch.Tensor, key: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    """L2-normalize query and key along the given dimension (e.g. for QK-norm in attention).

    Returns:
        Tuple of (query_normalized, key_normalized), same shapes as inputs.
    """
    query = F.normalize(query, p=2.0, dim=dim, eps=eps)
    key = F.normalize(key, p=2.0, dim=dim, eps=eps)
    return query, key


class L2Norm(torch.nn.Module):
    """L2 normalization as a module, for use as a LazyConfig target.

    Normalizes along the last dimension by default, matching the convention
    of torch.nn.RMSNorm and torch.nn.LayerNorm.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-12):
        """Store normalization dimension and epsilon."""
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalize along the configured dimension."""
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)

    def extra_repr(self) -> str:
        """Return dim and eps for repr()."""
        return f"dim={self.dim}, eps={self.eps}"
