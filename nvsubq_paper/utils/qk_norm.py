# TODO: Add license header here


"""QK normalization utilities."""

import torch
import torch.nn.functional as F


def apply_qk_norm(
    query: torch.Tensor, key: torch.Tensor, dim: int = -1, eps: float = 1e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply L2 normalization to query and key tensors."""
    norm = L2Norm(dim=dim, eps=eps)
    return norm(query), norm(key)


class L2Norm(torch.nn.Module):
    """L2 normalization as a module, for use as a LazyConfig target.

    Normalizes along the last dimension by default, matching the convention
    of torch.nn.RMSNorm and torch.nn.LayerNorm.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"
