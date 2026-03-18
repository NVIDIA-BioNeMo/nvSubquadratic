"""RMSNorm — Root Mean Square Layer Normalization.

Uses QuACK's fused Triton kernel on CUDA, with a pure-PyTorch fallback for CPU.
"""

import torch
import torch.nn as nn
from quack import rmsnorm as _quack_rmsnorm


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Set dimension and epsilon for normalization."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight._no_weight_decay = True
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over last dimension and scale by weight."""
        if x.is_cuda:
            return _quack_rmsnorm(x, self.weight, eps=self.eps)
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class PerHeadRMSNorm(nn.Module):
    """RMSNorm applied independently to each head.

    Accepts [..., hidden_dim], reshapes to [..., num_heads, head_dim],
    normalizes over head_dim, and flattens back. Each head has its own
    learnable scale vector of size head_dim.
    """

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        """Store head layout and create per-head RMSNorm."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm = RMSNorm(dim=head_dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each head independently over head_dim."""
        shape = x.shape
        x = x.view(*shape[:-1], self.num_heads, self.head_dim)
        x = self.norm(x)
        return x.view(shape)

    def extra_repr(self) -> str:
        """Return head layout for repr()."""
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}"
