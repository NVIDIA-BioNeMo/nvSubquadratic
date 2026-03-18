"""RMSNorm — Root Mean Square Layer Normalization.

Uses QuACK's fused kernel on CUDA when available (Hopper/Blackwell: H100, B200, B300 only).
On other GPUs (e.g. Ampere), when quack is not installed, or on CPU, uses a pure-PyTorch
fallback. There is no quack-kernels build for Ampere; install quack only on H100/B200
for acceleration.
"""

import warnings

import torch
import torch.nn as nn


try:
    from quack import rmsnorm as _quack_rmsnorm

    _has_quack = True
except ImportError:
    _has_quack = False
    _quack_rmsnorm = None


def _rmsnorm_pytorch(x: torch.Tensor, weight: torch.nn.Parameter, eps: float) -> torch.Tensor:
    """Pure PyTorch RMSNorm (used when quack is unavailable or on CPU)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


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
        if _has_quack and x.is_cuda:
            try:
                return _quack_rmsnorm(x, self.weight, eps=self.eps)
            except RuntimeError as e:
                if "cudaErrorInvalidDeviceFunction" in str(e) or "CUDA" in str(e):
                    warnings.warn(
                        f"QuACK RMSNorm kernel failed ({e}); falling back to PyTorch. "
                        "Install quack-kernels for a GPU that supports it, or use CPU.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return _rmsnorm_pytorch(x, self.weight, self.eps)
                raise
        return _rmsnorm_pytorch(x, self.weight, self.eps)


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
