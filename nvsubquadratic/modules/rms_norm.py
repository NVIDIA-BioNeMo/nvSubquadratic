"""RMSNorm — Root Mean Square Layer Normalization.

Uses QuACK's fused Triton kernel on CUDA when available, with a
pure-PyTorch fallback for CPU / environments without QuACK.
"""

import torch
import torch.nn as nn

try:
    from quack import rmsnorm as _quack_rmsnorm
except ImportError:
    import warnings
    warnings.warn(
        "quack.rmsnorm not found — falling back to pure-PyTorch RMSNorm. "
        "Install QuACK for a fused Triton RMSNorm kernel.",
        stacklevel=2,
    )
    _quack_rmsnorm = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight._no_weight_decay = True
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _quack_rmsnorm is not None and x.is_cuda:
            return _quack_rmsnorm(x, self.weight, eps=self.eps)
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)
