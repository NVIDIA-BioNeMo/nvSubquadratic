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

    _quack_available = True
except ImportError:
    _quack_rmsnorm = None
    _quack_available = False


def _cuda_supports_quack(device: torch.device) -> bool:
    """QuACK kernels are built for Hopper/Blackwell (SM 9.x). Ampere (8.x) and older fail in forward or backward."""
    if not device.type == "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 9


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

    def flop_count(self, num_tokens: int) -> int:
        """Count FLOPs for RMS normalization over ``num_tokens`` token vectors.

        Operations per token (D = ``self.weight.shape[0]``):
          1. Square each element:           D FLOPs
          2. Mean over D + rsqrt:           D FLOPs (amortized reduction + 1 rsqrt)
          3. Multiply by learned scale:     D FLOPs

        Total: 3 * num_tokens * D.

        Args:
            num_tokens: Number of token vectors being normalized.

        Returns:
            Total FLOPs as an integer.
        """
        dim = self.weight.shape[0]
        return 3 * num_tokens * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over last dimension and scale by weight."""
        # Only use quack on GPUs that support it (SM 9+). On Ampere (8.x), quack backward fails;
        # we must not call quack at all so autograd never invokes its backward.
        if _quack_available and x.is_cuda and _cuda_supports_quack(x.device):
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

    def flop_count(self, num_tokens: int) -> int:
        """Count FLOPs for per-head RMS normalization on ``num_tokens`` tokens.

        Each token is reshaped to [num_heads, head_dim] and RMSNorm is applied
        independently per head.  Total cost is the same as a full RMSNorm over
        ``hidden_dim = num_heads * head_dim``:

        Total: 3 * num_tokens * num_heads * head_dim.

        Args:
            num_tokens: Number of token vectors being normalized.

        Returns:
            Total FLOPs as an integer.
        """
        return 3 * num_tokens * self.num_heads * self.head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize each head independently over head_dim."""
        shape = x.shape
        x = x.view(*shape[:-1], self.num_heads, self.head_dim)
        x = self.norm(x)
        return x.view(shape)

    def extra_repr(self) -> str:
        """Return head layout for repr()."""
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}"
