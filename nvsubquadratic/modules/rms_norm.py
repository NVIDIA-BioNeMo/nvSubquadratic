"""RMSNorm — Root Mean Square Layer Normalization.

Uses QuACK's fused kernel on CUDA when available (Hopper/Blackwell: H100, B200, B300 only)
and ``use_quack=True`` (the default).  On other GPUs (e.g. Ampere), when quack is not
installed, on CPU, or when ``use_quack=False``, uses a pure-PyTorch fallback.

Set ``use_quack=False`` when running under ``torch.compile`` to let the compiler
fuse the norm with adjacent operations instead of treating the QuACK kernel as an
opaque barrier.
"""

import warnings

import torch
import torch.nn as nn

from nvsubquadratic.utils.quack_utils import cuda_supports_quack as _cuda_supports_quack


try:
    from quack import rmsnorm as _quack_rmsnorm

    _quack_available = True
except ImportError:
    _quack_rmsnorm = None
    _quack_available = False


def _rmsnorm_pytorch(x: torch.Tensor, weight: torch.nn.Parameter, eps: float) -> torch.Tensor:
    """Pure PyTorch RMSNorm (used when quack is unavailable or on CPU)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes over the last dimension and scales by a learnable weight.
    Accepts tensors of any shape ``[..., dim]``.

    Two backends are available:

    - **QuACK** (default): Fused CUDA kernel from the ``quack`` package.
      Only runs on Hopper/Blackwell GPUs (SM 9.0+).  Opaque to
      ``torch.compile`` — acts as a fusion barrier.
    - **PyTorch**: Pure ``torch`` ops (``pow``, ``mean``, ``rsqrt``).  Fully
      visible to ``torch.compile``, enabling fusion with adjacent operations.

    Args:
        dim: Size of the last dimension to normalize over.
        eps: Small constant added to the variance for numerical stability.
        use_quack: If ``True`` (default), use the QuACK fused kernel when
            available.  Set to ``False`` to force the PyTorch path, which
            lets ``torch.compile`` fuse the norm with surrounding ops.
    """

    def __init__(self, dim: int, eps: float = 1e-6, use_quack: bool = True):
        """Initialize RMSNorm."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight._no_weight_decay = True
        self.eps = eps
        self.use_quack = use_quack

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
        if self.use_quack and _quack_available and x.is_cuda and _cuda_supports_quack(x.device):
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

    Accepts ``[..., hidden_dim]``, reshapes to ``[..., num_heads, head_dim]``,
    normalizes over ``head_dim``, and flattens back.  Each head has its own
    learnable scale vector of size ``head_dim``.

    Args:
        num_heads: Number of attention heads.
        head_dim: Dimension of each head.
        eps: Small constant added to the variance for numerical stability.
        use_quack: If ``True`` (default), use the QuACK fused kernel when
            available.  Set to ``False`` to force the PyTorch path, which
            lets ``torch.compile`` fuse the norm with surrounding ops.
    """

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6, use_quack: bool = True):
        """Initialize PerHeadRMSNorm."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm = RMSNorm(dim=head_dim, eps=eps, use_quack=use_quack)

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
