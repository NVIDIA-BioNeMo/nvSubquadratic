"""RMSNorm — Channel-First variant (normalizes over dim=1).

Uses QuACK's fused channel-first kernel on CUDA when available (Hopper/Blackwell:
H100, B200, B300 only) and ``use_quack=True`` (the default).  On other GPUs,
when quack is not installed, on CPU, or when ``use_quack=False``, uses a
pure-PyTorch fallback.

Set ``use_quack=False`` when running under ``torch.compile`` to let the compiler
fuse the norm with adjacent operations instead of treating the QuACK kernel as an
opaque barrier.

Expects input tensors in channel-first layout, e.g. ``[B, C, H, W]`` or
``[B, C, T]``.  Use this only at call sites where the data is *already*
channel-first (e.g. inside the Hyena mixer).  For trunk code where tensors
are ``[B, T, C]``, use the regular channel-last ``RMSNorm`` instead.

Modules that handle channel-first norms specially (e.g. ``Hyena``) can
duck-type on the ``channels_first`` attribute (always ``True`` here).
"""

import torch
import torch.nn as nn

from nvsubquadratic.utils.quack_utils import cuda_supports_quack as _cuda_supports_quack


try:
    from quack import rmsnorm_channel_first as _quack_rmsnorm_cf

    _quack_cf_available = True
except ImportError:
    _quack_rmsnorm_cf = None
    _quack_cf_available = False


def _rmsnorm_channel_first_pytorch(x: torch.Tensor, weight: torch.nn.Parameter, eps: float) -> torch.Tensor:
    """Pure PyTorch channel-first RMSNorm: normalize over dim=1."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    if weight is not None:
        shape = [1, -1] + [1] * (x.dim() - 2)
        x = x * weight.float().view(shape)
    return x.to(input_dtype)


class RMSNormChannelFirst(nn.Module):
    """Root Mean Square Layer Normalization — channel-first layout.

    Normalizes over ``dim=1`` (the channel dimension) and scales by a learnable
    weight.  Accepts tensors of shape ``[B, C, *spatial]`` (e.g. ``[B, C, H, W]``).

    Two backends are available:

    - **QuACK** (default): Fused CUDA kernel from the ``quack`` package
      (``rmsnorm_channel_first``).  Only runs on Hopper/Blackwell GPUs
      (SM 9.0+).  Opaque to ``torch.compile`` — acts as a fusion barrier.
    - **PyTorch**: Pure ``torch`` ops (``pow``, ``mean``, ``rsqrt``).  Fully
      visible to ``torch.compile``, enabling fusion with adjacent operations.

    Args:
        dim: Number of channels (size of dim=1) to normalize over.
        eps: Small constant added to the variance for numerical stability.
        use_quack: If ``True`` (default), use the QuACK fused kernel when
            available.  Set to ``False`` to force the PyTorch path, which
            lets ``torch.compile`` fuse the norm with surrounding ops.
    """

    channels_first: bool = True

    def __init__(self, dim: int, eps: float = 1e-6, use_quack: bool = True):
        """Initialize RMSNormChannelFirst."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight._no_weight_decay = True
        self.eps = eps
        self.use_quack = use_quack

    def flop_count(self, num_tokens: int) -> int:
        """Count FLOPs — identical cost to channel-last RMSNorm."""
        dim = self.weight.shape[0]
        return 3 * num_tokens * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize over channel dimension (dim=1) and scale by weight."""
        if self.use_quack and _quack_cf_available and x.is_cuda and _cuda_supports_quack(x.device):
            return _quack_rmsnorm_cf(x, self.weight, eps=self.eps)
        return _rmsnorm_channel_first_pytorch(x, self.weight, self.eps)
