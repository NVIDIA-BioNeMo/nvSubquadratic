"""RMSNorm — Channel-First variant (normalises over ``dim=1``).

This is the channel-first counterpart of :mod:`nvsubquadratic.modules.rms_norm`.
It applies Root Mean Square Layer Normalization (Zhang & Sennrich, NeurIPS 2019)
along the **channel axis** (``dim=1``) of tensors stored in channel-first layout
such as ``[B, C, H, W]`` or ``[B, C, T]``:

.. code-block:: text

    RMS(x)_b  = sqrt( (1/C) Σ_c x[b, c, ...]²  +  ε )   # scalar per sample
    out[b, c, ...] = (x[b, c, ...] / RMS(x)_b) * γ_c

where ``γ ∈ ℝ^C`` is a learned per-channel scale initialised to ones.

**When to use this module vs. :class:`~nvsubquadratic.modules.rms_norm.RMSNorm`**

- Use :class:`RMSNormChannelFirst` where data is *already* channel-first
  (e.g. inside :class:`~nvsubquadratic.modules.hyena_nd.HyenaOperatorND` which
  uses ``[B, C, *spatial]`` tensors throughout).
- Use the regular channel-last :class:`~nvsubquadratic.modules.rms_norm.RMSNorm`
  in trunk code where tensors are ``[B, T, C]`` (ViT / transformer layout).

**Duck-typing sentinel**

The class-level attribute ``channels_first = True`` lets callers (e.g.
``HyenaOperatorND``) detect the layout at construction time without an
``isinstance`` check.

**Backends** (same policy as :mod:`nvsubquadratic.modules.rms_norm`):

- **QuACK** (default, ``use_quack=True``): fused CUDA kernel
  ``quack.rmsnorm_channel_first``; SM ≥ 9.0 only.
- **PyTorch** (``use_quack=False`` or fallback): pure ``torch`` ops, upcasts
  to float32 for numerical safety; fully visible to ``torch.compile``.

Reference:
    Zhang, B. & Sennrich, R., "Root Mean Square Layer Normalization",
    NeurIPS 2019.  arXiv:1910.07467.
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
    """Root Mean Square Layer Normalization — channel-first layout (arXiv:1910.07467).

    Normalises a tensor along **``dim=1``** (the channel axis) and scales the
    result by a learned per-channel weight.  Accepts tensors of shape
    ``[B, C, *spatial]``, e.g. ``[B, C, H, W]`` for 2-D or ``[B, C, L]`` for 1-D.

    .. code-block:: text

        RMS(x)_b   = sqrt( mean_C(x[b, :, ...]²) + ε )   # scalar per sample
        out[b,:,…] = x[b,:,…] / RMS(x)_b  *  γ            # γ: [C, 1, …, 1]

    **Duck-typing sentinel**: the class attribute ``channels_first = True``
    allows callers (e.g. ``HyenaOperatorND``) to detect the layout without
    an ``isinstance`` check.

    Attributes:
        weight (nn.Parameter): Learnable scale ``γ`` of shape ``(C,)``,
            ones-initialised.  Tagged ``_no_weight_decay = True``.
        eps (float): Stability constant added inside the square root.
        use_quack (bool): Whether to attempt the QuACK kernel path.

    Args:
        dim: Number of channels ``C`` (size of ``dim=1``).
        eps: Small positive constant for numerical stability.  Default ``1e-6``.
        use_quack: If ``True`` (default), use the QuACK fused kernel when
            available.  Set to ``False`` to force the PyTorch path, which
            lets ``torch.compile`` fuse the norm with surrounding ops.
    """

    channels_first: bool = True

    def __init__(self, dim: int, eps: float = 1e-6, use_quack: bool = True):
        """Initialise RMSNormChannelFirst with ones-initialised weight.

        Args:
            dim: Number of channels ``C``; determines the shape of ``weight``.
            eps: Stability constant.  Default ``1e-6``.
            use_quack: Enable QuACK kernel path.  Default ``True``.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight._no_weight_decay = True
        self.eps = eps
        self.use_quack = use_quack

    def flop_count(self, num_tokens: int) -> int:
        """Return approximate FLOP count, identical in cost to channel-last RMSNorm.

        The three operations — square, mean+rsqrt, scale — each touch every
        element once, giving ``3 * num_tokens * C`` FLOPs regardless of the
        memory layout.

        Args:
            num_tokens: Total number of spatial positions in the batch
                (``B * prod(spatial_shape)``).

        Returns:
            Integer FLOP estimate: ``3 * num_tokens * C``.
        """
        dim = self.weight.shape[0]
        return 3 * num_tokens * dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise over the channel dimension (``dim=1``) and scale by weight.

        Args:
            x: Input tensor of shape ``[B, C, *spatial]``.

        Returns:
            torch.Tensor: Normalised and scaled tensor of the same shape and
            dtype as ``x``.
        """
        if self.use_quack and _quack_cf_available and x.is_cuda and _cuda_supports_quack(x.device):
            return _quack_rmsnorm_cf(x, self.weight, eps=self.eps)
        return _rmsnorm_channel_first_pytorch(x, self.weight, self.eps)
