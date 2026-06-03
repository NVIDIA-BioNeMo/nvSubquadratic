# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RMSNorm — Root Mean Square Layer Normalization.

RMSNorm (Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019) is a
simplified variant of LayerNorm that omits the mean-centering step.  For a vector
``x ∈ ℝ^D`` it computes:

.. code-block:: text

    RMS(x)  = sqrt( (1/D) Σ_d x_d²  +  ε )
    x̂       = x / RMS(x)
    out     = γ ⊙ x̂

where ``γ ∈ ℝ^D`` is a learned per-element scale and ``ε`` is a small stability
constant.  Dropping the mean-centering from LayerNorm cuts the compute roughly in
half for large ``D`` while typically achieving comparable training stability.

**Backends**

Two backend implementations are selected at runtime:

- **QuACK** (default, ``use_quack=True``): fused CUDA kernel from the ``quack``
  package.  Active only on Hopper / Blackwell GPUs (SM ≥ 9.0 — H100, B200, B300).
  On older GPUs (e.g. Ampere A100) QuACK's backward kernel is incompatible; the
  module detects this and falls back to PyTorch automatically.  The QuACK kernel is
  opaque to ``torch.compile`` and acts as a fusion barrier.

- **PyTorch** (``use_quack=False`` or fallback): pure-Python ``torch`` ops
  (``pow``, ``mean``, ``rsqrt``), upcasted to float32 for numerical safety.  Fully
  visible to the compiler, enabling fusion with adjacent ops.

Set ``use_quack=False`` explicitly when running under ``torch.compile`` so the compiler
can fuse the norm with surrounding operations.

Reference:
    Zhang, B. & Sennrich, R., "Root Mean Square Layer Normalization",
    NeurIPS 2019.  arXiv:1910.07467.
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
    """Root Mean Square Layer Normalization (Zhang & Sennrich, arXiv:1910.07467).

    Normalises a tensor over its last dimension using RMS statistics, then
    scales the result by a learned per-channel weight:

    .. code-block:: text

        RMS(x) = sqrt( mean(x²) + ε )      # scalar per token
        out    = (x / RMS(x)) * γ           # γ broadcast over leading dims

    Accepts tensors of any shape ``[*leading, D]``; only the last dimension is
    normalised.  The learned weight ``γ`` has shape ``(D,)`` and is **excluded
    from weight decay** via the ``_no_weight_decay`` tag.

    **Backend selection** (see module docstring for full details):

    - ``use_quack=True`` (default): QuACK fused kernel on SM ≥ 9.0 GPUs.
      Falls back to PyTorch automatically on older GPUs.
    - ``use_quack=False``: Pure PyTorch; preferred under ``torch.compile``.

    Attributes:
        weight (nn.Parameter): Learnable scale ``γ`` of shape ``(dim,)``,
            ones-initialised.  Tagged ``_no_weight_decay = True``.
        eps (float): Stability constant added inside the square root.
        use_quack (bool): Whether to attempt the QuACK kernel path.

    Args:
        dim: Size of the last dimension ``D`` to normalise over.
        eps: Small positive constant for numerical stability.  Default ``1e-6``.
        use_quack: If ``True`` (default), use the QuACK fused kernel when
            available.  Set to ``False`` to force the PyTorch path, which
            lets ``torch.compile`` fuse the norm with surrounding ops.
    """

    def __init__(self, dim: int, eps: float = 1e-6, use_quack: bool = True):
        """Initialise RMSNorm with ones-initialised weight.

        Args:
            dim: Channel dimension ``D``; determines the shape of ``weight``.
            eps: Stability constant added to the RMS denominator.  Default ``1e-6``.
            use_quack: Enable QuACK kernel path.  Default ``True``.
        """
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
        """Apply RMS normalisation over the last dimension.

        Args:
            x: Input tensor of shape ``[*leading, D]``.  Any number of
               leading dimensions is supported; only the last axis is
               normalised.

        Returns:
            torch.Tensor: Normalised and scaled tensor, same shape and dtype
            as ``x``.
        """
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
    """RMSNorm applied independently to each attention head (QK-norm).

    Accepts a flat hidden representation of shape ``[*leading, H·D]``,
    reshapes to ``[*leading, H, D]``, applies an independent :class:`RMSNorm`
    to each head's ``D``-dimensional slice, then flattens back to
    ``[*leading, H·D]``.  Each head has its own learnable scale ``γ ∈ ℝ^D``.

    This is the *QK-norm* technique used in ViT-5 / vit5_attention.py to
    stabilise attention logit magnitudes at large model sizes:

    .. code-block:: text

        x  : [*leading, H*D]
        x  → reshape → [*leading, H, D]
        x  → RMSNorm per head → [*leading, H, D]
        x  → flatten → [*leading, H*D]

    Attributes:
        num_heads (int): Number of attention heads ``H``.
        head_dim (int): Dimension per head ``D``.
        norm (RMSNorm): Shared :class:`RMSNorm` instance applied to each head
            slice (the weight ``γ`` has shape ``(D,)``).

    Args:
        num_heads: Number of attention heads ``H``.
        head_dim: Dimension of each head ``D``.
        eps: Small constant for numerical stability.  Default ``1e-6``.
        use_quack: If ``True`` (default), use the QuACK fused kernel when
            available.  Set to ``False`` to force the PyTorch path so that
            ``torch.compile`` can fuse the norm with surrounding ops.
    """

    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6, use_quack: bool = True):
        """Initialise PerHeadRMSNorm.

        Args:
            num_heads: Number of attention heads ``H``.
            head_dim: Dimension per head ``D``; the inner :class:`RMSNorm`
                normalises over this dimension.
            eps: Stability constant.  Default ``1e-6``.
            use_quack: Enable QuACK kernel path.  Default ``True``.
        """
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
        """Apply per-head RMS normalisation.

        Args:
            x: Input tensor of shape ``[*leading, num_heads * head_dim]``.

        Returns:
            torch.Tensor: Normalised tensor of the same shape as ``x``, where
            each head's ``head_dim``-slice has been RMS-normalised and scaled
            by the shared learnable ``γ``.
        """
        shape = x.shape
        x = x.view(*shape[:-1], self.num_heads, self.head_dim)
        x = self.norm(x)
        return x.view(shape)

    def extra_repr(self) -> str:
        """Return head layout for repr()."""
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}"
