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

# TODO: Add license header here


"""Query-Key (QK) normalisation utilities for attention stabilisation.

QK-norm (Henry et al., "Query-Key Normalization for Transformers", 2020;
also used in ViT-5 / vit5_attention.py) L2-normalises the query and key
vectors before computing attention logits:

.. code-block:: text

    q̂ = q / ||q||₂    k̂ = k / ||k||₂
    logits = (q̂ · k̂) / τ          # τ is a learned temperature

By bounding the dot-product to ``[-1, 1]`` (cosine similarity), QK-norm
prevents attention logit explosion in deep or wide models and removes the
need for ``1 / sqrt(d_head)`` scaling (though a learned temperature is
typically kept for flexibility).

This module provides:

- :func:`apply_qk_norm`: functional form, normalises a ``(query, key)`` pair.
- :class:`L2Norm`: ``nn.Module`` wrapper suitable as a :class:`LazyConfig`
  target; also satisfies the ``channels_first`` duck-type used by norms in
  this codebase.
"""

import torch
import torch.nn.functional as F


def apply_qk_norm(query: torch.Tensor, key: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    """L2-normalise query and key tensors along a given dimension.

    Computes ``F.normalize(q, p=2, dim=dim)`` and the equivalent for ``k``.
    The resulting vectors have unit L2 norm along ``dim``, so their dot
    product is bounded to ``[-1, 1]`` (cosine similarity).

    Args:
        query: Query tensor of shape ``[B, H, T, D]`` (or any layout where
            ``dim`` selects the head/feature axis to normalise over).
        key: Key tensor; must be broadcast-compatible with ``query``.
        dim: Axis to normalise over.  Default ``-1`` (last axis = feature
            dimension in ``[B, H, T, D]`` layout).
        eps: Small constant added to the L2 norm for numerical stability.
            Default ``1e-12``.

    Returns:
        Tuple ``(query_normed, key_normed)`` with the same shapes and dtypes
        as the inputs.
    """
    query = F.normalize(query, p=2.0, dim=dim, eps=eps)
    key = F.normalize(key, p=2.0, dim=dim, eps=eps)
    return query, key


class L2Norm(torch.nn.Module):
    """L2 normalisation layer — learnable-parameter-free, :class:`LazyConfig`-friendly.

    Wraps ``F.normalize(x, p=2, dim=self.dim)`` as an ``nn.Module`` so it can
    be used as a ``norm_cfg`` target in :func:`~nvsubquadratic.lazy_config.instantiate`
    wherever a plain normalisation module is expected (e.g. as the QK-norm in
    :class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention`).

    **Duck-typing**

    The :attr:`channels_first` property returns ``True`` when ``dim == 1``,
    matching the convention used by
    :class:`~nvsubquadratic.modules.rms_norm_channel_first.RMSNormChannelFirst`
    so callers can detect the memory layout without an ``isinstance`` check.

    Attributes:
        dim (int): Axis to normalise along.
        eps (float): Stability constant for the L2 norm denominator.

    Args:
        dim: Dimension to normalise over.  Default ``-1`` (last axis).
        eps: Small positive constant added to the L2 norm.  Default ``1e-12``.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-12):
        """Initialise L2Norm.

        Args:
            dim: Axis to normalise over.  Default ``-1``.
            eps: Stability constant.  Default ``1e-12``.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    @property
    def channels_first(self) -> bool:
        """``True`` when normalising over ``dim=1`` (channel-first layout)."""
        return self.dim == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalise ``x`` along ``self.dim``.

        Args:
            x: Input tensor of any shape.

        Returns:
            torch.Tensor: Unit-norm tensor along ``self.dim``, same shape
            and dtype as ``x``.
        """
        return F.normalize(x, p=2.0, dim=self.dim, eps=self.eps)

    def extra_repr(self) -> str:
        """Return dim and eps for repr()."""
        return f"dim={self.dim}, eps={self.eps}"
