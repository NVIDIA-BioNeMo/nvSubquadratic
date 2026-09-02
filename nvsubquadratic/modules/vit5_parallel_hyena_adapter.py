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

"""Zero-init parallel bottleneck adapter for ViT-5 attention blocks.

Implements the compound sequence mixer used in the zero-init parallel adapter
ablation (C0/C1/C2 arms in the experiment brief).  The mixer combines a frozen
attention branch with a bottleneck branch that is **bit-exactly inert at
initialisation**:

.. code-block:: text

    output = Attn(x) + W_up( inner_mixer( W_down(x) ) )

where ``W_up`` is initialised to all zeros, making the adapter a no-op at
step 0.  The ``inner_mixer`` can be any ``[B, T, rank] → [B, T, rank]`` module
(including ``nn.Identity`` for the C0 capacity ablation).

Arms:
    - **C0**: ``inner_mixer = nn.Identity()`` — no token mixing, tests capacity.
    - **C1**: ``inner_mixer = ViT5DepthwiseConv2dMixer`` — local 2-D prior.
    - **C2**: ``inner_mixer = ViT5HyenaAdapter`` — global implicit filter.

**Why zero the weight, not a scalar gate?**
A zero gate in front of a branch that produces NaN propagates NaN (``0.0 *
nan == nan``); a zero *weight matrix* always maps any finite input to exactly
``0.0``.  See §4.1 of the brief.

**Interface contract**
The module satisfies the same ``[B, T, C] → [B, T, C]`` interface as
:class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention`, so it can be
dropped into :class:`~nvsubquadratic.modules.vit5_residual_block.ViT5ResidualBlock`
as the ``sequence_mixer_cfg`` target without any block modifications.

The bottleneck branch operates at width ``rank`` (typically 16–64).
``W_down: hidden_dim → rank`` and ``W_up: rank → hidden_dim`` are both
``bias=False`` linear projections.
"""

from typing import Optional

import torch
import torch.nn as nn

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class ViT5ParallelHyenaSequenceMixer(nn.Module):
    r"""Combines frozen ViT5Attention with a zero-init bottleneck parallel branch.

    Drop-in replacement for
    :class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention` as the
    ``sequence_mixer`` inside
    :class:`~nvsubquadratic.modules.vit5_residual_block.ViT5ResidualBlock`.

    The inner mixer is pluggable — pass any ``[B, T, rank] → [B, T, rank]``
    module to select C0 (identity), C1 (local conv), or C2 (Hyena):

    - **C0**: ``inner_mixer_cfg=None`` → ``nn.Identity()`` — capacity ablation.
    - **C1**: ``inner_mixer_cfg=LazyConfig(ViT5DepthwiseConv2dMixer)(...)`` —
      local 2-D convolutional prior.
    - **C2**: ``inner_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(...)`` — global
      implicit filter (pass the ``ViT5HyenaAdapter`` LazyConfig directly, NOT
      the inner ``QKVSequenceMixer`` — ``ViT5HyenaAdapter`` already handles the
      ``[B, T, rank]`` reshape contract).

    Data flow::

        x: [B, T, C]  (pre-normed by ViT5ResidualBlock)
            │
            ├──► attn(x) ──────────────────────────────────────────────►┐
            │                                                             │
            └──► w_down(x): [B, T, rank]                                 │
                    │                                                     │
                    ▼                                                     │
                inner_mixer(·): [B, T, rank]                             │
                    │                                                     │
                    ▼                                                     │
                w_up(·): [B, T, C]  ← zero-init weights                 │
                    │                                                     │
                    └──────────────────────────────────────────────────►(+)
                                                                          │
                                                                          ▼
                                                                   [B, T, C]

    ``W_up`` weights are zeroed in ``__init__``, so the entire adapter output
    is exactly ``0`` at step 0 and the block is bit-identical to a model that
    has no adapter.

    Attributes:
        attn (nn.Module): Instantiated attention module.
        w_down (nn.Linear): Down-projection ``hidden_dim → rank``, no bias.
        inner_mixer (nn.Module): Pluggable token mixer at bottleneck width.
            ``nn.Identity`` for C0, a depthwise conv for C1, ``ViT5HyenaAdapter``
            for C2.
        w_up (nn.Linear): Up-projection ``rank → hidden_dim``, no bias,
            **zero-initialised**.
        rank (int): Bottleneck width.
        hidden_dim (int): Transformer hidden dimension ``C``.
    """

    def __init__(
        self,
        attn_cfg: LazyConfig,
        hidden_dim: int,
        rank: int,
        inner_mixer_cfg: Optional[LazyConfig] = None,
        conditioning_source: str = "none",
    ) -> None:
        r"""Instantiate attention, down/up projections, and the inner mixer.

        Args:
            attn_cfg: LazyConfig for the attention module.  Must accept
                ``x: Tensor[B, T, C]`` and return the same shape.
            hidden_dim: Transformer hidden dimension ``C``.
            rank: Bottleneck width ``r``.  ``W_down: C → r``, ``W_up: r → C``.
            inner_mixer_cfg: Optional LazyConfig for a ``[B, T, rank] → [B, T, rank]``
                token mixer.  When ``None``, ``nn.Identity`` is used (C0 ablation).
                For C1, pass a :class:`ViT5DepthwiseConv2dMixer` config.
                For C2, pass a :class:`~nvsubquadratic.modules.vit5_hyena_adapter.ViT5HyenaAdapter`
                config (with ``inner_mixer_cfg`` and ``grid_w`` set inside it).
            conditioning_source: Source of the FiLM control variable ``z(x)``.

                * ``"none"`` (default) — unconditional kernel.  For a Hyena inner
                  mixer this yields a **static** implicit long convolution, which
                  is *not* HyenaND: Wessels et al. (arXiv:2607.19378) define the
                  operator as ``K_i(x) = w(c_i) ⊙ f_θ(c_i; z(x))``, i.e. input
                  dependence is constitutive.  Use only as an explicit
                  static-kernel ablation arm.
                * ``"token_mean"`` — mean-pool the input tokens into
                  ``(B, hidden_dim)`` and forward as ``conditioning=``.  Requires
                  an inner mixer accepting that kwarg (the ``ViT5HyenaAdapter`` →
                  ``QKVSequenceMixer`` → ``Hyena`` chain) whose SIREN kernel was
                  built with a ``film_cfg`` sized for ``cond_dim=hidden_dim``.
                  ``nn.Identity`` and depthwise conv will raise; pair them with
                  ``"none"``.

                ``z(x)`` is pooled once per instance, so the inner mixer still
                issues a single ND FFT call.  FiLM initialises to ``(γ, β) = (1, 0)``
                — unmodulated at init — so the zero-init ``W_up`` bit-exactness
                guarantee holds either way.

        Raises:
            ValueError: If ``conditioning_source`` is not a recognised value.
        """
        super().__init__()
        if conditioning_source not in ("none", "token_mean"):
            raise ValueError(f"conditioning_source must be 'none' or 'token_mean', got {conditioning_source!r}")

        self.hidden_dim = hidden_dim
        self.rank = rank
        self.conditioning_source = conditioning_source

        self.attn = instantiate(attn_cfg)
        self.w_down = nn.Linear(hidden_dim, rank, bias=False)
        self.inner_mixer: nn.Module = instantiate(inner_mixer_cfg) if inner_mixer_cfg is not None else nn.Identity()
        self.w_up = nn.Linear(rank, hidden_dim, bias=False)

        # Zero-init W_up so the adapter is a bit-exact no-op at step 0.
        nn.init.zeros_(self.w_up.weight)

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        r"""Apply attention plus the zero-init Hyena adapter.

        At step 0, ``W_up.weight == 0`` so the adapter contributes exactly
        ``0`` to any finite input.  ``_kwargs`` (e.g. ``conditioning`` from
        register pooling) are accepted for interface compatibility but silently
        ignored — :class:`~nvsubquadratic.modules.vit5_attention.ViT5Attention`
        does not accept conditioning, and the Hyena branch is unconditional.

        Args:
            x: Pre-normed token sequence ``[B, T, C]`` (the norm is applied by
                the enclosing
                :class:`~nvsubquadratic.modules.vit5_residual_block.ViT5ResidualBlock`
                before this call).  ``T`` must satisfy ``T % grid_w == 0``.
            **_kwargs: Accepted for interface compatibility; not forwarded.

        Returns:
            Tensor of shape ``[B, T, C]``:
            ``attn(x) + w_up(hyena_adapter(w_down(x)))``.
        """
        attn_out = self.attn(x)
        h = self.w_down(x)
        if self.conditioning_source == "token_mean":
            h = self.inner_mixer(h, conditioning=x.mean(dim=1))
        else:
            h = self.inner_mixer(h)
        h = self.w_up(h)
        return attn_out + h

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        """Count FLOPs for one forward pass (one sample).

        Sums over:
        1. Attention: ``self.attn.flop_count(num_tokens, inference)``
        2. Down-projection: ``2 * T * C * rank`` MACs
        3. Inner mixer: ``self.inner_mixer.flop_count(num_tokens, inference)`` if available.
        4. Up-projection: ``2 * T * rank * C`` MACs

        Args:
            num_tokens: Sequence length ``T``.
            inference: Forwarded to sub-modules.

        Returns:
            Total FLOPs as an integer.
        """
        T, C, r = num_tokens, self.hidden_dim, self.rank
        flops = 0
        if hasattr(self.attn, "flop_count"):
            flops += self.attn.flop_count(num_tokens, inference=inference)
        flops += 2 * T * C * r  # w_down
        if hasattr(self.inner_mixer, "flop_count"):
            flops += self.inner_mixer.flop_count(num_tokens, inference=inference)
        flops += 2 * T * r * C  # w_up
        return flops

    def extra_repr(self) -> str:
        """Return width, rank, and conditioning source for PyTorch's module repr."""
        return f"hidden_dim={self.hidden_dim}, rank={self.rank}, conditioning_source={self.conditioning_source}"
