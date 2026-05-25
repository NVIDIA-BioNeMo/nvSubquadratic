# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Adapted from https://github.com/implicit-long-convs/ccnn_v2

"""QKV cross-attention condition mixer for injecting a conditioning signal into a feature map.

**Role in the residual block**

The conditioning mixer is the *middle* sub-branch of
:class:`~nvsubquadratic.modules.residual_block.ResidualBlock`.  After the
sequence mixer has let spatial positions exchange information, the condition
mixer injects an external conditioning signal ``c`` — such as a diffusion
timestep embedding, a class label embedding, or a physics parameter vector —
into the residual stream ``x``:

.. code-block:: text

    x ──[seq_mixer]──► x'
    x'──[cond_mixer(x', c)]──► x''    ← this module
    x''─[mlp]──────────────► output

**Comparison with other conditioning strategies**

* **FiLM / AdaLN-Zero** — applies a *per-channel* affine transform
  ``y = γ(c) ⊙ x + β(c)`` to the feature map (see
  :mod:`nvsubquadratic.modules.film`).  This is fast and parameter-efficient
  but does not allow the conditioning signal to attend selectively to specific
  spatial positions.

* **Cross-attention** — routes the conditioning signal through standard
  Q/K/V attention so that every position in ``x`` can attend to all
  conditioning tokens.  Highly expressive but O(L_x · L_c) in cost.

* **QKVConditionMixer** *(this module)* — implements a lightweight cross-
  attention variant where queries come from the feature map ``x`` and keys/
  values come from the conditioning signal ``c``.  The concrete attention
  computation is delegated to a configurable inner ``mixer`` module (supplied
  via ``mixer_cfg``), giving the same operator-agnostic dispatch pattern as
  :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`.  The
  result is more expressive than FiLM (it can attend to individual conditioning
  tokens) while keeping the projections and mixing strategy swappable.

**Conditioning signal shapes**

The condition tensor ``c`` may be:

* ``(B, C)`` — a global (non-spatial) conditioning vector, e.g. a pre-pooled
  timestep or class embedding.  Internally unsqueezed to ``(B, 1, C)`` before
  the K/V projection so that the inner mixer sees a single conditioning token.
* ``(B, *spatial_dims_cond, C)`` — spatially distributed conditioning tokens
  (e.g. encoder output in an encoder-decoder architecture).  Must have the
  same number of dimensions as the feature map ``x``.

In both cases the channel dimension ``C`` must equal ``hidden_dim``.

All tensors use **channels-last** layout: ``(B, *spatial, C)``.
"""

from typing import Callable

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class QKVConditionMixer(torch.nn.Module):
    """Cross-attention condition mixer that routes a conditioning signal into the feature map.

    This module implements the *condition mixer branch* of
    :class:`~nvsubquadratic.modules.residual_block.ResidualBlock`.  It injects
    an external conditioning signal ``c`` (e.g. a timestep embedding, class
    label, or physics parameter vector) into the residual stream ``x`` via
    learned Q, K, V projections and a pluggable inner mixing operator:

    .. code-block:: text

        x  ─[q_proj]──────────────────────────► Q ─┐
        c  ─[kv_proj]──► split ──► K, V ────────────► inner_mixer(Q, K, V) ─[out_proj]──► y

    Queries are derived from the current feature map ``x`` so that each spatial
    position can attend selectively to the conditioning tokens.  Keys and values
    are derived from the conditioning signal ``c``.  This is therefore a form of
    **cross-attention** conditioning — more expressive than FiLM
    (which applies a uniform per-channel affine transform regardless of spatial
    content) and more efficient than full self-attention between concatenated
    feature and conditioning tokens.

    The inner mixing computation is delegated to ``mixer_cfg``, following the
    same operator-agnostic dispatch pattern as
    :class:`~nvsubquadratic.modules.sequence_mixer.QKVSequenceMixer`.  Any
    module whose ``forward(q, k, v)`` conforms to channels-last tensors of
    shape ``(B, *spatial, C)`` can be used as the inner mixer.

    **Weight initialisation**

    Optional curried initialisers ``init_method_in`` and ``init_method_out``
    allow the caller to supply per-projection weight schedules (e.g.
    depth-scaled Gaussian init from GPT / Megatron).  Both follow the
    signature ``fn(dim: int) -> fn(tensor: Tensor) -> None``.  When omitted,
    PyTorch's default Kaiming-uniform init is used.

    **Weight decay**

    No weight-decay tags are set on the projections; the caller is responsible
    for any per-parameter weight-decay grouping (see the analogous logic in
    :class:`~nvsubquadratic.modules.film.KernelFiLMGenerator`).

    Attributes:
        mixer (torch.nn.Module): The instantiated inner mixing operator.  Its
            ``forward(q, k, v)`` method receives channels-last tensors of shape
            ``(B, *spatial, C)`` and must return a tensor of the same shape as
            ``q``.
        kv_proj (torch.nn.Linear): Combined K+V projection (no bias) that maps
            the conditioning signal from ``C`` to ``2·C``.  Weight shape:
            ``(2·hidden_dim, hidden_dim)``.
        q_proj (torch.nn.Linear): Query projection (no bias) that maps the
            feature map from ``C`` to ``C``.  Weight shape:
            ``(hidden_dim, hidden_dim)``.
        out_proj (torch.nn.Linear): Output projection (no bias) that maps the
            mixer output back to ``C``.  Weight shape:
            ``(hidden_dim, hidden_dim)``.
    """

    def __init__(
        self,
        hidden_dim: int,
        mixer_cfg: LazyConfig,
        init_method_in: Callable[[torch.Tensor], torch.Tensor] | None = None,
        init_method_out: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initialise the QKVConditionMixer.

        Args:
            hidden_dim: Channel dimension ``C`` shared by the feature map ``x``
                and the conditioning signal ``c``.  All four linear projections
                (``q_proj``, ``kv_proj``, ``out_proj``) are sized using this
                value.
            mixer_cfg: :class:`~nvsubquadratic.lazy_config.LazyConfig` for the
                inner mixing operator.  The instantiated module's ``forward``
                must accept ``(q, k, v)`` as positional arguments and return a
                tensor of the same shape as ``q``.  Any attention-compatible
                module (e.g. a dot-product attention layer) can be used here.
            init_method_in: Optional *curried* weight initialiser applied to
                both ``q_proj.weight`` and ``kv_proj.weight``.  Must have the
                signature ``fn(dim: int) -> fn(tensor: Tensor) -> None``.
                When provided, ``fn(hidden_dim)`` is called and the returned
                callable is applied in-place to each weight matrix.  Pass
                ``None`` to keep PyTorch's default Kaiming-uniform init.
            init_method_out: Same as ``init_method_in`` but applied to
                ``out_proj.weight``.  A common choice is a depth-scaled
                Gaussian (GPT / Megatron style) to control the residual branch
                variance at initialisation.  Pass ``None`` to keep the default.
        """
        super().__init__()

        # Instantiate condition mixer layer (expects a module taking q, k, v)
        self.mixer = instantiate(mixer_cfg)
        # Combined KV projection (no bias)
        self.kv_proj = torch.nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        # Q projection (no bias)
        self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Output projection
        self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Initialize projections
        if init_method_in is not None:
            init_method_in(hidden_dim)(self.kv_proj.weight.data)
            init_method_in(hidden_dim)(self.q_proj.weight.data)
        if init_method_out is not None:
            init_method_out(hidden_dim)(self.out_proj.weight.data)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Inject the conditioning signal into the feature map via cross-attention.

        Computes queries from the current feature map ``x`` and keys/values
        from the conditioning signal ``condition``, then mixes them with the
        inner ``mixer`` and projects the result back to ``hidden_dim``.

        The signal flow is:

        .. code-block:: text

            Q = q_proj(x)                     # (B, *spatial_dims, C)
            K, V = split(kv_proj(condition))  # each (B, *spatial_dims_cond, C)
            y = out_proj(mixer(Q, K, V))      # (B, *spatial_dims, C)

        A global (non-spatial) conditioning vector of shape ``(B, C)`` is
        automatically unsqueezed to ``(B, 1, C)`` before the K/V projection
        so that the inner mixer sees a single conditioning token per sample.

        Args:
            x: Feature map tensor of shape ``(B, *spatial_dims, C)``, where
                ``B`` is the batch size, ``spatial_dims`` is one or more
                spatial axes (e.g. ``(H, W)`` for 2-D images or ``(T,)`` for
                1-D sequences), and ``C = hidden_dim``.  Must have at least
                three dimensions (batch + one spatial axis + channel).
            condition: Conditioning signal tensor.  Two shapes are accepted:

                * ``(B, C)`` — global conditioning vector (e.g. a timestep or
                  class embedding).  Unsqueezed internally to ``(B, 1, C)``
                  before projection.
                * ``(B, *spatial_dims_cond, C)`` — spatially distributed
                  conditioning tokens (e.g. encoder output).  Must have the
                  same number of dimensions as ``x``.  The spatial extent
                  ``spatial_dims_cond`` need not match ``spatial_dims`` of
                  ``x``.

                The channel dimension ``C`` must equal ``hidden_dim`` in both
                cases.

        Returns:
            Output tensor of shape ``(B, *spatial_dims, C)`` — same spatial
            layout as ``x``, with the conditioning signal blended in via
            cross-attention.

        Raises:
            ValueError: If ``x`` has fewer than three dimensions (i.e. is
                missing at least one spatial axis).
            ValueError: If ``condition.ndim`` is neither ``2`` (global vector)
                nor equal to ``x.ndim`` (matching spatial rank).
        """
        if x.ndim < 3:
            raise ValueError(f"x must have at least one spatial dimension; got shape {x.shape}.")

        # Support global conditioning ([B, hidden_dim]) as well as spatial conditioning
        if condition.ndim == 2:
            # Unsqueeze the conditioning vector to create a single spatial dim.
            condition = condition.unsqueeze(1)
        elif condition.ndim != x.ndim:
            raise ValueError(
                f"Condition must have either 2 dimensions (global) or match x's spatial rank. "
                f"Got condition.ndim={condition.ndim}, expected {x.ndim}."
            )

        # Q projection from the current stream
        q = self.q_proj(x)
        # KV projection from the condition signal
        kv = self.kv_proj(condition)
        k, v = torch.chunk(kv, 2, dim=-1)
        # Condition mixer (e.g., cross-attention, etc.)
        x = self.mixer(q, k, v)
        # Output projection
        x = self.out_proj(x)
        return x
