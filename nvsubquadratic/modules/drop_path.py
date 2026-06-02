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

"""DropPath (Stochastic Depth): randomly drop entire residual branches during training.

Stochastic Depth (Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016)
is a regularisation technique that randomly bypasses entire residual branches with
per-sample probability ``p`` during training:

.. code-block:: text

    # training
    keep_prob = 1 - drop_prob
    mask      = Bernoulli(keep_prob) ∈ {0, 1}^B     # one scalar per sample
    out       = x * mask / keep_prob                 # inverted dropout scaling

    # inference
    out = x                                          # pure identity

The inverted-dropout scaling (dividing by ``keep_prob``) keeps the expected
output magnitude equal to the input magnitude, so no rescaling is needed at
inference time.

Unlike standard Dropout — which drops individual *elements* — DropPath drops the
*entire residual contribution* of a sample, which gives a stronger regularisation
signal in deep residual networks.  The effective network depth seen by each sample
is therefore drawn from a uniform distribution over ``[1, L]`` during training,
where ``L`` is the total depth.

**Integration pattern** (``vit5_residual_block.py``)::

    x = x + drop_path(ls_attn(mixer(norm(x))), drop_prob, self.training)
    x = x + drop_path(ls_mlp(mlp(mlp_norm(x))), drop_prob, self.training)

Reference:
    Huang, G., et al., "Deep Networks with Stochastic Depth",
    ECCV 2016.  arXiv:1603.09382.
"""

import torch
import torch.nn as nn


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Apply per-sample stochastic depth (functional form).

    During training each sample in the batch is independently kept or dropped
    with probability ``1 - drop_prob`` / ``drop_prob`` respectively.  The
    kept samples are rescaled by ``1 / (1 - drop_prob)`` to preserve the
    expected magnitude.  At inference time the function is an identity.

    Args:
        x: Input tensor of shape ``[B, *]`` — any layout; the drop mask has
           shape ``(B, 1, …, 1)`` and broadcasts over all non-batch dimensions.
        drop_prob: Probability of dropping a sample's contribution.
            ``0.0`` disables dropping; ``1.0`` zeros every sample (safe —
            the implementation guards against dividing by ``keep_prob`` when
            it is zero, so no inf/NaN is produced).
        training: Whether the model is currently in training mode.
            Set to ``False`` (or call ``model.eval()``) to disable dropping.

    Returns:
        torch.Tensor: Same shape and dtype as ``x``.  During training,
        approximately ``drop_prob * B`` samples are zeroed and the rest are
        rescaled.  During inference, returns ``x`` unchanged.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (stochastic depth) per sample — ``nn.Module`` wrapper.

    Thin stateful wrapper around the functional :func:`drop_path` that stores
    the drop probability and reads ``self.training`` automatically, making it
    a plug-in replacement wherever an ``nn.Module`` is required.

    **Effect on training vs. inference**

    - **Training** (``model.train()``): each sample's residual branch output
      is dropped with probability ``drop_prob`` and kept samples are rescaled
      by ``1 / (1 - drop_prob)``.
    - **Inference** (``model.eval()``): the module is a pure identity; no
      Bernoulli sampling or scaling is performed.

    Attributes:
        drop_prob (float): Probability of dropping a sample's residual output.
            Typically set between ``0.0`` (no drop) and ``0.3`` for deep ViTs.

    Args:
        drop_prob: Drop probability.  Defaults to ``0.0`` (disabled).
    """

    def __init__(self, drop_prob: float = 0.0):
        """Initialise DropPath.

        Args:
            drop_prob: Probability of dropping each sample's residual update.
                ``0.0`` disables the module (pure identity).  Default ``0.0``.
        """
        super().__init__()
        self.drop_prob = drop_prob

    def flop_count(self) -> int:
        """Return FLOP count — always zero.

        DropPath is a stochastic identity (training) or pure identity
        (inference).  The Bernoulli sampling and scalar division are
        negligible and not counted as floating-point arithmetic.

        Returns:
            Always ``0``.
        """
        return 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth to the input tensor.

        Args:
            x: Input tensor of shape ``[B, *]``.

        Returns:
            torch.Tensor: Same shape and dtype as ``x``, with per-sample
            dropping applied during training.
        """
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        """Return drop probability for ``repr()``."""
        return f"drop_prob={self.drop_prob:.3f}"
