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

"""Minimal LoRA (Low-Rank Adaptation) linear layer.

Implements the standard LoRA delta ``W_orig + B @ A`` (Hu et al., 2022) as a
drop-in replacement for :class:`torch.nn.Linear`.  Used only in arm B of the
parallel Hyena adapter experiment as a parameter-matched baseline for arm C.

The frozen pretrained weight ``W_orig`` is stored as a non-parameter buffer so
that optimisers never see it.  Only ``lora_A`` (``in → rank``) and ``lora_B``
(``rank → out``) are trainable.  ``lora_B`` is zero-initialised so the module
is a bit-exact no-op at step 0 (matching the zero-init property of arm C).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    r"""Linear layer with a frozen weight plus a trainable low-rank delta.

    Computes ``y = x @ (W + B @ A)^T + bias`` where:

    * ``W`` — frozen copy of the pretrained weight (``[out, in]`` buffer).
    * ``A`` — trainable ``[rank, in]`` factor, Kaiming-uniform init.
    * ``B`` — trainable ``[out, rank]`` factor, **zero-init** so the delta is
      zero at step 0.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: LoRA rank ``r``.
        bias: If ``True``, a learnable bias is added (default ``False``).
            The bias is always trainable regardless of the frozen weight.

    Attributes:
        frozen_weight (Tensor): Non-parameter buffer holding ``W``.
        lora_A (nn.Parameter): ``[rank, in_features]``, Kaiming-uniform.
        lora_B (nn.Parameter): ``[out_features, rank]``, zero-initialised.
        bias (nn.Parameter | None): Optional ``[out_features]`` bias.
        rank (int): LoRA rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = False,
    ) -> None:
        """Allocate frozen weight buffer, LoRA factors, and optional bias."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.register_buffer("frozen_weight", torch.zeros(out_features, in_features))

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def load_pretrained_weight(self, weight: torch.Tensor) -> None:
        r"""Copy ``weight`` into the frozen buffer (call after loading checkpoint).

        Args:
            weight: Tensor of shape ``[out_features, in_features]``.
        """
        with torch.no_grad():
            self.frozen_weight.copy_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute ``x @ (frozen_weight + lora_B @ lora_A)^T + bias``.

        Args:
            x: Input tensor ``[..., in_features]``.

        Returns:
            Output tensor ``[..., out_features]``.
        """
        effective_weight = self.frozen_weight + self.lora_B @ self.lora_A
        return F.linear(x, effective_weight, self.bias)

    def extra_repr(self) -> str:
        """Return module summary string."""
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"
