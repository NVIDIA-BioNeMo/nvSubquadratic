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

"""QKV condition mixer for conditioning."""

from typing import Callable

import torch

from nvsubquadratic.lazy_config import LazyConfig, instantiate


class QKVConditionMixer(torch.nn.Module):
    """QKV condition mixer."""

    def __init__(
        self,
        hidden_dim: int,
        mixer_cfg: LazyConfig,
        init_method_in: Callable[[torch.Tensor], torch.Tensor] | None = None,
        init_method_out: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """Initialize the QKV condition mixer.

        Args:
            hidden_dim: Hidden dimension.
            mixer_cfg: LazyConfig for the condition mixer layer.
            init_method_in: Optional initialization method for the KV and Q projections.
            init_method_out: Optional initialization method for the output projection.
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
        """Forward pass of the QKV condition mixer.

        Args:
            x: Input tensor of shape [B, * spatial_dims, hidden_dim].
            condition: Condition tensor of shape [B, * spatial_dims_condition, hidden_dim] or [B, hidden_dim].

        Returns:
            Output tensor of shape [B, * spatial_dims, hidden_dim].
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
