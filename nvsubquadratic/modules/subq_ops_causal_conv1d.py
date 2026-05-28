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

"""Causal depthwise 1D convolution backed by the subq_ops CUDA kernel.

Drop-in for ``torch.nn.Conv1d`` in Hyena's ``short_conv`` slot when the host
is 1D autoregressive.  Subclasses :class:`torch.nn.Conv1d` so Hyena's
``isinstance(..., torch.nn.Conv1d)`` check passes and parameter layout
(``weight``/``bias``) matches the surrounding state-dict conventions.

Constraints (asserted at construction):

- ``groups == in_channels == out_channels`` (depthwise only — matches the
  upstream kernel's ``[C, K]`` weight shape).
- ``stride == 1`` and ``dilation == 1`` (the kernel does not support either).
- ``padding`` is ignored; the kernel applies causal left-only padding internally.
"""

from __future__ import annotations

import torch


class SubqOpsCausalConv1d(torch.nn.Conv1d):
    """Depthwise causal 1D conv using ``subquadratic_ops_torch.causal_conv1d``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        groups: int,
        bias: bool = False,
        activation: str = "identity",
    ) -> None:
        """Build a depthwise causal conv with optional SiLU activation.

        Args:
            in_channels: Channel count (must equal ``out_channels`` and ``groups``).
            out_channels: Output channels (depthwise: same as ``in_channels``).
            kernel_size: Causal kernel length.
            groups: Must equal ``in_channels`` for depthwise layout.
            bias: Whether to include a per-channel bias.
            activation: ``"identity"`` or ``"silu"`` applied inside the CUDA kernel.
        """
        if not (groups == in_channels == out_channels):
            raise ValueError(
                "SubqOpsCausalConv1d is depthwise-only: groups must equal "
                f"in_channels must equal out_channels.  Got groups={groups}, "
                f"in_channels={in_channels}, out_channels={out_channels}."
            )
        if activation not in ("identity", "silu"):
            raise ValueError(f"activation must be 'identity' or 'silu'. Got {activation!r}.")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            bias=bias,
            padding=0,
        )
        self.activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Run causal depthwise conv; input shape ``[B, C, L]``."""
        from nvsubquadratic.ops.causal_conv1d_custom import causal_conv1d

        # nn.Conv1d depthwise weight is [C, 1, K]; upstream kernel expects [C, K].
        weight_2d = self.weight.squeeze(1)
        return causal_conv1d(input, weight_2d, self.bias, self.activation)
