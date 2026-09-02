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

"""Local 2-D depthwise convolutional mixer for ViT-5 bottleneck branches.

Satisfies the ``[B, T, C] → [B, T, C]`` token-sequence interface and is used
as the C1 inner mixer in
:class:`~nvsubquadratic.modules.vit5_parallel_hyena_adapter.ViT5ParallelHyenaSequenceMixer`.
It applies a depthwise (groups=channels) 2-D convolution with a 3×3 kernel
after reshaping the flat token sequence to a 2-D spatial grid and before
reshaping back.

Inductive bias: strictly local — each token's output depends only on its
3×3 neighbourhood.  Used as the ablation C1 between the capacity ablation C0
(no mixing at all) and the global implicit filter C2 (Hyena2D).

If C1 ≈ C2 on the far-displacement buckets, the global Hyena filter earned
nothing beyond what a local convolution provides.
"""

import torch
import torch.nn as nn


class ViT5DepthwiseConv2dMixer(nn.Module):
    r"""Depthwise 3×3 2-D convolution over a reshaped ``[B, T, C]`` sequence.

    Data flow::

        x: [B, T, C]
            │
            ▼  reshape  (T = H * grid_w)
        x: [B, H, grid_w, C]
            │
            ▼  permute → [B, C, H, grid_w]
        depthwise_conv2d (kernel 3×3, padding=1, groups=C)
            │
            ▼  permute → [B, H, grid_w, C]
            │
            ▼  reshape back
        x: [B, T, C]

    The convolution uses ``padding=1`` to preserve spatial dimensions, and
    ``groups=C`` to keep channels independent (depthwise).  No bias.

    Args:
        channels: Number of channels (equals ``rank`` in the bottleneck branch).
        grid_w: Spatial grid width.  ``T`` must satisfy ``T % grid_w == 0``.
        kernel_size: Convolution kernel size (default 3).

    Attributes:
        conv (nn.Conv2d): Depthwise grouped convolution.
        grid_w (int): Grid width used to infer height at forward time.
    """

    def __init__(self, channels: int, grid_w: int, kernel_size: int = 3) -> None:
        """Allocate depthwise Conv2d with no bias."""
        super().__init__()
        self.grid_w = grid_w
        self.channels = channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply depthwise 3×3 conv after reshaping to 2-D grid.

        Args:
            x: Token sequence ``[B, T, C]``.  ``T % grid_w`` must equal 0.

        Returns:
            Tensor of shape ``[B, T, C]`` after depthwise 2-D convolution.
        """
        B, T, C = x.shape
        H = T // self.grid_w
        x = x.reshape(B, H, self.grid_w, C)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.conv(x)  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x.reshape(B, T, C)
        return x

    def flop_count(self, num_tokens: int, inference: bool = False) -> int:
        r"""Count FLOPs for the depthwise convolution (one sample).

        ``FLOPs = 2 * T * C * kernel_size^2`` — each output element is the
        sum of ``kernel_size^2`` multiply-adds, but the conv is depthwise so
        no cross-channel mixing.

        Args:
            num_tokens: Sequence length ``T``.
            inference: Unused; provided for interface compatibility.

        Returns:
            Integer FLOPs.
        """
        k = self.conv.kernel_size[0]
        return 2 * num_tokens * self.channels * k * k

    def extra_repr(self) -> str:
        """Return ``'channels=..., grid_w=...'`` for PyTorch's module repr."""
        return f"channels={self.channels}, grid_w={self.grid_w}"
