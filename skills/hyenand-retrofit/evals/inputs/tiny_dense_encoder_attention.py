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

"""Small dense vision encoder with runtime rectangular grids and no CLS token."""

import torch
from torch import nn
from torch.nn import functional as F


class Attention(nn.Module):
    """Single-head self-attention with explicit pretrained-compatible projections."""

    def __init__(self, dim):
        """Build the input and output projections."""
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """Mix dense patch tokens."""
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        y = F.scaled_dot_product_attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))
        return self.proj(y.squeeze(1))


class Block(nn.Module):
    """Pre-norm attention and MLP residual block."""

    def __init__(self, dim):
        """Build one encoder block."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x):
        """Return one feature per input patch."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TinyDenseEncoder(nn.Module):
    """Return every patch feature for variable-size images divisible by patch size."""

    def __init__(self, dim=16, patch_size=4, depth=2):
        """Construct the patch embedding and encoder blocks."""
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, image):
        """Encode row-major patch tokens without a classification readout."""
        patches = self.patch_embed(image)
        x = patches.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


if __name__ == "__main__":
    model = TinyDenseEncoder()
    print(model(torch.randn(2, 3, 12, 20)).shape)
