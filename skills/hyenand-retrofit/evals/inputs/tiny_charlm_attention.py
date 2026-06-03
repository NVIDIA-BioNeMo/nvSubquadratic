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

"""Tiny char-level causal LM with multi-head self-attention.

4 transformer blocks, hidden_dim=128, num_heads=4, sequence length 256,
vocab_size=256. Each block applies a causal attention mask. Used as a
small test target for attention -> HyenaND retrofits on 1D causal hosts.

No CLS / prefix tokens. Retrofit must use causal-LM defaults:
``use_rope=True``, ``fft_padding="causal"``, mask
``parametrization="exp_decay"``, ``omega_0=100``.
"""

import torch
import torch.nn as nn


class CausalBlock(nn.Module):
    """Pre-norm transformer block with causal multi-head self-attention."""

    def __init__(self, dim: int = 128, num_heads: int = 4, mlp_ratio: float = 4.0):
        """Configure attention and MLP sublayers for hidden size ``dim``."""
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal attention and MLP; input shape ``[B, T, C]``."""
        T = x.size(1)
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TinyCharLM(nn.Module):
    """Small causal character LM for HyenaND retrofit evals."""

    def __init__(
        self,
        vocab_size: int = 256,
        seq_len: int = 256,
        dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
    ):
        """Build embeddings, ``num_blocks`` causal blocks, and output head."""
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.blocks = nn.ModuleList([CausalBlock(dim, num_heads) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return logits for token indices ``[B, T]``."""
        x = self.tok_embed(idx) + self.pos_embed[:, : idx.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


if __name__ == "__main__":
    model = TinyCharLM()
    idx = torch.randint(0, 256, (2, 256))
    y = model(idx)
    print(y.shape)  # [2, 256, 256]
