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

# David W. Romero, 2025-09-09

"""Tests for PatchEmbedHierarchical and PatchMerging2D.

Validates:
1. PatchEmbedHierarchical: shape, channels-last in/out, LayerNorm applied
2. PatchMerging2D: shape contracts exactly (÷2 spatial, ×2 channels)
3. PatchMerging2D: odd-size padding path produces correct output shape
4. PatchMerging2D: gradient flow through norm and reduction linear
5. PatchMerging2D: Swin-exact slice order (x0/x1/x2/x3)
6. flop_count: analytic formulas are self-consistent
"""

import pytest
import torch
import torch.nn as nn

from nvsubquadratic.modules.patch_merging import PatchEmbedHierarchical, PatchMerging2D


# ── PatchEmbedHierarchical ────────────────────────────────────────────────────


class TestPatchEmbedHierarchical:
    def test_output_shape_224(self):
        stem = PatchEmbedHierarchical(in_channels=3, embed_dim=96)
        x = torch.randn(2, 224, 224, 3)
        y = stem(x)
        assert y.shape == (2, 56, 56, 96), f"Expected (2,56,56,96) got {y.shape}"

    def test_output_shape_generic(self):
        stem = PatchEmbedHierarchical(in_channels=3, embed_dim=64)
        x = torch.randn(1, 128, 128, 3)
        y = stem(x)
        assert y.shape == (1, 32, 32, 64)

    def test_channels_last_output(self):
        stem = PatchEmbedHierarchical(in_channels=3, embed_dim=96)
        x = torch.randn(1, 224, 224, 3)
        y = stem(x)
        # Last dim is embed_dim, not spatial
        assert y.shape[-1] == 96

    def test_layernorm_applied(self):
        """After LayerNorm the output should not be identical to a raw conv output."""
        stem = PatchEmbedHierarchical(in_channels=3, embed_dim=8)
        # Overwrite norm with Identity to compare
        stem_no_norm = PatchEmbedHierarchical(in_channels=3, embed_dim=8)
        stem_no_norm.proj.weight = stem.proj.weight
        stem_no_norm.proj.bias = stem.proj.bias
        stem_no_norm.norm = nn.Identity()

        x = torch.randn(1, 8, 8, 3)
        with torch.no_grad():
            y_with = stem(x)
            y_without = stem_no_norm(x)
        # They are the same only if the raw output is already normalised — unlikely
        assert not torch.allclose(y_with, y_without), "LayerNorm should change output"

    def test_gradient_flow(self):
        stem = PatchEmbedHierarchical(in_channels=3, embed_dim=16)
        x = torch.randn(1, 32, 32, 3, requires_grad=True)
        y = stem(x)
        y.sum().backward()
        assert x.grad is not None
        assert stem.proj.weight.grad is not None

    def test_flop_count(self):
        stem = PatchEmbedHierarchical(in_channels=3, embed_dim=96)
        flops = stem.flop_count(224, 224)
        assert flops > 0
        # Conv: 2 * 3 * 96 * 16 * (56*56)
        conv_flops = 2 * 3 * 96 * 16 * 56 * 56
        norm_flops = 2 * 56 * 56 * 96
        assert flops == conv_flops + norm_flops


# ── PatchMerging2D ─────────────────────────────────────────────────────────────


class TestPatchMerging2D:
    @pytest.mark.parametrize(
        "H,W,C",
        [
            (56, 56, 96),
            (28, 28, 192),
            (14, 14, 384),
        ],
    )
    def test_output_shape_even(self, H, W, C):
        pm = PatchMerging2D(dim=C)
        x = torch.randn(2, H, W, C)
        y = pm(x)
        assert y.shape == (2, H // 2, W // 2, 2 * C), f"Expected (2,{H // 2},{W // 2},{2 * C}), got {y.shape}"

    def test_full_swin_pyramid(self):
        """Verify Swin-T spatial progression 56→28→14→7."""
        x = torch.randn(1, 56, 56, 96)
        pm1 = PatchMerging2D(dim=96)
        pm2 = PatchMerging2D(dim=192)
        pm3 = PatchMerging2D(dim=384)

        x = pm1(x)
        assert x.shape == (1, 28, 28, 192)
        x = pm2(x)
        assert x.shape == (1, 14, 14, 384)
        x = pm3(x)
        assert x.shape == (1, 7, 7, 768)

    def test_odd_height_padding(self):
        """Odd H triggers the padding branch; output shape is still ⌈H/2⌉."""
        pm = PatchMerging2D(dim=32)
        x = torch.randn(1, 7, 7, 32)
        y = pm(x)
        assert y.shape == (1, 4, 4, 64)  # ⌈7/2⌉ = 4

    def test_odd_width_padding(self):
        pm = PatchMerging2D(dim=16)
        x = torch.randn(1, 6, 7, 16)
        y = pm(x)
        assert y.shape == (1, 3, 4, 32)

    def test_slice_order_matches_swin(self):
        """x0=(0,0), x1=(1,0), x2=(0,1), x3=(1,1) — Swin/VMamba order."""
        B, H, W, C = 1, 4, 4, 4
        pm = PatchMerging2D(dim=C)
        # Craft input so each spatial cell = unique integer
        x = torch.zeros(B, H, W, C)
        for i in range(H):
            for j in range(W):
                x[0, i, j, :] = float(i * W + j)

        # Manually compute expected cat
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        cat_ref = torch.cat([x0, x1, x2, x3], dim=-1)  # (1,2,2,16)

        # Run the module's norm+reduction with identity-ish weights to inspect pre-norm
        with torch.no_grad():
            pm.norm = nn.Identity()
            pm.reduction = nn.Identity()
            y = pm(x)
        assert torch.allclose(y, cat_ref), "Slice order does not match Swin convention"

    def test_gradient_flow(self):
        pm = PatchMerging2D(dim=16)
        x = torch.randn(1, 8, 8, 16, requires_grad=True)
        y = pm(x)
        y.sum().backward()
        assert x.grad is not None
        assert pm.reduction.weight.grad is not None
        assert pm.norm.weight.grad is not None

    def test_flop_count_even(self):
        dim = 96
        pm = PatchMerging2D(dim=dim)
        flops = pm.flop_count(56, 56)
        # H_out=28, W_out=28, tokens=784
        tokens = 28 * 28
        expected_norm = 2 * tokens * 4 * dim
        expected_linear = 2 * tokens * 4 * dim * 2 * dim
        assert flops == expected_norm + expected_linear

    def test_flop_count_odd(self):
        dim = 32
        pm = PatchMerging2D(dim=dim)
        flops = pm.flop_count(7, 7)
        # H_out=4, W_out=4
        tokens = 4 * 4
        expected = 2 * tokens * 4 * dim + 2 * tokens * 4 * dim * 2 * dim
        assert flops == expected
