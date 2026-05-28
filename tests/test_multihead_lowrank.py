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

# TODO: Add license header here

"""Tests for low-rank multi-head CKConv and FFT convolution.

Tests verify:
1. Low-rank FFT conv ops produce correct shapes
2. Low-rank FFT conv matches full-rank when rank == head_dim (exact equivalence)
3. CKConvMultiheadND with kernel_rank produces correct shapes and gradients
4. CKConvMultiheadND with kernel_rank + FiLM (batched kernels)
5. CKConvMultiheadND without kernel_rank (full-rank) still works
6. Low-rank reduces parameter count vs full-rank

Run:
    PYTHONPATH=. python -m pytest tests/test_multihead_lowrank.py -v
"""

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_multihead_nd import CKConvMultiheadND
from nvsubquadratic.modules.film import KernelFiLMGenerator
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.ops.fftconv_multihead import (
    fftconv2d_multihead_bhl,
    fftconv2d_multihead_lowrank_bhl,
    fftconv2d_multihead_lowrank_circular_bhl,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def multihead_dims():
    """Standard dimensions used across tests."""
    return {"B": 2, "num_heads": 6, "head_dim": 64, "H": 14, "W": 14, "hidden_dim": 384}


@pytest.fixture
def film_cfg():
    """FiLM generator config for conditioned kernels."""
    return LazyConfig(KernelFiLMGenerator)(
        cond_dim=384,
        kernel_hidden_dim=32,
        num_film_layers=2,
        film_hidden_dim=64,
    )


def _make_ckconv(*, hidden_dim, num_heads, kernel_rank=None, film_cfg=None):
    """Helper to build a CKConvMultiheadND with given rank and optional FiLM."""
    head_dim = hidden_dim // num_heads
    if kernel_rank is not None:
        out_dim = num_heads * 2 * kernel_rank * head_dim
    else:
        out_dim = num_heads * head_dim * head_dim

    return CKConvMultiheadND(
        data_dim=2,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        kernel_cfg=LazyConfig(SIRENKernelND)(
            data_dim=2,
            out_dim=out_dim,
            mlp_hidden_dim=32,
            num_layers=3,
            embedding_dim=32,
            omega_0=10.0,
            L_cache=15,
            use_bias=True,
            hidden_omega_0=1.0,
            film_cfg=film_cfg,
        ),
        mask_cfg=LazyConfig(torch.nn.Identity)(),
        grid_type="double",
        fft_padding="zero",
        kernel_rank=kernel_rank,
    )


# ─── 1. Low-rank FFT conv op tests ──────────────────────────────────────────────


class TestLowRankFFTConv:
    def test_lowrank_output_shape(self, multihead_dims):
        """Low-rank FFT conv produces the correct output shape."""
        d = multihead_dims
        rank = 8
        x = torch.randn(d["B"], d["num_heads"], d["head_dim"], d["H"], d["W"])
        kernel_u = torch.randn(d["num_heads"], d["head_dim"], rank, 28, 28)
        kernel_v = torch.randn(d["num_heads"], rank, d["head_dim"], 28, 28)
        shortcut = torch.randn(d["hidden_dim"])

        out = fftconv2d_multihead_lowrank_bhl(x, kernel_u, kernel_v, shortcut)
        assert out.shape == (d["B"], d["num_heads"], d["head_dim"], d["H"], d["W"])

    def test_lowrank_circular_output_shape(self, multihead_dims):
        """Circular low-rank FFT conv produces the correct output shape."""
        d = multihead_dims
        rank = 4
        x = torch.randn(d["B"], d["num_heads"], d["head_dim"], d["H"], d["W"])
        kernel_u = torch.randn(d["num_heads"], d["head_dim"], rank, d["H"], d["W"])
        kernel_v = torch.randn(d["num_heads"], rank, d["head_dim"], d["H"], d["W"])
        shortcut = torch.randn(d["hidden_dim"])

        out = fftconv2d_multihead_lowrank_circular_bhl(x, kernel_u, kernel_v, shortcut)
        assert out.shape == (d["B"], d["num_heads"], d["head_dim"], d["H"], d["W"])

    def test_lowrank_matches_manual_freq_domain_computation(self):
        """Verify low-rank FFT conv matches manual K_fft = U_fft @ V_fft computation.

        In frequency domain, the low-rank conv computes:
            z_fft[n,r,f] = sum_i V_fft[n,r,i,f] * x_fft[n,i,f]
            y_fft[n,o,f] = sum_r U_fft[n,o,r,f] * z_fft[n,r,f]

        This is equivalent to full-rank with K_fft[n,o,i,f] = sum_r U_fft[n,o,r,f] * V_fft[n,r,i,f].
        We verify by constructing K_fft explicitly and comparing.
        """
        B, num_heads, head_dim, H, W = 1, 2, 8, 7, 7
        K_x, K_y = 14, 14
        rank = 4

        x = torch.randn(B, num_heads, head_dim, H, W)
        kernel_u = torch.randn(num_heads, head_dim, rank, K_x, K_y) * 0.01
        kernel_v = torch.randn(num_heads, rank, head_dim, K_x, K_y) * 0.01
        shortcut = torch.randn(num_heads * head_dim)

        # Compute via low-rank function
        out_lr = fftconv2d_multihead_lowrank_bhl(x, kernel_u, kernel_v, shortcut)

        # Compute manually: K_fft = U_fft @ V_fft, then full-rank conv with K_fft
        fft_h = min(H + (K_x + 1) // 2, 2 * H)
        fft_w = min(W + (K_y + 1) // 2, 2 * W)

        x_fft = torch.fft.rfft2(x, s=(fft_h, fft_w))
        u_fft = torch.fft.rfft2(kernel_u, s=(fft_h, fft_w))
        v_fft = torch.fft.rfft2(kernel_v, s=(fft_h, fft_w))

        # K_fft[n,o,i,f1,f2] = sum_r U_fft[n,o,r,f1,f2] * V_fft[n,r,i,f1,f2]
        k_fft = torch.einsum("norhw,nrihw->noihw", u_fft, v_fft)

        # Full-rank conv with synthesized K_fft
        out_fft = torch.einsum("bnihw,noihw->bnohw", x_fft, k_fft)
        crop_h = K_x // 2
        crop_w = K_y // 2
        out_manual_full = torch.fft.irfft2(out_fft, s=(fft_h, fft_w))
        out_manual = out_manual_full[..., crop_h : crop_h + H, crop_w : crop_w + W]
        # Add shortcut
        shortcut_reshaped = shortcut.view(1, num_heads, head_dim, 1, 1)
        out_manual = out_manual + x * shortcut_reshaped

        torch.testing.assert_close(out_lr, out_manual, atol=1e-4, rtol=1e-4)

    def test_lowrank_matches_fullrank_when_rank_equals_head_dim(self):
        """Low-rank with rank == head_dim exactly matches full-rank conv.

        The two-step low-rank conv computes y = U @ (V @ x) which by
        associativity equals y = (U @ V) @ x = K @ x. When rank == head_dim,
        setting U = spatial delta (identity in freq domain) and V = K gives
        K_eff = I @ K = K, so the output matches full-rank exactly.
        """
        B, num_heads, head_dim, H, W = 2, 2, 8, 7, 7
        K_x, K_y = 14, 14
        rank = head_dim  # full rank

        x = torch.randn(B, num_heads, head_dim, H, W)
        # Full-rank kernel: [num_heads, head_dim, head_dim, K_x, K_y]
        kernel_full = torch.randn(num_heads, head_dim, head_dim, K_x, K_y) * 0.01
        shortcut = torch.randn(num_heads * head_dim)

        # U = identity at each spatial position: [num_heads, head_dim, rank, K_x, K_y]
        # V = kernel: [num_heads, rank, head_dim, K_x, K_y]
        # But U and V are in spatial domain and K_fft = FFT(U) @ FFT(V) ≠ FFT(U @ V)
        # So we need to set U to a spatial delta (identity only at center, zero elsewhere)
        # and V = kernel, so that FFT(U) = I (constant) and K_fft = I @ FFT(V) = FFT(V) = FFT(K).
        kernel_u = torch.zeros(num_heads, head_dim, rank, K_x, K_y)
        # Place identity at spatial center (the DC/origin point for the kernel)
        for n in range(num_heads):
            for d in range(head_dim):
                kernel_u[n, d, d, 0, 0] = 1.0  # delta at origin
        kernel_v = kernel_full.clone()  # V has shape [num_heads, head_dim_out=rank, head_dim_in, K_x, K_y]

        out_full = fftconv2d_multihead_bhl(x, kernel_full, shortcut)
        out_lr = fftconv2d_multihead_lowrank_bhl(x, kernel_u, kernel_v, shortcut)

        torch.testing.assert_close(out_lr, out_full, atol=1e-4, rtol=1e-4)

    def test_lowrank_gradient_flow(self, multihead_dims):
        """Gradients flow through low-rank FFT conv."""
        d = multihead_dims
        rank = 8
        x = torch.randn(d["B"], d["num_heads"], d["head_dim"], d["H"], d["W"], requires_grad=True)
        kernel_u = torch.randn(d["num_heads"], d["head_dim"], rank, 28, 28, requires_grad=True)
        kernel_v = torch.randn(d["num_heads"], rank, d["head_dim"], 28, 28, requires_grad=True)

        out = fftconv2d_multihead_lowrank_bhl(x, kernel_u, kernel_v)
        out.sum().backward()

        assert x.grad is not None
        assert kernel_u.grad is not None
        assert kernel_v.grad is not None


# ─── 2. CKConvMultiheadND module tests ──────────────────────────────────────────


class TestCKConvMultiheadNDLowRank:
    def test_lowrank_output_shape_blh(self, multihead_dims):
        """Low-rank CKConvMultiheadND produces correct shape with BLH input."""
        d = multihead_dims
        model = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=8)
        x = torch.randn(d["B"], d["H"], d["W"], d["hidden_dim"])
        out = model(x)
        assert out.shape == x.shape

    def test_lowrank_output_shape_bhl(self, multihead_dims):
        """Low-rank CKConvMultiheadND produces correct shape with BHL input."""
        d = multihead_dims
        model = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=8)
        x = torch.randn(d["B"], d["hidden_dim"], d["H"], d["W"])
        out = model(x, is_bhl_input=True)
        assert out.shape == x.shape

    def test_lowrank_gradient_flow(self, multihead_dims):
        """All parameters receive gradients in the low-rank path."""
        d = multihead_dims
        model = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=8)
        x = torch.randn(d["B"], d["H"], d["W"], d["hidden_dim"])
        out = model(x)
        out.sum().backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_lowrank_with_film(self, multihead_dims, film_cfg):
        """Low-rank works with FiLM conditioning (batched kernels)."""
        d = multihead_dims
        model = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=8, film_cfg=film_cfg)
        x = torch.randn(d["B"], d["H"], d["W"], d["hidden_dim"])
        cond = torch.randn(d["B"], d["hidden_dim"])
        out = model(x, conditioning=cond)
        assert out.shape == x.shape

        out.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_fullrank_still_works(self, multihead_dims, film_cfg):
        """Full-rank path (kernel_rank=None) is not broken."""
        d = multihead_dims
        model = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=None, film_cfg=film_cfg)
        x = torch.randn(d["B"], d["H"], d["W"], d["hidden_dim"])
        cond = torch.randn(d["B"], d["hidden_dim"])
        out = model(x, conditioning=cond)
        assert out.shape == x.shape

        out.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_lowrank_reduces_params(self, multihead_dims):
        """Low-rank model has fewer parameters than full-rank."""
        d = multihead_dims
        model_lr = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=8)
        model_full = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=None)

        params_lr = sum(p.numel() for p in model_lr.parameters())
        params_full = sum(p.numel() for p in model_full.parameters())

        assert params_lr < params_full, (
            f"Low-rank ({params_lr}) should have fewer params than full-rank ({params_full})"
        )
        # With rank=8 and head_dim=64, SIREN output is 4x smaller
        # so the SIREN out_linear should be ~4x smaller
        ratio = params_full / params_lr
        assert ratio > 2.0, f"Expected significant param reduction, got only {ratio:.1f}x"

    def test_extra_repr_includes_rank(self, multihead_dims):
        """extra_repr shows kernel_rank when set."""
        d = multihead_dims
        model = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=8)
        assert "kernel_rank=8" in model.extra_repr()

    def test_extra_repr_no_rank_when_fullrank(self, multihead_dims):
        """extra_repr does not show kernel_rank when using full-rank."""
        d = multihead_dims
        model = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=None)
        assert "kernel_rank" not in model.extra_repr()

    @pytest.mark.parametrize("rank", [1, 4, 8, 16])
    def test_various_ranks(self, multihead_dims, rank):
        """Different rank values all produce correct output."""
        d = multihead_dims
        model = _make_ckconv(hidden_dim=d["hidden_dim"], num_heads=d["num_heads"], kernel_rank=rank)
        x = torch.randn(d["B"], d["H"], d["W"], d["hidden_dim"])
        out = model(x)
        assert out.shape == x.shape
