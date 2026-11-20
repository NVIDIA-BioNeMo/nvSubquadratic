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

"""Tests for anisotropic kernel grids in ``nvsubquadratic.modules.kernels_nd``.

These tests cover the per-axis ``L_cache`` support added to the SIREN and
Random Fourier kernel families, plus the CKConvND plumbing that injects
``L_cache`` and ``grid_size`` into the kernel and mask configs.

Covers:
    - ``_normalize_l_cache``: scalar broadcast, sequence pass-through,
      validation errors (length, lower bound, type).
    - ``SIRENPositionalEmbeddingND`` / ``RandomFourierPositionalEmbeddingND``:
      anisotropic grid_cache shape, scalar/sequence init equivalence,
      per-axis runtime cache extension preserves step size, sub-cache
      slicing centers correctly along each axis.
    - ``SIRENKernelND`` / ``RandomFourierKernelND``: kernel shape on
      anisotropic grids, Wang init scaling uses ``prod(L_per_axis)``
      (not ``L**data_dim``), gradient flow.
    - Subclasses (``MultiOmegaSIRENKernelND``, learnable-ω variants):
      sequence ``L_cache`` flows through.
    - Repeated runtime cache extension: growing an axis a second time, or
      growing another axis later, keeps every point at a multiple of the
      original step, so the kernel at ``L_cache`` does not depend on which
      longer inputs came before.
    - ``CKConvND``: scalar and sequence ``L_cache`` both reach the kernel
      and mask correctly under ``grid_type='single'`` and ``'double'``.
    - First-layer precision: the SIREN, multi-ω₀, learnable-ω₀ and
      random-Fourier embeddings keep their projection and sine/cosine in
      float32 under bf16 autocast (values and parameter gradients against a
      float64 reference, with and without a bias) and with bf16 weights; the
      SIREN embedding also with float64 weights and, on CUDA, under
      ``torch.compile`` and TF32.  Hooks on the first layer still fire, and
      whole embeddings or kernels pickled with the older plain
      ``torch.nn.Linear`` first layer get the float32 one when unpickled.

Usage (CPU only):
    PYTHONPATH=. conda run -n nv-subq python -m pytest \\
        tests/modules/test_kernels_nd.py -v -o addopts=""
"""

import contextlib
import copy
import math
import pickle

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import (
    BlockDiagonalLearnableOmegaSIRENKernelND,
    BlockDiagonalMultiOmegaSIRENKernelND,
    LearnableOmegaSIRENKernelND,
    LearnableOmegaSIRENPositionalEmbeddingND,
    MultiOmegaSIRENKernelND,
    MultiOmegaSIRENPositionalEmbeddingND,
    RandomFourierKernelND,
    RandomFourierPositionalEmbeddingND,
    SIRENKernelND,
    SIRENPositionalEmbeddingND,
    _FirstLayerLinear,
    _normalize_l_cache,
)
from nvsubquadratic.modules.masks_nd import GaussianModulationND


# ---------------------------------------------------------------------------
# _normalize_l_cache
# ---------------------------------------------------------------------------


class TestNormalizeLCache:
    def test_scalar_broadcasts(self):
        assert _normalize_l_cache(8, 1) == (8,)
        assert _normalize_l_cache(8, 3) == (8, 8, 8)

    def test_tuple_passes_through(self):
        assert _normalize_l_cache((8, 64, 64), 3) == (8, 64, 64)

    def test_list_passes_through(self):
        assert _normalize_l_cache([8, 16], 2) == (8, 16)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            _normalize_l_cache((8, 64), 3)

    def test_too_small_raises(self):
        with pytest.raises(ValueError, match=">= 2"):
            _normalize_l_cache(1, 3)
        with pytest.raises(ValueError, match=">= 2"):
            _normalize_l_cache((8, 1, 16), 3)

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="bool"):
            _normalize_l_cache(True, 3)

    def test_str_rejected(self):
        with pytest.raises(TypeError, match=r"L_cache must be"):
            _normalize_l_cache("64", 3)


# ---------------------------------------------------------------------------
# SIRENPositionalEmbeddingND — per-axis L_cache
# ---------------------------------------------------------------------------


class TestSIRENPosEmbAnisotropic:
    def test_anisotropic_grid_cache_shape(self):
        emb = SIRENPositionalEmbeddingND(data_dim=3, embedding_dim=4, L_cache=(8, 32, 64), omega_0=1.0)
        # cache shape is (1, 2*L_0 - 1, 2*L_1 - 1, 2*L_2 - 1, data_dim)
        assert emb.grid_cache.shape == (1, 15, 63, 127, 3)
        assert emb.L_cache_per_axis == (8, 32, 64)

    def test_anisotropic_forward_full_resolution(self):
        emb = SIRENPositionalEmbeddingND(data_dim=3, embedding_dim=4, L_cache=(8, 32, 64), omega_0=1.0)
        out, grid = emb(seq_lens=(8, 32, 64))
        assert out.shape == (1, 15, 63, 127, 4)
        assert grid.shape == (1, 15, 63, 127, 3)

    def test_anisotropic_forward_subgrid_slice(self):
        # Smaller seq_lens than L_cache_per_axis → centered slice on each axis.
        emb = SIRENPositionalEmbeddingND(data_dim=2, embedding_dim=4, L_cache=(8, 16), omega_0=1.0)
        _, grid = emb(seq_lens=(4, 8))
        assert grid.shape == (1, 7, 15, 2)
        # Center coordinate must be 0 along every axis (cache is centered).
        center = grid[0, 3, 7]
        torch.testing.assert_close(center, torch.zeros(2, dtype=center.dtype))

    def test_scalar_and_uniform_sequence_are_equivalent(self):
        # L_cache=8 should produce the same cache as L_cache=(8, 8, 8).
        torch.manual_seed(0)
        emb_scalar = SIRENPositionalEmbeddingND(data_dim=3, embedding_dim=4, L_cache=8, omega_0=1.0)
        torch.manual_seed(0)
        emb_seq = SIRENPositionalEmbeddingND(data_dim=3, embedding_dim=4, L_cache=(8, 8, 8), omega_0=1.0)
        torch.testing.assert_close(emb_scalar.grid_cache, emb_seq.grid_cache)
        torch.testing.assert_close(emb_scalar.linear.weight, emb_seq.linear.weight)
        assert emb_scalar.L_cache_per_axis == emb_seq.L_cache_per_axis == (8, 8, 8)

    def test_per_axis_cache_extension_preserves_step(self):
        # Build with L_cache=(8, 16); call with seq_lens=(12, 16) to grow only axis 0.
        emb = SIRENPositionalEmbeddingND(data_dim=2, embedding_dim=2, L_cache=(8, 16), omega_0=1.0)
        original_step_axis_0 = emb.step_sizes[0]  # 1/(8-1) = 1/7
        _, grid = emb(seq_lens=(12, 16))
        # Axis 1 unchanged; axis 0 grew to 12.
        assert emb.L_cache_per_axis == (12, 16)
        # Slice covers seq_lens=(12, 16) → grid shape (1, 23, 31, 2).
        assert grid.shape == (1, 23, 31, 2)
        # Step size along axis 0 must equal the *original* (1/7), not the
        # naive new spacing 2/(2*12-2)=1/11.
        actual_step_axis_0 = float(grid[0, 1, 0, 0] - grid[0, 0, 0, 0])
        torch.testing.assert_close(
            torch.tensor(actual_step_axis_0),
            torch.tensor(original_step_axis_0),
            rtol=1e-5,
            atol=1e-6,
        )
        # Step size along axis 1 must still be 1/(16-1)=1/15.
        actual_step_axis_1 = float(grid[0, 0, 1, 1] - grid[0, 0, 0, 1])
        torch.testing.assert_close(
            torch.tensor(actual_step_axis_1),
            torch.tensor(1.0 / 15),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_l_cache_attribute_preserves_input(self):
        emb_scalar = SIRENPositionalEmbeddingND(data_dim=3, embedding_dim=4, L_cache=10, omega_0=1.0)
        emb_seq = SIRENPositionalEmbeddingND(data_dim=3, embedding_dim=4, L_cache=(8, 16, 32), omega_0=1.0)
        assert emb_scalar.L_cache == 10
        assert emb_seq.L_cache == (8, 16, 32)
        assert emb_scalar.L_cache_per_axis == (10, 10, 10)
        assert emb_seq.L_cache_per_axis == (8, 16, 32)


# ---------------------------------------------------------------------------
# SIRENKernelND — Wang init uses prod(L_per_axis)
# ---------------------------------------------------------------------------


class TestSIRENKernelAnisotropic:
    def _build(self, L_cache):
        torch.manual_seed(0)
        return SIRENKernelND(
            out_dim=4,
            data_dim=3,
            mlp_hidden_dim=8,
            num_layers=2,
            embedding_dim=8,
            omega_0=10.0,
            L_cache=L_cache,
            use_bias=True,
        )

    def test_anisotropic_kernel_shape(self):
        kern = self._build(L_cache=(8, 32, 64))
        out, _ = kern(seq_lens=(8, 32, 64))
        assert out.shape == (1, 15, 63, 127, 4)

    def test_wang_init_uses_product_of_per_axis(self):
        # Manually compare out_linear weight std before/after Wang scaling.
        # The pre-Wang std equals the SIREN-init std for a (mlp_hidden_dim x out_dim) layer.
        # We verify that the *ratio* between two configs equals the expected sqrt(prod ratio).
        torch.manual_seed(0)
        kern_iso = SIRENKernelND(
            out_dim=4,
            data_dim=3,
            mlp_hidden_dim=8,
            num_layers=2,
            embedding_dim=8,
            omega_0=10.0,
            L_cache=(16, 16, 16),
            use_bias=True,
        )
        torch.manual_seed(0)
        kern_aniso = SIRENKernelND(
            out_dim=4,
            data_dim=3,
            mlp_hidden_dim=8,
            num_layers=2,
            embedding_dim=8,
            omega_0=10.0,
            L_cache=(8, 16, 32),
            use_bias=True,
        )
        # Same RNG ⇒ pre-Wang weights are identical; post-Wang they differ by
        # sqrt(prod_aniso / prod_iso).  Hence the ratio of L2-norms equals
        # sqrt(prod_aniso / prod_iso).
        prod_iso = 16 * 16 * 16
        prod_aniso = 8 * 16 * 32
        ratio_expected = math.sqrt(prod_iso / prod_aniso)
        ratio_actual = (kern_aniso.out_linear.weight.norm() / kern_iso.out_linear.weight.norm()).item()
        assert math.isclose(ratio_actual, ratio_expected, rel_tol=1e-5), (
            f"Wang scaling mismatch: actual ratio={ratio_actual:.6f}, expected={ratio_expected:.6f}"
        )

    def test_uniform_sequence_matches_scalar(self):
        # L_cache=10 vs (10, 10, 10) → same kernel weights (same RNG).
        torch.manual_seed(7)
        kern_scalar = SIRENKernelND(
            out_dim=4,
            data_dim=3,
            mlp_hidden_dim=8,
            num_layers=2,
            embedding_dim=8,
            omega_0=5.0,
            L_cache=10,
            use_bias=True,
        )
        torch.manual_seed(7)
        kern_seq = SIRENKernelND(
            out_dim=4,
            data_dim=3,
            mlp_hidden_dim=8,
            num_layers=2,
            embedding_dim=8,
            omega_0=5.0,
            L_cache=(10, 10, 10),
            use_bias=True,
        )
        torch.testing.assert_close(kern_scalar.out_linear.weight, kern_seq.out_linear.weight)
        torch.testing.assert_close(
            kern_scalar.positional_embedding.linear.weight,
            kern_seq.positional_embedding.linear.weight,
        )

    def test_gradients_flow(self):
        kern = self._build(L_cache=(8, 16, 16))
        out, _ = kern(seq_lens=(8, 16, 16))
        out.sum().backward()
        assert all(p.grad is not None for p in kern.parameters() if p.requires_grad), (
            "every kernel parameter should receive a gradient"
        )


# ---------------------------------------------------------------------------
# RandomFourierPositionalEmbeddingND / RandomFourierKernelND
# ---------------------------------------------------------------------------


class TestRandomFourierAnisotropic:
    def test_pos_emb_shape(self):
        emb = RandomFourierPositionalEmbeddingND(data_dim=3, embedding_dim=4, L_cache=(4, 8, 16), omega_0=1.0)
        assert emb.grid_cache.shape == (1, 7, 15, 31, 3)
        out, grid = emb(seq_lens=(4, 8, 16))
        assert out.shape == (1, 7, 15, 31, 4)
        assert grid.shape == (1, 7, 15, 31, 3)

    def test_kernel_shape_and_wang(self):
        kern = RandomFourierKernelND(
            out_dim=4,
            data_dim=2,
            mlp_hidden_dim=4,
            num_layers=2,
            embedding_dim=4,
            omega_0=1.0,
            L_cache=(8, 16),
            use_bias=True,
            nonlinear_cfg=LazyConfig(torch.nn.GELU)(),
        )
        out, _ = kern(seq_lens=(8, 16))
        assert out.shape == (1, 15, 31, 4)
        # Wang scaling uses prod(L_cache_per_axis)=128, matching ``isotropic`` 8x16.
        assert kern.L_cache_per_axis == (8, 16)


# ---------------------------------------------------------------------------
# Subclass pass-through: MultiOmega / Learnable-ω₀ variants
# ---------------------------------------------------------------------------


class TestSubclassPassThrough:
    def test_multi_omega_sequence_l_cache(self):
        kern = MultiOmegaSIRENKernelND(
            out_dim=4,
            data_dim=3,
            mlp_hidden_dim=8,
            num_layers=2,
            embedding_dim=8,
            omega_0_per_row=[1.0] * 8,
            L_cache=(8, 16, 32),
            use_bias=True,
        )
        out, _ = kern(seq_lens=(8, 16, 32))
        assert out.shape == (1, 15, 31, 63, 4)
        assert kern.positional_embedding.L_cache_per_axis == (8, 16, 32)

    def test_block_diagonal_multi_omega_sequence_l_cache(self):
        kern = BlockDiagonalMultiOmegaSIRENKernelND(
            out_dim=8,
            data_dim=3,
            mlp_hidden_dim=8,
            num_layers=2,
            embedding_dim=8,
            L_cache=(8, 16, 32),
            use_bias=True,
            num_blocks=2,
            omega_0_min=1.0,
            omega_0_max=4.0,
        )
        out, _ = kern(seq_lens=(8, 16, 32))
        assert out.shape == (1, 15, 31, 63, 8)

    def test_learnable_omega_sequence_l_cache(self):
        kern = LearnableOmegaSIRENKernelND(
            out_dim=4,
            data_dim=3,
            mlp_hidden_dim=8,
            num_layers=2,
            embedding_dim=8,
            omega_0=10.0,
            L_cache=(8, 16, 32),
            use_bias=True,
        )
        out, _ = kern(seq_lens=(8, 16, 32))
        assert out.shape == (1, 15, 31, 63, 4)
        assert kern.positional_embedding.L_cache_per_axis == (8, 16, 32)

    def test_block_diagonal_learnable_omega_sequence_l_cache(self):
        kern = BlockDiagonalLearnableOmegaSIRENKernelND(
            out_dim=8,
            data_dim=3,
            mlp_hidden_dim=8,
            num_layers=2,
            embedding_dim=8,
            L_cache=(8, 16, 32),
            use_bias=True,
            num_blocks=2,
        )
        out, _ = kern(seq_lens=(8, 16, 32))
        assert out.shape == (1, 15, 31, 63, 8)


# ---------------------------------------------------------------------------
# Repeated runtime cache extension: the grid must not depend on call history
# ---------------------------------------------------------------------------

EXTENDING_EMBEDDINGS = [
    SIRENPositionalEmbeddingND,
    LearnableOmegaSIRENPositionalEmbeddingND,
    RandomFourierPositionalEmbeddingND,
]


def _multiples_of_step(seq_len: int, step: float) -> torch.Tensor:
    """Coordinates ``k * step`` for ``k = -(seq_len - 1), ..., seq_len - 1``, as float32."""
    return (torch.arange(-(seq_len - 1), seq_len, dtype=torch.float64) * step).to(torch.float32)


@pytest.mark.parametrize("embedding_cls", EXTENDING_EMBEDDINGS, ids=lambda cls: cls.__name__)
class TestRepeatedCacheExtension:
    """Every grid point stays at ``k / (L_cache - 1)`` however the cache grew.

    ``test_per_axis_cache_extension_preserves_step`` covers a single extension.
    These grow the cache again, which must not rescale an axis that already
    grew.
    """

    def test_second_extension_keeps_original_step(self, embedding_cls):
        emb = embedding_cls(data_dim=1, embedding_dim=4, L_cache=8, omega_0=1.0)
        emb(seq_lens=(12,))
        _, grid = emb(seq_lens=(16,))
        torch.testing.assert_close(grid[0, :, 0], _multiples_of_step(16, 1 / 7))

    def test_growing_another_axis_keeps_extended_axis_step(self, embedding_cls):
        emb = embedding_cls(data_dim=2, embedding_dim=4, L_cache=(8, 16), omega_0=1.0)
        emb(seq_lens=(12, 16))  # grow axis 0 only
        # Then grow axis 1 only, asking for less than the cache on axis 0, so
        # the rebuild cannot take axis 0's extent from the request.
        _, grid = emb(seq_lens=(4, 20))
        torch.testing.assert_close(grid[0, :, 0, 0], _multiples_of_step(4, 1 / 7))
        torch.testing.assert_close(grid[0, 0, :, 1], _multiples_of_step(20, 1 / 15))


def _siren_kernel(L_cache: tuple[int, ...]) -> SIRENKernelND:
    return SIRENKernelND(
        out_dim=4,
        data_dim=len(L_cache),
        mlp_hidden_dim=8,
        num_layers=2,
        embedding_dim=8,
        omega_0=1.0,
        L_cache=L_cache,
        use_bias=True,
    )


def _random_fourier_kernel(L_cache: tuple[int, ...]) -> RandomFourierKernelND:
    return RandomFourierKernelND(
        out_dim=4,
        data_dim=len(L_cache),
        mlp_hidden_dim=4,
        num_layers=2,
        embedding_dim=4,
        omega_0=1.0,
        L_cache=L_cache,
        use_bias=True,
        nonlinear_cfg=LazyConfig(torch.nn.GELU)(),
    )


# (L_cache, calls that grow the cache). In 2D and 3D each call grows a
# different axis and asks for less than the cache on an axis that already grew.
EXTENSION_HISTORIES = {
    "1d": ((8,), [(12,), (16,)]),
    "2d": ((8, 16), [(12, 16), (4, 20), (14, 6)]),
    "3d": ((4, 6, 8), [(6, 6, 8), (2, 9, 8), (5, 3, 12)]),
}


@pytest.mark.parametrize("history", EXTENSION_HISTORIES.values(), ids=EXTENSION_HISTORIES.keys())
@pytest.mark.parametrize("build_kernel", [_siren_kernel, _random_fourier_kernel], ids=["siren", "random_fourier"])
def test_kernel_does_not_depend_on_extension_history(build_kernel, history):
    """A kernel grown call by call must equal one grown straight to the final cache, at every size."""
    L_cache, calls = history
    final = tuple(max(sizes) for sizes in zip(L_cache, *calls))
    torch.manual_seed(0)
    stepwise = build_kernel(L_cache)
    torch.manual_seed(0)
    direct = build_kernel(L_cache)
    before, _ = stepwise(seq_lens=L_cache)
    for seq_lens in calls:
        stepwise(seq_lens=seq_lens)
    direct(seq_lens=final)
    for seq_lens in (L_cache, final, *calls):
        assert torch.equal(stepwise(seq_lens=seq_lens)[0], direct(seq_lens=seq_lens)[0]), seq_lens
    # Rebuilding the grid moves its points by float32 rounding only, so the
    # kernel at L_cache keeps its pre-extension value. omega_0=1 keeps that
    # rounding well inside the default tolerance.
    torch.testing.assert_close(stepwise(seq_lens=L_cache)[0], before)


# ---------------------------------------------------------------------------
# CKConvND plumbing: L_cache + grid_size injection on anisotropic configs
# ---------------------------------------------------------------------------


class TestCKConvAnisotropic:
    @staticmethod
    def _kernel_cfg(L_cache):
        return LazyConfig(SIRENKernelND)(
            out_dim=4,
            data_dim=3,
            mlp_hidden_dim=4,
            num_layers=2,
            embedding_dim=4,
            omega_0=10.0,
            L_cache=L_cache,
            use_bias=True,
        )

    @staticmethod
    def _mask_cfg():
        return LazyConfig(GaussianModulationND)(
            data_dim=3,
            num_channels=4,
            grid_size=1,  # placeholder; CKConv will overwrite based on L_cache
        )

    def test_double_grid_passes_l_cache_through(self):
        kern_cfg = self._kernel_cfg((8, 16, 32))
        mask_cfg = self._mask_cfg()
        ck = CKConvND(
            data_dim=3,
            hidden_dim=4,
            kernel_cfg=kern_cfg,
            mask_cfg=mask_cfg,
            grid_type="double",
            fft_padding="zero",
        )
        # double-grid does not shrink L_cache; the kernel sees the original sequence.
        assert ck.kernel.L_cache_per_axis == (8, 16, 32)
        # mask grid_size is set to the *largest* per-axis cache → 2*32 - 1.
        assert ck.mask.std_param.shape[0] == 3  # data_dim
        # Reverse-engineer ``grid_size`` from the stored ``_min_step``:
        # ``_min_step = 2 / (grid_size - 1)`` ⇒ grid_size = 2/_min_step + 1.
        gs = round(2.0 / ck.mask._min_step + 1)
        assert gs == 2 * 32 - 1

    def test_single_grid_halves_l_cache_per_axis(self):
        kern_cfg = self._kernel_cfg((8, 16, 32))
        mask_cfg = self._mask_cfg()
        ck = CKConvND(
            data_dim=3,
            hidden_dim=4,
            kernel_cfg=kern_cfg,
            mask_cfg=mask_cfg,
            grid_type="single",
            fft_padding="zero",
        )
        # single-grid halves each axis: (L+1)//2 → (4, 8, 16).
        assert ck.kernel.L_cache_per_axis == (4, 8, 16)
        # mask grid_size uses 2 * max(per_axis) - 1 = 2*16 - 1 = 31.
        gs = round(2.0 / ck.mask._min_step + 1)
        assert gs == 2 * 16 - 1

    def test_scalar_l_cache_unchanged_behaviour(self):
        # Backward compat: a scalar L_cache still goes through CKConv unchanged.
        kern_cfg = self._kernel_cfg(16)
        mask_cfg = self._mask_cfg()
        ck = CKConvND(
            data_dim=3,
            hidden_dim=4,
            kernel_cfg=kern_cfg,
            mask_cfg=mask_cfg,
            grid_type="single",
            fft_padding="zero",
        )
        # single-grid halves: (16+1)//2 = 8.  Scalar in → scalar out.
        assert ck.kernel.L_cache == 8
        assert ck.kernel.L_cache_per_axis == (8, 8, 8)
        gs = round(2.0 / ck.mask._min_step + 1)
        assert gs == 2 * 8 - 1

    def test_forward_runs_on_anisotropic_grid(self):
        kern_cfg = self._kernel_cfg((4, 8, 16))
        mask_cfg = self._mask_cfg()
        ck = CKConvND(
            data_dim=3,
            hidden_dim=4,
            kernel_cfg=kern_cfg,
            mask_cfg=mask_cfg,
            grid_type="double",
            fft_padding="zero",
        )
        # Input shape (B=1, C=4, D=4, H=8, W=16) — match L_cache exactly.
        x = torch.randn(1, 4, 4, 8, 16)
        out = ck(x, is_bhl_input=True)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# First-layer precision: the phase and its sine stay in float32
# ---------------------------------------------------------------------------


def _with_bias(emb):
    """Give the first layer a non-zero bias (every constructor initialises it to zero)."""
    with torch.no_grad():
        emb.linear.bias.uniform_(-1.0, 1.0)
    return emb


FIRST_LAYER_EMBEDDINGS = {
    "siren": lambda data_dim, L_cache: _with_bias(
        SIRENPositionalEmbeddingND(data_dim=data_dim, embedding_dim=16, L_cache=L_cache, omega_0=10.0)
    ),
    "multi_omega_siren": lambda data_dim, L_cache: _with_bias(
        MultiOmegaSIRENPositionalEmbeddingND(
            data_dim=data_dim, embedding_dim=16, L_cache=L_cache, omega_0_per_row=[1.0] * 8 + [30.0] * 8
        )
    ),
    "learnable_omega_siren": lambda data_dim, L_cache: _with_bias(
        LearnableOmegaSIRENPositionalEmbeddingND(data_dim=data_dim, embedding_dim=16, L_cache=L_cache, omega_0=10.0)
    ),
    "random_fourier": lambda data_dim, L_cache: _with_bias(
        RandomFourierPositionalEmbeddingND(data_dim=data_dim, embedding_dim=16, L_cache=L_cache, omega_0=10.0)
    ),
}
FIRST_LAYER_GRIDS = [(1, 4096), (2, (64, 48)), (3, 12)]


def _first_layer_reference(name, grid, weight, bias, omega_0_multiplier=None):
    """Float64 embedding computed directly from the first-layer parameters."""
    phase = grid.double() @ weight.double().T + bias.double()
    if name == "learnable_omega_siren":
        phase = phase * omega_0_multiplier.double()
    if name == "random_fourier":
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
    return torch.sin(phase)


def _relative_l2(got, want):
    return ((got.double() - want).norm() / want.norm()).item()


@pytest.mark.parametrize("data_dim, L_cache", FIRST_LAYER_GRIDS, ids=["1d", "2d", "3d"])
@pytest.mark.parametrize("name", FIRST_LAYER_EMBEDDINGS)
def test_first_layer_is_unaffected_by_autocast(name, data_dim, L_cache):
    """Autocast leaves the first layer in float32 (``F.linear`` used to run it in bf16)."""
    torch.manual_seed(0)
    emb = FIRST_LAYER_EMBEDDINGS[name](data_dim, L_cache)
    seq_lens = _normalize_l_cache(L_cache, data_dim)
    want, _ = emb(seq_lens)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got, _ = emb(seq_lens)
    assert got.dtype == torch.float32
    assert torch.equal(got, want)


@pytest.mark.parametrize("name", FIRST_LAYER_EMBEDDINGS)
def test_first_layer_with_bf16_weights_rounds_only_the_output(name):
    """bf16 parameters still get a float32 phase and sine; only the returned embedding is rounded."""
    torch.manual_seed(0)
    emb = FIRST_LAYER_EMBEDDINGS[name](1, 4096)
    for param in emb.parameters():
        param.data = param.data.to(torch.bfloat16).to(torch.float32)
    want, _ = emb((4096,))
    for param in emb.parameters():
        param.data = param.data.to(torch.bfloat16)
    got, _ = emb((4096,))
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, want.to(torch.bfloat16))


@pytest.mark.parametrize("data_dim, L_cache", FIRST_LAYER_GRIDS, ids=["1d", "2d", "3d"])
@pytest.mark.parametrize("name", FIRST_LAYER_EMBEDDINGS)
def test_first_layer_and_its_gradients_match_float64_under_autocast(name, data_dim, L_cache):
    """Under bf16 autocast the embedding and its parameter gradients match a float64 reference.

    With ``F.linear`` both were off by 3.5-22% (relative L2); now they agree to
    float32 rounding.
    """
    torch.manual_seed(0)
    emb = FIRST_LAYER_EMBEDDINGS[name](data_dim, L_cache)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got, grid = emb(_normalize_l_cache(L_cache, data_dim))
    cotangent = torch.randn(got.shape, dtype=torch.float64)
    got.backward(cotangent.to(got.dtype))

    params = {"weight": emb.linear.weight, "bias": emb.linear.bias}
    if name == "learnable_omega_siren":
        params["omega_0_scale"] = emb.omega_0_scale
    ref = {key: param.detach().double().requires_grad_() for key, param in params.items()}
    multiplier = emb.omega_0_const * ref["omega_0_scale"] if name == "learnable_omega_siren" else None
    want = _first_layer_reference(name, grid, ref["weight"], ref["bias"], multiplier)
    want.backward(cotangent)

    assert _relative_l2(got, want) < 1e-5
    for key, param in params.items():
        assert _relative_l2(param.grad, ref[key].grad) < 2e-5, key


NO_BIAS_EMBEDDINGS = {
    "siren": SIRENPositionalEmbeddingND,
    "learnable_omega_siren": LearnableOmegaSIRENPositionalEmbeddingND,
    "random_fourier": RandomFourierPositionalEmbeddingND,
}


@pytest.mark.parametrize("name", NO_BIAS_EMBEDDINGS)
def test_first_layer_without_bias_matches_float64_under_autocast(name):
    """``use_bias=False`` takes the projection's bias-free branch and still stays in float32."""
    torch.manual_seed(0)
    emb = NO_BIAS_EMBEDDINGS[name](data_dim=2, embedding_dim=16, L_cache=(64, 48), omega_0=10.0, use_bias=False)
    assert emb.linear.bias is None
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got, grid = emb((64, 48))
    multiplier = emb.omega_0_const * emb.omega_0_scale if name == "learnable_omega_siren" else None
    want = _first_layer_reference(name, grid, emb.linear.weight, torch.zeros(()), multiplier)
    assert _relative_l2(got, want) < 1e-5


@pytest.mark.parametrize("name", ["siren", "multi_omega_siren", "random_fourier"])
def test_first_layer_hooks_see_the_float32_phase(name):
    """``positional_embedding.linear`` is still called as a module, so its forward hooks fire.

    ``FiLMMonitorCallback`` reads the sine's argument through such a hook.
    """
    torch.manual_seed(0)
    emb = FIRST_LAYER_EMBEDDINGS[name](2, (64, 48))
    seen = []
    emb.linear.register_forward_hook(lambda module, args, output: seen.append(output))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, grid = emb((64, 48))
    assert len(seen) == 1
    assert seen[0].dtype == torch.float32
    phase = grid.double() @ emb.linear.weight.double().T + emb.linear.bias.double()
    assert (seen[0].double() - phase).abs().max() < 1e-4


def _as_older_pickle(module):
    """Pickle round trip of ``module`` as saved before ``_FirstLayerLinear`` existed.

    Pickle stores each child module's class, so an older whole module came
    back with a plain ``torch.nn.Linear`` first layer.
    """
    older = copy.deepcopy(module)
    for child in older.modules():
        if type(child) is _FirstLayerLinear:
            child.__class__ = torch.nn.Linear
    return pickle.loads(pickle.dumps(older))


def _bf16_parameters(module):
    """Cast only the parameters to bf16 (as a parameter-only cast does); the float32 grid stays."""
    for param in module.parameters():
        param.data = param.data.to(torch.bfloat16)
    return module


@pytest.mark.parametrize("name", FIRST_LAYER_EMBEDDINGS)
def test_first_layer_of_an_older_pickled_embedding_runs_in_float32(name):
    """A whole embedding pickled with the older plain ``torch.nn.Linear`` gets the float32 first layer.

    Without ``__setstate__`` the SIREN, multi-ω₀ and random-Fourier embeddings
    kept the bf16 projection under autocast (here 3.9%, 8.9% and 17% relative
    L2), and with bf16 parameters ``F.linear`` failed on the float32 grid.  The
    learnable-ω₀ embedding calls ``_first_layer_projection`` itself, so for it
    only the class changes.
    """
    torch.manual_seed(0)
    emb = FIRST_LAYER_EMBEDDINGS[name](2, (64, 48))
    older = _as_older_pickle(emb)
    assert type(older.linear) is _FirstLayerLinear
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(older((64, 48))[0], emb((64, 48))[0])
    assert torch.equal(_bf16_parameters(older)((64, 48))[0], _bf16_parameters(emb)((64, 48))[0])


@pytest.mark.parametrize("build_kernel", [_siren_kernel, _random_fourier_kernel], ids=["siren", "random_fourier"])
def test_first_layer_of_an_older_pickled_kernel_runs_in_float32(build_kernel):
    """Unpickling a whole kernel restores its embedding's float32 first layer too."""
    torch.manual_seed(0)
    kernel = build_kernel((16, 12))
    older = _as_older_pickle(kernel)
    assert type(older.positional_embedding.linear) is _FirstLayerLinear
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(older(seq_lens=(16, 12))[0], kernel(seq_lens=(16, 12))[0])


def test_siren_first_layer_keeps_every_coordinate_under_autocast():
    """At ``L_cache=8192`` all 16383 coordinates keep their own phase under bf16 autocast.

    bf16 has 8 significant bits, so rounding the coordinates merged up to 32
    neighbouring grid points into one value, and rounding the phase (up to
    about 63 rad here, where bf16 values are 0.25 apart) moved it by up to
    0.125 rad.
    """
    torch.manual_seed(0)
    emb = _with_bias(SIRENPositionalEmbeddingND(data_dim=1, embedding_dim=32, L_cache=8192, omega_0=10.0))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got, grid = emb((8192,))
    want = torch.sin(grid.double() @ emb.linear.weight.double().T + emb.linear.bias.double())
    assert (got.double() - want).abs().max() < 2e-5
    assert (got[0, 1:] != got[0, :-1]).any(dim=-1).all()


def test_siren_first_layer_with_float64_weights_stays_float64():
    """float64 parameters get a float64 phase and sine, not a float32 one."""
    torch.manual_seed(0)
    emb = _with_bias(SIRENPositionalEmbeddingND(data_dim=2, embedding_dim=16, L_cache=(64, 48), omega_0=10.0))
    emb.linear.double()
    got, grid = emb((64, 48))
    want = torch.sin(grid.double() @ emb.linear.weight.T + emb.linear.bias)
    assert got.dtype == torch.float64
    assert (got - want).abs().max() < 1e-12


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("data_dim, L_cache", FIRST_LAYER_GRIDS, ids=["1d", "2d", "3d"])
@pytest.mark.parametrize("setting", ["bf16_autocast", "tf32_matmul"])
def test_siren_first_layer_on_cuda_matches_float64(setting, data_dim, L_cache):
    """On CUDA, eager and ``torch.compile`` both match a float64 reference under bf16 autocast or TF32.

    With ``F.linear``, CUDA autocast ran the first layer in bf16, except that
    Inductor rewrites a one-feature (1D) projection into float32 elementwise
    math, so only compiled 1D was correct.  Without autocast, a float32 matmul
    may run in TF32 under ``torch.set_float32_matmul_precision("high")``, which
    ``experiments/run.py`` sets.
    """
    torch.manual_seed(0)
    emb = SIRENPositionalEmbeddingND(data_dim=data_dim, embedding_dim=32, L_cache=L_cache, omega_0=10.0)
    emb = _with_bias(emb).cuda()
    seq_lens = _normalize_l_cache(L_cache, data_dim)
    compiled = torch.compile(emb)
    previous_precision = torch.get_float32_matmul_precision()
    try:
        if setting == "tf32_matmul":
            torch.set_float32_matmul_precision("high")
            context = contextlib.nullcontext()
        else:
            context = torch.autocast("cuda", dtype=torch.bfloat16)
        with context:
            eager, grid = emb(seq_lens)
            got, _ = compiled(seq_lens)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    want = torch.sin(grid.double() @ emb.linear.weight.double().T + emb.linear.bias.double())
    assert (eager.double() - want).abs().max() < 2e-5
    assert (got.double() - want).abs().max() < 2e-5
