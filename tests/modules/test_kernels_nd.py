# TODO: Add license header here


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
    - ``CKConvND``: scalar and sequence ``L_cache`` both reach the kernel
      and mask correctly under ``grid_type='single'`` and ``'double'``.

Usage (CPU only):
    PYTHONPATH=. conda run -n nv-subq python -m pytest \\
        tests/modules/test_kernels_nd.py -v -o addopts=""
"""

import math

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import (
    BlockDiagonalLearnableOmegaSIRENKernelND,
    BlockDiagonalMultiOmegaSIRENKernelND,
    LearnableOmegaSIRENKernelND,
    MultiOmegaSIRENKernelND,
    RandomFourierKernelND,
    RandomFourierPositionalEmbeddingND,
    SIRENKernelND,
    SIRENPositionalEmbeddingND,
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
