# TODO: Add license header here


"""Tests for the multi-ω₀ SIREN kernels and the block-aligned Gaussian mask.

Covers:
    - MultiOmegaSIRENPositionalEmbeddingND: per-row init statistics, scalar
      equivalence, validation errors.
    - MultiOmegaSIRENKernelND: shape, gradients, schedule surfaced on module,
      validation errors.
    - BlockDiagonalMultiOmegaSIRENKernelND: block schedule, block-mask effect
      on hidden + output linears (off_block_scale=0 strict / =1 no-op /
      intermediate scaling), linear vs log schedule, explicit
      omega_0_per_block override, divisibility errors.
    - BlockAlignedGaussianModulationND: std_param reversal, alignment with
      block-structured SIRENs (widest Gaussian on the lowest-ω₀ block).

Usage (CPU only):
    PYTHONPATH=. conda run -n nv-subq python -m pytest \\
        tests/modules/test_multi_omega_siren.py -v -o addopts=""
"""

import math
from typing import ClassVar

import pytest
import torch

from nvsubquadratic.modules.kernels_nd import (
    BlockDiagonalMultiOmegaSIRENKernelND,
    MultiOmegaSIRENKernelND,
    MultiOmegaSIRENPositionalEmbeddingND,
    SIRENPositionalEmbeddingND,
    _build_omega_0_per_block,
)
from nvsubquadratic.modules.masks_nd import (
    BlockAlignedGaussianModulationND,
    GaussianModulationND,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_abs_per_row(weight: torch.Tensor) -> torch.Tensor:
    """``weight`` has shape [out, in]; returns a [out]-shaped tensor of per-row mean(|W|)."""
    return weight.detach().abs().mean(dim=1)


# ---------------------------------------------------------------------------
# _build_omega_0_per_block
# ---------------------------------------------------------------------------


class TestBuildOmega0PerBlock:
    def test_linear_schedule(self):
        vals = _build_omega_0_per_block(num_blocks=5, omega_0_min=1.0, omega_0_max=5.0, schedule="linear")
        expected = torch.linspace(1.0, 5.0, 5, dtype=torch.float64)
        torch.testing.assert_close(vals, expected)

    def test_log_schedule(self):
        vals = _build_omega_0_per_block(num_blocks=4, omega_0_min=1.0, omega_0_max=1000.0, schedule="log")
        expected = torch.logspace(0.0, 3.0, 4, dtype=torch.float64)
        torch.testing.assert_close(vals, expected)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="schedule"):
            _build_omega_0_per_block(num_blocks=4, omega_0_min=1.0, omega_0_max=5.0, schedule="cosine")

    def test_nonpositive_endpoint_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _build_omega_0_per_block(num_blocks=4, omega_0_min=0.0, omega_0_max=5.0, schedule="linear")

    def test_inverted_range_raises(self):
        with pytest.raises(ValueError, match=">="):
            _build_omega_0_per_block(num_blocks=4, omega_0_min=5.0, omega_0_max=1.0, schedule="linear")

    def test_single_block(self):
        vals = _build_omega_0_per_block(num_blocks=1, omega_0_min=3.0, omega_0_max=3.0, schedule="linear")
        torch.testing.assert_close(vals, torch.tensor([3.0], dtype=torch.float64))


# ---------------------------------------------------------------------------
# MultiOmegaSIRENPositionalEmbeddingND
# ---------------------------------------------------------------------------


class TestMultiOmegaPositionalEmbedding:
    def _make(self, *, data_dim=2, embedding_dim=16, L_cache=10, schedule=None):
        if schedule is None:
            schedule = torch.linspace(1.0, 10.0, embedding_dim)
        return MultiOmegaSIRENPositionalEmbeddingND(
            data_dim=data_dim,
            embedding_dim=embedding_dim,
            L_cache=L_cache,
            omega_0_per_row=schedule,
            use_bias=True,
        )

    def test_shapes(self):
        emb = self._make(data_dim=2, embedding_dim=8, L_cache=10)
        out, grid = emb(seq_lens=(10, 10))
        # Grid cache has extent (2*L-1) per dim.
        assert out.shape == (1, 19, 19, 8)
        assert grid.shape == (1, 19, 19, 2)

    def test_wrong_schedule_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            MultiOmegaSIRENPositionalEmbeddingND(
                data_dim=2,
                embedding_dim=8,
                L_cache=10,
                omega_0_per_row=[1.0, 2.0, 3.0],  # too short
            )

    def test_nonpositive_omega_raises(self):
        with pytest.raises(ValueError, match="positive"):
            MultiOmegaSIRENPositionalEmbeddingND(
                data_dim=2,
                embedding_dim=4,
                L_cache=10,
                omega_0_per_row=[1.0, 2.0, 0.0, 3.0],
            )

    def test_per_row_init_statistics(self):
        """Per-row mean(|W|) must track 2π·ω₀_k / (2·d) = π·ω₀_k / d.

        A single seed is noisy in low dims (``data_dim``=2 ⇒ only 2 draws per
        row), so we average over many seeds to get a statistically-tight
        estimate.
        """
        data_dim = 2
        embedding_dim = 16
        L_cache = 10
        schedule = torch.linspace(1.0, 10.0, embedding_dim)
        n_seeds = 400

        acc = torch.zeros(embedding_dim)
        for s in range(n_seeds):
            torch.manual_seed(1234 + s)
            emb = MultiOmegaSIRENPositionalEmbeddingND(
                data_dim=data_dim,
                embedding_dim=embedding_dim,
                L_cache=L_cache,
                omega_0_per_row=schedule,
            )
            acc += _mean_abs_per_row(emb.linear.weight)
        observed = acc / n_seeds

        expected = math.pi * schedule / float(data_dim)
        # Relative error averaged across rows; per-row fluctuations can be
        # somewhat larger, but the mean trend must be tight.
        rel_err = (observed - expected).abs() / expected
        assert rel_err.mean().item() < 0.05, (
            f"Per-row mean(|W|) deviates from π·ω₀/d: observed={observed.tolist()}, expected={expected.tolist()}"
        )

    def test_constant_schedule_matches_scalar_baseline_in_expectation(self):
        """With a constant ω₀ schedule, the per-row class produces weights that
        are *statistically* equivalent to the scalar-ω₀ baseline.

        They are not byte-identical (different RNG consumption), so we compare
        means over many seeds.
        """
        data_dim = 2
        embedding_dim = 16
        L_cache = 10
        omega = 5.0
        n_seeds = 200

        mean_multi = torch.zeros(embedding_dim)
        mean_base = torch.zeros(embedding_dim)
        for s in range(n_seeds):
            torch.manual_seed(7 + s)
            multi = MultiOmegaSIRENPositionalEmbeddingND(
                data_dim=data_dim,
                embedding_dim=embedding_dim,
                L_cache=L_cache,
                omega_0_per_row=[omega] * embedding_dim,
            )
            torch.manual_seed(7 + s)
            base = SIRENPositionalEmbeddingND(
                data_dim=data_dim,
                embedding_dim=embedding_dim,
                L_cache=L_cache,
                omega_0=omega,
            )
            mean_multi += _mean_abs_per_row(multi.linear.weight)
            mean_base += _mean_abs_per_row(base.linear.weight)
        mean_multi /= n_seeds
        mean_base /= n_seeds

        rel_err = (mean_multi - mean_base).abs() / mean_base
        assert rel_err.mean().item() < 0.05

    def test_no_weight_decay_flag(self):
        emb = self._make()
        for p in emb.parameters():
            assert hasattr(p, "_no_weight_decay") and p._no_weight_decay

    def test_omega_0_attribute(self):
        schedule = [1.0, 2.0, 3.0, 4.0]
        emb = MultiOmegaSIRENPositionalEmbeddingND(
            data_dim=2,
            embedding_dim=4,
            L_cache=10,
            omega_0_per_row=schedule,
        )
        assert emb.omega_0 == pytest.approx(sum(schedule) / len(schedule))
        torch.testing.assert_close(emb.omega_0_per_row, torch.tensor(schedule, dtype=torch.float32))


# ---------------------------------------------------------------------------
# MultiOmegaSIRENKernelND
# ---------------------------------------------------------------------------


class TestMultiOmegaSIRENKernel:
    def _make(
        self, *, out_dim=16, embedding_dim=16, mlp_hidden_dim=32, num_layers=3, L_cache=10, schedule=None, data_dim=2
    ):
        if schedule is None:
            schedule = torch.linspace(1.0, 12.0, embedding_dim)
        return MultiOmegaSIRENKernelND(
            out_dim=out_dim,
            data_dim=data_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            embedding_dim=embedding_dim,
            omega_0_per_row=schedule,
            L_cache=L_cache,
            use_bias=True,
            hidden_omega_0=1.0,
        )

    def test_shape(self):
        k = self._make(out_dim=8, embedding_dim=16, mlp_hidden_dim=32, num_layers=3, L_cache=10, data_dim=2)
        kernel, grid = k(seq_lens=(10, 10))
        assert kernel.shape == (1, 19, 19, 8)
        assert grid.shape == (1, 19, 19, 2)

    def test_gradients_flow(self):
        k = self._make()
        kernel, _ = k(seq_lens=(10, 10))
        kernel.sum().backward()
        for name, p in k.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"no grad for {name}"

    def test_positional_embedding_is_multi(self):
        k = self._make()
        assert isinstance(k.positional_embedding, MultiOmegaSIRENPositionalEmbeddingND)

    def test_schedule_surfaced(self):
        schedule = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        k = MultiOmegaSIRENKernelND(
            out_dim=8,
            data_dim=2,
            mlp_hidden_dim=16,
            num_layers=2,
            embedding_dim=8,
            omega_0_per_row=schedule,
            L_cache=10,
            use_bias=True,
        )
        torch.testing.assert_close(k.omega_0_per_row, torch.tensor(schedule, dtype=torch.float32))
        assert k.omega_0 == pytest.approx(sum(schedule) / len(schedule))

    def test_wrong_schedule_length_raises(self):
        with pytest.raises(ValueError, match="length"):
            MultiOmegaSIRENKernelND(
                out_dim=8,
                data_dim=2,
                mlp_hidden_dim=16,
                num_layers=2,
                embedding_dim=8,
                omega_0_per_row=[1.0, 2.0, 3.0],  # too short
                L_cache=10,
                use_bias=True,
            )


# ---------------------------------------------------------------------------
# BlockDiagonalMultiOmegaSIRENKernelND
# ---------------------------------------------------------------------------


def _block_partition(weight: torch.Tensor, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``weight`` [out, in] into on-diagonal and off-diagonal block entries.

    Returns two 1-D tensors (on_block_values, off_block_values).
    """
    out_dim, in_dim = weight.shape
    assert out_dim % num_blocks == 0 and in_dim % num_blocks == 0
    rpb = out_dim // num_blocks
    cpb = in_dim // num_blocks
    on_vals, off_vals = [], []
    for i in range(num_blocks):
        for j in range(num_blocks):
            block = weight[i * rpb : (i + 1) * rpb, j * cpb : (j + 1) * cpb]
            (on_vals if i == j else off_vals).append(block.flatten())
    return torch.cat(on_vals), torch.cat(off_vals)


class TestBlockDiagonalMultiOmegaSIRENKernel:
    DEFAULTS: ClassVar[dict] = {
        "out_dim": 32,
        "data_dim": 2,
        "mlp_hidden_dim": 32,
        "num_layers": 3,
        "embedding_dim": 32,
        "L_cache": 10,
        "use_bias": True,
    }

    def _make(self, **overrides):
        cfg = {**self.DEFAULTS, **overrides}
        return BlockDiagonalMultiOmegaSIRENKernelND(**cfg)

    def test_production_defaults_instantiate(self):
        k = self._make()
        assert k.num_blocks == 8
        assert k.off_block_scale == pytest.approx(0.1)
        expected = torch.linspace(1.0, 12.0, 8, dtype=torch.float32)
        torch.testing.assert_close(k.omega_0_per_block, expected)

    def test_shape_and_gradients(self):
        k = self._make()
        kernel, _ = k(seq_lens=(10, 10))
        assert kernel.shape == (1, 19, 19, 32)
        kernel.sum().backward()
        for name, p in k.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"no grad for {name}"

    def test_strict_block_diagonal_off_zero(self):
        """With ``off_block_scale=0.0`` every off-block entry of every hidden
        + output linear is exactly zero."""
        k = self._make(off_block_scale=0.0)
        for linear in k.hidden_linears:
            _, off = _block_partition(linear.weight.data, k.num_blocks)
            assert off.abs().max().item() == 0.0
        _, off = _block_partition(k.out_linear.weight.data, k.num_blocks)
        assert off.abs().max().item() == 0.0

    def test_off_block_scaling_ratio(self):
        """With intermediate ``off_block_scale``, off/on mean(|W|) across many
        seeds should equal ``off_block_scale`` (within statistical error)."""
        scale = 0.25
        num_blocks = 8
        n_seeds = 30

        on_totals = []
        off_totals = []
        for s in range(n_seeds):
            torch.manual_seed(4321 + s)
            k = self._make(off_block_scale=scale)
            # Aggregate all hidden linears and the output linear.
            on_all, off_all = [], []
            for linear in k.hidden_linears:
                on, off = _block_partition(linear.weight.data, num_blocks)
                on_all.append(on)
                off_all.append(off)
            on, off = _block_partition(k.out_linear.weight.data, num_blocks)
            on_all.append(on)
            off_all.append(off)
            on_totals.append(torch.cat(on_all).abs().mean().item())
            off_totals.append(torch.cat(off_all).abs().mean().item())
        ratio = sum(off_totals) / sum(on_totals)
        assert abs(ratio - scale) / scale < 0.10, f"off/on mean-|W| ratio {ratio:.3f} should be ~{scale}"

    def test_off_block_scale_one_equals_parent(self):
        """With ``off_block_scale=1.0`` the mask is all ones and the kernel
        reduces to the parent :class:`MultiOmegaSIRENKernelND` with the same
        expanded schedule."""
        num_blocks = 4
        seed = 99
        torch.manual_seed(seed)
        bd = BlockDiagonalMultiOmegaSIRENKernelND(
            out_dim=16,
            data_dim=2,
            mlp_hidden_dim=16,
            num_layers=3,
            embedding_dim=16,
            L_cache=8,
            use_bias=True,
            num_blocks=num_blocks,
            omega_0_min=1.0,
            omega_0_max=4.0,
            schedule="linear",
            off_block_scale=1.0,
        )
        omega_per_block = torch.linspace(1.0, 4.0, num_blocks)
        omega_per_row = omega_per_block.repeat_interleave(16 // num_blocks)

        torch.manual_seed(seed)
        ref = MultiOmegaSIRENKernelND(
            out_dim=16,
            data_dim=2,
            mlp_hidden_dim=16,
            num_layers=3,
            embedding_dim=16,
            omega_0_per_row=omega_per_row,
            L_cache=8,
            use_bias=True,
        )

        # With mask == 1, the hidden + out weights must be bit-identical.
        for l_bd, l_ref in zip(bd.hidden_linears, ref.hidden_linears):
            torch.testing.assert_close(l_bd.weight.data, l_ref.weight.data)
        torch.testing.assert_close(bd.out_linear.weight.data, ref.out_linear.weight.data)

    def test_linear_vs_log_schedule(self):
        lin = self._make(schedule="linear", omega_0_min=1.0, omega_0_max=16.0)
        log = self._make(schedule="log", omega_0_min=1.0, omega_0_max=16.0)
        torch.testing.assert_close(
            lin.omega_0_per_block,
            torch.linspace(1.0, 16.0, 8, dtype=torch.float32),
        )
        torch.testing.assert_close(
            log.omega_0_per_block,
            torch.logspace(0.0, math.log10(16.0), 8, dtype=torch.float32),
        )

    def test_explicit_omega_0_per_block_override(self):
        custom = [1.0, 2.5, 6.0, 12.0]
        k = self._make(
            num_blocks=4,
            omega_0_per_block=custom,
            omega_0_min=999.0,  # must be ignored when override is given
            omega_0_max=9999.0,
            schedule="linear",
        )
        torch.testing.assert_close(k.omega_0_per_block, torch.tensor(custom, dtype=torch.float32))

    def test_override_wrong_length_raises(self):
        with pytest.raises(ValueError, match="omega_0_per_block length"):
            self._make(num_blocks=4, omega_0_per_block=[1.0, 2.0])

    def test_non_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            self._make(num_blocks=7)  # embedding/mlp/out = 32 not divisible by 7

    def test_per_row_omega_matches_per_block(self):
        """Every row inside a block must see the same ω₀."""
        k = self._make(num_blocks=4, omega_0_min=1.0, omega_0_max=4.0, schedule="linear")
        rpb = 32 // 4
        for b in range(4):
            block_rows = k.omega_0_per_row[b * rpb : (b + 1) * rpb]
            assert torch.allclose(block_rows, block_rows[0].expand_as(block_rows))
            assert block_rows[0].item() == pytest.approx(float(k.omega_0_per_block[b]))


# ---------------------------------------------------------------------------
# BlockAlignedGaussianModulationND
# ---------------------------------------------------------------------------


GRID_SIZE_31 = 31


class TestBlockAlignedGaussianModulation:
    def _make_aligned(self, **overrides):
        defaults = {
            "data_dim": 2,
            "num_channels": 32,
            "grid_size": GRID_SIZE_31,
            "parametrization": "direct",
        }
        return BlockAlignedGaussianModulationND(**{**defaults, **overrides})

    def _make_baseline(self, **overrides):
        defaults = {
            "data_dim": 2,
            "num_channels": 32,
            "grid_size": GRID_SIZE_31,
            "parametrization": "direct",
        }
        return GaussianModulationND(**{**defaults, **overrides})

    def test_std_param_is_baseline_flipped(self):
        aligned = self._make_aligned()
        baseline = self._make_baseline()
        torch.testing.assert_close(
            aligned.std_param.data,
            baseline.std_param.data.flip(dims=[-1]),
        )

    def test_widest_channel_is_first(self):
        """Widest (largest-std) channel is index 0 in the aligned variant.

        In the baseline it's the last channel.
        """
        aligned = self._make_aligned()
        stds = aligned.std_param.data[0]  # [num_channels]
        assert stds[0].item() == stds.max().item()
        assert stds[-1].item() == stds.min().item()
        assert stds[0].item() > stds[-1].item()

    def test_forward_output_valid(self):
        """Mask values lie in [0, 1]; center is exactly 1 for every channel."""
        aligned = self._make_aligned(num_channels=16)
        lin = torch.linspace(-1, 1, GRID_SIZE_31)
        grid = torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).unsqueeze(0)
        x = torch.ones(1, GRID_SIZE_31, GRID_SIZE_31, 16)
        out = aligned(grid, x)
        assert out.shape == (1, GRID_SIZE_31, GRID_SIZE_31, 16)
        assert (out >= 0).all()
        assert (out <= 1.0 + 1e-6).all()
        c = GRID_SIZE_31 // 2
        torch.testing.assert_close(out[0, c, c, :], torch.ones(16), atol=1e-6, rtol=0)

    def test_clamp_bounds_inherit_from_baseline(self):
        aligned = self._make_aligned()
        baseline = self._make_baseline()
        assert aligned.min_std == pytest.approx(baseline.min_std)
        assert aligned.max_std == pytest.approx(baseline.max_std)

    def test_no_weight_decay_flag(self):
        aligned = self._make_aligned()
        for p in aligned.parameters():
            assert hasattr(p, "_no_weight_decay") and p._no_weight_decay

    @pytest.mark.parametrize("parametrization", ["log", "softplus", "direct"])
    def test_parametrizations(self, parametrization):
        aligned = self._make_aligned(parametrization=parametrization)
        baseline = self._make_baseline(parametrization=parametrization)
        torch.testing.assert_close(
            aligned.std_param.data,
            baseline.std_param.data.flip(dims=[-1]),
        )
