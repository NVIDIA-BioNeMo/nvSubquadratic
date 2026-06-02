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

"""Tests for GaussianModulationND, ExponentialModulationND, and _std_from_attenuation.

Covers:
    - _std_from_attenuation helper: forward/inverse accuracy, edge cases
    - GaussianModulationND: derived bounds, init_extent, hardcoded 0.1
      attenuation at init_extent, shapes, clamping, parametrizations,
      validation errors
    - CKConvND grid_size injection for grid_type="single" and "double"
    - ExponentialModulationND basic sanity

Usage (CPU only, no GPU needed):
    PYTHONPATH=. conda run -n nv-subq python -m pytest tests/modules/test_masks_nd.py -v -o addopts=""
"""

import math

import pytest
import torch

from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import (
    ExponentialModulationND,
    GaussianModulationND,
    _std_from_attenuation,
)


# ---------------------------------------------------------------------------
# _std_from_attenuation
# ---------------------------------------------------------------------------


class TestStdFromAttenuation:
    """Unit tests for the _std_from_attenuation helper."""

    def test_inverse_accuracy_1d(self):
        """Recovered std reproduces the target attenuation in 1D."""
        for attn in [0.01, 0.05, 0.1, 0.5, 0.9, 0.99]:
            for pos in [0.01, 0.1, 0.5, 1.0, 2.0]:
                std = _std_from_attenuation(attn, pos, data_dim=1)
                recovered = math.exp(-0.5 * (pos / std) ** 2)
                assert abs(recovered - attn) < 1e-12, f"1D: attn={attn}, pos={pos} -> std={std}, recovered={recovered}"

    def test_inverse_accuracy_2d(self):
        """Recovered std reproduces the target attenuation in 2D (corner point)."""
        for attn in [0.01, 0.1, 0.5, 0.9]:
            for pos in [0.1, 0.5, 1.0]:
                std = _std_from_attenuation(attn, pos, data_dim=2)
                recovered = math.exp(-0.5 * 2 * (pos / std) ** 2)
                assert abs(recovered - attn) < 1e-12, f"2D: attn={attn}, pos={pos} -> std={std}, recovered={recovered}"

    def test_higher_attenuation_gives_larger_std(self):
        """A less aggressive decay (higher attenuation value) requires a wider Gaussian."""
        std_low = _std_from_attenuation(0.01, 1.0, 1)
        std_high = _std_from_attenuation(0.5, 1.0, 1)
        assert std_high > std_low

    def test_rejects_invalid_attenuation(self):
        with pytest.raises(AssertionError):
            _std_from_attenuation(0.0, 1.0, 1)
        with pytest.raises(AssertionError):
            _std_from_attenuation(1.0, 1.0, 1)
        with pytest.raises(AssertionError):
            _std_from_attenuation(-0.1, 1.0, 1)

    def test_rejects_invalid_position(self):
        with pytest.raises(AssertionError):
            _std_from_attenuation(0.1, 0.0, 1)
        with pytest.raises(AssertionError):
            _std_from_attenuation(0.1, -1.0, 1)


# ---------------------------------------------------------------------------
# GaussianModulationND
# ---------------------------------------------------------------------------


GRID_SIZE_31 = 31
MIN_STEP_31 = 2.0 / (GRID_SIZE_31 - 1)  # ≈ 0.0667


class TestGaussianModulationND:
    """GaussianModulationND constructed with attenuation-based parameters."""

    def _make_mask(self, min_attn=0.1, max_attn=0.05, init_extent=1.0, num_ch=64, parametrization="direct"):
        return GaussianModulationND(
            data_dim=2,
            num_channels=num_ch,
            min_attenuation_at_step=min_attn,
            max_attenuation_at_limit=max_attn,
            init_extent=init_extent,
            grid_size=GRID_SIZE_31,
            parametrization=parametrization,
        )

    def _make_grid(self, size=GRID_SIZE_31):
        lin = torch.linspace(-1, 1, size)
        return torch.stack(torch.meshgrid(lin, lin, indexing="ij"), dim=-1).unsqueeze(0)

    # -- shapes & basic properties --------------------------------------

    def test_param_shape(self):
        mask = self._make_mask(num_ch=16)
        assert mask.std_param.shape == (2, 16)

    def test_forward_shape(self):
        mask = self._make_mask(num_ch=16)
        grid = self._make_grid()
        x = torch.ones(1, GRID_SIZE_31, GRID_SIZE_31, 16)
        out = mask(grid, x)
        assert out.shape == (1, GRID_SIZE_31, GRID_SIZE_31, 16)

    def test_center_value_is_one(self):
        """Mask at the origin should be 1.0 for all channels."""
        mask = self._make_mask(num_ch=16)
        grid = self._make_grid()
        x = torch.ones(1, GRID_SIZE_31, GRID_SIZE_31, 16)
        out = mask(grid, x)
        center = GRID_SIZE_31 // 2
        torch.testing.assert_close(out[0, center, center, :], torch.ones(16), atol=1e-6, rtol=0)

    def test_decay_away_from_center(self):
        """Mask values should strictly decrease away from center along an axis
        (for the widest channel, which has enough spread to avoid underflow)."""
        mask = self._make_mask(num_ch=16)
        grid = self._make_grid()
        x = torch.ones(1, GRID_SIZE_31, GRID_SIZE_31, 16)
        out = mask(grid, x)
        c = GRID_SIZE_31 // 2
        widest_ch = 15
        center_row = out[0, c, :, widest_ch]
        for i in range(1, c):
            assert center_row[c + i].item() <= center_row[c + i - 1].item()
            assert center_row[c - i].item() <= center_row[c - i + 1].item()

    def test_no_weight_decay_flag(self):
        mask = self._make_mask(num_ch=4)
        for p in mask.parameters():
            assert hasattr(p, "_no_weight_decay") and p._no_weight_decay

    # -- derived bounds ------------------------------------------------

    def test_min_std_from_attenuation(self):
        """min_std equals _std_from_attenuation(min_attn, min_step, 1)."""
        mask = self._make_mask(min_attn=0.1)
        expected = _std_from_attenuation(0.1, MIN_STEP_31, 1)
        assert mask.min_std == pytest.approx(expected, rel=1e-6)

    def test_max_std_from_attenuation(self):
        """max_std equals _std_from_attenuation(max_attn, 1.0, 1)."""
        mask = self._make_mask(max_attn=0.05)
        expected = _std_from_attenuation(0.05, 1.0, 1)
        assert mask.max_std == pytest.approx(expected, rel=1e-6)

    def test_init_std_low_equals_min_std(self):
        """Narrowest channel starts at exactly min_std."""
        mask = self._make_mask()
        assert mask.std_param.data.min().item() == pytest.approx(mask.min_std, rel=1e-4)

    # -- init_extent and the hardcoded 0.1 attenuation -----------------

    def test_init_extent_attenuation_is_01_when_unclamped(self):
        """When the (extent · 0.4724) high end fits inside [min_std, max_std],
        the widest initial channel has the expected 1D mask value of 0.1 at
        position ``extent``.

        ``init_extent`` multiplicatively scales both ends of the per-axis
        logspace ramp.  As long as the high end isn't clipped by the
        ``max_std`` clamp, the relationship
        ``init_std_high = _std_from_attenuation(0.1, extent, 1)`` holds.
        """
        # _make_mask uses min_attn=0.1, max_attn=0.05, grid_size=31 →
        # max_std ≈ 0.4087, init_std_high_unit ≈ 0.4724.  Pick extents
        # small enough that ``extent · 0.4724 < max_std``.
        max_std = _std_from_attenuation(0.05, 1.0, 1)
        init_std_high_unit = _std_from_attenuation(0.1, 1.0, 1)
        for extent in [0.25, 0.5, 0.75]:
            assert extent * init_std_high_unit < max_std, "test setup: extent must keep high end unclamped"
            mask = self._make_mask(init_extent=extent, num_ch=256)
            init_std_high = mask.std_param.data.max().item()
            expected_std = _std_from_attenuation(0.1, extent, 1)
            assert init_std_high == pytest.approx(expected_std, rel=1e-3)

    def test_init_extent_high_end_is_clamped_to_max_std(self):
        """When ``extent · init_std_high_unit`` exceeds ``max_std``, the high
        end of the ramp is clamped at ``max_std``."""
        mask = self._make_mask(init_extent=1.0, num_ch=256)
        # extent=1.0 with the test config has init_std_high_unit > max_std,
        # so the high end must clip exactly at max_std.
        init_std_high_unit = _std_from_attenuation(0.1, 1.0, 1)
        assert init_std_high_unit > mask.max_std
        init_std_high = mask.std_param.data.max().item()
        assert init_std_high == pytest.approx(mask.max_std, rel=1e-6)

    def test_init_extent_mask_value_matches_1d(self):
        """Evaluate the mask at (init_extent, 0) — for an unclamped ``extent``
        the single-axis value should be close to 0.1."""
        extent = 0.5  # unclamped at default test config
        mask = self._make_mask(init_extent=extent, num_ch=64)
        grid = self._make_grid()
        x = torch.ones(1, GRID_SIZE_31, GRID_SIZE_31, 64)
        out = mask(grid, x)

        center = GRID_SIZE_31 // 2
        xs = grid[0, :, center, 0]
        idx = (xs - extent).abs().argmin().item()
        widest_ch = 63  # last channel = widest

        mask_val = out[0, idx, center, widest_ch].item()
        assert mask_val < 0.15, f"Expected near 0.1 at extent={extent}, got {mask_val}"
        assert mask_val > 0.01, f"Unexpectedly low: {mask_val}"

    def test_smaller_extent_gives_smaller_init_std_high(self):
        """More local init_extent should give a smaller init_std_high (low end too)."""
        mask_global = self._make_mask(init_extent=1.0)
        mask_local = self._make_mask(init_extent=0.25)
        assert mask_local.std_param.data.max().item() < mask_global.std_param.data.max().item()
        # And the low end also moves: with multiplicative scaling, a smaller
        # extent pushes the bottom of the ramp below the reference min_std.
        assert mask_local.std_param.data.min().item() < mask_global.std_param.data.min().item() + 1e-12

    def test_extent_above_one_lifts_low_end_of_ramp(self):
        """``init_extent > 1`` must lift the **bottom** of the ramp above the
        reference ``min_std`` — otherwise short anisotropic axes would still
        have unusably-narrow channels at init.  This is the regression test
        for the multiplicative semantic."""
        mask_ref = self._make_mask(init_extent=1.0, num_ch=256)
        mask_wide = self._make_mask(init_extent=4.0, num_ch=256)
        ref_low = mask_ref.std_param.data.min().item()
        wide_low = mask_wide.std_param.data.min().item()
        # extent=4 should push the low end up by ~4x (modulo clamping).
        # At default test config min_std ≈ 0.0314, max_std ≈ 0.4087, so
        # 4*min_std ≈ 0.126 fits comfortably inside the band.
        assert wide_low > ref_low * 3.5
        assert wide_low <= mask_wide.max_std + 1e-6

    def test_extent_saturates_entire_ramp_at_max_std(self):
        """Sufficiently-large ``init_extent`` collapses the per-axis ramp to a
        constant ``max_std`` (axis effectively unmasked at init)."""
        mask = self._make_mask(init_extent=1e6, num_ch=64)
        # Both ends saturate at max_std.
        assert mask.std_param.data.min().item() == pytest.approx(mask.max_std, rel=1e-6)
        assert mask.std_param.data.max().item() == pytest.approx(mask.max_std, rel=1e-6)

    # -- clamping -------------------------------------------------------

    def test_clamping_to_max_std(self):
        """After forward, no channel exceeds max_std."""
        mask = self._make_mask(init_extent=1.0)
        grid = self._make_grid()
        x = torch.ones(1, GRID_SIZE_31, GRID_SIZE_31, 64)
        _ = mask(grid, x)  # triggers clamp hook
        assert mask.std_param.data.max().item() <= mask.max_std + 1e-7

    def test_clamping_to_min_std(self):
        """After forward, no channel goes below min_std."""
        mask = self._make_mask()
        grid = self._make_grid()
        x = torch.ones(1, GRID_SIZE_31, GRID_SIZE_31, 64)
        _ = mask(grid, x)
        assert mask.std_param.data.min().item() >= mask.min_std - 1e-7

    # -- attenuation values on grid -------------------------------------

    def test_narrowest_channel_attenuation_at_step(self):
        """The narrowest channel's 1D mask value at one step from center is
        close to min_attenuation_at_step."""
        min_attn = 0.1
        mask = self._make_mask(min_attn=min_attn)
        grid = self._make_grid()
        x = torch.ones(1, GRID_SIZE_31, GRID_SIZE_31, 64)
        out = mask(grid, x)

        center = GRID_SIZE_31 // 2
        # one step along axis: (center, center+1) — y-axis step while x=0
        step_val = out[0, center, center + 1, 0].item()
        # 1D attenuation at step for the narrowest channel
        expected_1d = math.exp(-0.5 * (MIN_STEP_31 / mask.min_std) ** 2)
        assert step_val == pytest.approx(expected_1d, rel=1e-3)

    def test_widest_possible_channel_attenuation_at_boundary(self):
        """A channel at max_std has 1D mask value == max_attenuation_at_limit
        at the grid boundary (position 1)."""
        max_attn = 0.05
        mask = self._make_mask(max_attn=max_attn, init_extent=1.0)
        max_std = mask.max_std
        expected_1d = math.exp(-0.5 * (1.0 / max_std) ** 2)
        assert expected_1d == pytest.approx(max_attn, rel=1e-6)

    # -- parametrizations -----------------------------------------------

    @pytest.mark.parametrize("parametrization", ["log", "softplus", "direct"])
    def test_parametrizations_produce_valid_output(self, parametrization):
        """All parametrizations produce mask values in [0, 1]."""
        mask = self._make_mask(parametrization=parametrization, num_ch=8)
        grid = self._make_grid()
        x = torch.ones(1, GRID_SIZE_31, GRID_SIZE_31, 8)
        out = mask(grid, x)
        assert (out >= 0).all()
        assert (out <= 1.0 + 1e-6).all()
        # Center should still be exactly 1
        c = GRID_SIZE_31 // 2
        torch.testing.assert_close(out[0, c, c, :], torch.ones(8), atol=1e-6, rtol=0)

    # -- defaults --------------------------------------------------------

    def test_defaults_work(self):
        """Constructing with just data_dim, num_channels, grid_size uses defaults."""
        mask = GaussianModulationND(data_dim=2, num_channels=8, grid_size=31)
        expected_min_std = _std_from_attenuation(0.1, MIN_STEP_31, 1)
        expected_max_std = _std_from_attenuation(0.95, 1.0, 1)
        assert mask.min_std == pytest.approx(expected_min_std, rel=1e-6)
        assert mask.max_std == pytest.approx(expected_max_std, rel=1e-6)

    # -- validation errors -----------------------------------------------

    def test_missing_grid_size_raises(self):
        """grid_size is a required positional arg."""
        with pytest.raises(TypeError):
            GaussianModulationND(data_dim=2, num_channels=4)

    def test_invalid_init_extent_raises(self):
        # Must be strictly > 0 and finite. Values > 1 are now allowed
        # (they multiplicatively scale the ramp — see the multiplicative
        # semantic introduced for anisotropic kernel grids).
        with pytest.raises(ValueError, match="init_extent"):
            self._make_mask(init_extent=0.0)
        with pytest.raises(ValueError, match="init_extent"):
            self._make_mask(init_extent=-0.5)
        with pytest.raises(ValueError, match="init_extent"):
            self._make_mask(init_extent=float("inf"))
        with pytest.raises(ValueError, match="init_extent"):
            self._make_mask(init_extent=float("nan"))

    # -- per-axis init_extent -------------------------------------------

    def test_per_axis_init_extent_shape(self):
        """Passing a sequence of length data_dim is accepted and stored as tuple."""
        mask = GaussianModulationND(
            data_dim=3,
            num_channels=8,
            grid_size=31,
            init_extent=(1.0, 0.5, 0.25),
        )
        assert mask.init_extent == (1.0, 0.5, 0.25)
        assert mask.std_param.shape == (3, 8)

    def test_per_axis_init_extent_sets_per_axis_init_std_high(self):
        """Each axis ramps to its own init_std_high determined by its init_extent.

        Default ``GaussianModulationND`` uses ``max_attenuation_at_limit=0.95``
        which gives ``max_std ≈ 3.121`` — much larger than
        ``init_std_high_unit ≈ 0.4724`` — so the high ends here are NOT
        clamped and equal ``_std_from_attenuation(0.1, extent, 1)`` exactly.
        """
        extents = (1.0, 0.5, 0.25)
        mask = GaussianModulationND(
            data_dim=3,
            num_channels=64,
            grid_size=31,
            init_extent=extents,
        )
        for axis, extent in enumerate(extents):
            expected_high = _std_from_attenuation(0.1, extent, 1)
            assert expected_high < mask.max_std, "test setup: high end must fit inside the clamp band"
            actual_high = mask.std_param.data[axis].max().item()
            assert actual_high == pytest.approx(expected_high, rel=1e-3)

    def test_per_axis_init_extent_above_one_lifts_per_axis_low(self):
        """``init_extent[d] > 1`` lifts the *bottom* of axis-d's ramp above
        ``min_std`` (multiplicative-semantic regression)."""
        # Use the default mask config (min_attn=0.1, max_attn=0.95) so the
        # clamp band is wide enough that 4× scaling fits comfortably.
        mask = GaussianModulationND(
            data_dim=3,
            num_channels=64,
            grid_size=127,  # matches an 8x64x64 anisotropic-grid config
            init_extent=(4.0, 1.0, 1.0),
        )
        depth_low = mask.std_param.data[0].min().item()
        hw_low = mask.std_param.data[1].min().item()
        # H/W keep the reference low; depth is lifted by ~4x.
        assert hw_low == pytest.approx(mask.min_std, rel=1e-3)
        assert depth_low > 3.5 * mask.min_std

    def test_per_axis_init_extent_narrower_axis_narrower_widest_channel(self):
        """A smaller init_extent on one axis yields a narrower widest-channel
        std on that axis compared to a wider-extent axis."""
        mask = GaussianModulationND(
            data_dim=2,
            num_channels=64,
            grid_size=31,
            init_extent=(1.0, 0.25),
        )
        widest_axis0 = mask.std_param.data[0].max().item()
        widest_axis1 = mask.std_param.data[1].max().item()
        assert widest_axis1 < widest_axis0

    def test_scalar_init_extent_matches_homogeneous_sequence(self):
        """Scalar and equivalent sequence init produce identical parameters."""
        scalar = GaussianModulationND(data_dim=3, num_channels=16, grid_size=31, init_extent=0.5)
        seq = GaussianModulationND(data_dim=3, num_channels=16, grid_size=31, init_extent=(0.5, 0.5, 0.5))
        torch.testing.assert_close(scalar.std_param.data, seq.std_param.data, atol=0, rtol=0)
        assert scalar.init_extent == seq.init_extent == (0.5, 0.5, 0.5)

    def test_per_axis_init_extent_wrong_length_raises(self):
        with pytest.raises(ValueError, match="length data_dim=3"):
            GaussianModulationND(data_dim=3, num_channels=8, grid_size=31, init_extent=(1.0, 0.5))

    def test_per_axis_init_extent_out_of_range_raises(self):
        # Sequence values must each be strictly > 0 and finite. Values > 1
        # are explicitly allowed (multiplicative semantic) and tested above.
        with pytest.raises(ValueError, match="init_extent"):
            GaussianModulationND(data_dim=2, num_channels=8, grid_size=31, init_extent=(0.5, 0.0))
        with pytest.raises(ValueError, match="init_extent"):
            GaussianModulationND(data_dim=2, num_channels=8, grid_size=31, init_extent=(1.0, -2.0))
        with pytest.raises(ValueError, match="init_extent"):
            GaussianModulationND(data_dim=2, num_channels=8, grid_size=31, init_extent=(1.0, float("inf")))

    def test_per_axis_init_extent_wrong_type_raises(self):
        with pytest.raises(TypeError, match="init_extent"):
            GaussianModulationND(data_dim=2, num_channels=8, grid_size=31, init_extent="not-a-float")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CKConvND grid_size injection
# ---------------------------------------------------------------------------


class TestCKConvGridInjection:
    """CKConvND injects grid_size into GaussianModulationND mask_cfg."""

    def _make_ckconv(self, grid_type, mask_cfg, L_cache=32):
        from nvsubquadratic.modules.ckconv_nd import CKConvND  # heavy imports
        from nvsubquadratic.modules.kernels_nd import SIRENKernelND

        kernel_cfg = LazyConfig(SIRENKernelND)(
            data_dim=2,
            out_dim=32,
            mlp_hidden_dim=16,
            num_layers=2,
            embedding_dim=16,
            omega_0=10.0,
            L_cache=L_cache,
            use_bias=True,
        )
        return CKConvND(
            data_dim=2,
            hidden_dim=32,
            kernel_cfg=kernel_cfg,
            mask_cfg=mask_cfg,
            grid_type=grid_type,
            fft_padding="circular",
        )

    def test_single_grid_type_injects_correct_grid_size(self):
        """grid_type='single' with L_cache=32 -> effective_L=16 -> grid_size=31."""
        mask_cfg = LazyConfig(GaussianModulationND)(
            data_dim=2,
            num_channels=32,
            min_attenuation_at_step=0.1,
            max_attenuation_at_limit=0.05,
            parametrization="direct",
        )
        ck = self._make_ckconv("single", mask_cfg, L_cache=32)
        expected_grid_size = 2 * ((32 + 1) // 2) - 1  # = 31
        expected_min_step = 2.0 / (expected_grid_size - 1)
        assert ck.mask.min_std == pytest.approx(_std_from_attenuation(0.1, expected_min_step, 1), rel=1e-6)

    def test_double_grid_type_injects_correct_grid_size(self):
        """grid_type='double' with L_cache=32 -> effective_L=32 -> grid_size=63."""
        from nvsubquadratic.modules.ckconv_nd import CKConvND
        from nvsubquadratic.modules.kernels_nd import SIRENKernelND

        mask_cfg = LazyConfig(GaussianModulationND)(
            data_dim=2,
            num_channels=32,
            min_attenuation_at_step=0.1,
            max_attenuation_at_limit=0.05,
            parametrization="direct",
        )
        kernel_cfg = LazyConfig(SIRENKernelND)(
            data_dim=2,
            out_dim=32,
            mlp_hidden_dim=16,
            num_layers=2,
            embedding_dim=16,
            omega_0=10.0,
            L_cache=32,
            use_bias=True,
        )
        ck = CKConvND(
            data_dim=2,
            hidden_dim=32,
            kernel_cfg=kernel_cfg,
            mask_cfg=mask_cfg,
            grid_type="double",
            fft_padding="zero",
        )
        expected_grid_size = 2 * 32 - 1  # = 63
        expected_min_step = 2.0 / (expected_grid_size - 1)
        assert ck.mask.min_std == pytest.approx(_std_from_attenuation(0.1, expected_min_step, 1), rel=1e-6)

    def test_identity_mask_no_injection_error(self):
        """torch.nn.Identity has no grid_size param — injection is skipped."""
        mask_cfg = LazyConfig(torch.nn.Identity)()
        ck = self._make_ckconv("single", mask_cfg)
        assert isinstance(ck.mask, torch.nn.Identity)

    def test_kernel_grid_spans_full_range(self):
        """After L_cache adjustment, the kernel grid should span [-1, 1]."""
        mask_cfg = LazyConfig(torch.nn.Identity)()
        ck = self._make_ckconv("single", mask_cfg, L_cache=32)
        spatial = [16, 16]
        _, grid = ck.kernel(spatial)
        # grid shape: [1, H, W, 2], coordinates should span [-1, 1]
        assert grid.min().item() == pytest.approx(-1.0, abs=0.01)
        assert grid.max().item() == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# ExponentialModulationND — basic sanity
# ---------------------------------------------------------------------------


class TestExponentialModulation:
    """Basic tests for ExponentialModulationND."""

    def test_output_shape(self):
        mask = ExponentialModulationND(data_dim=2, num_channels=8)
        grid = torch.linspace(-1, 1, 5)
        grid = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).unsqueeze(0)
        x = torch.ones(1, 5, 5, 8)
        out = mask(grid, x)
        assert out.shape == (1, 5, 5, 8)

    def test_center_value_is_one(self):
        mask = ExponentialModulationND(data_dim=2, num_channels=8)
        grid = torch.linspace(-1, 1, 5)
        grid = torch.stack(torch.meshgrid(grid, grid, indexing="ij"), dim=-1).unsqueeze(0)
        x = torch.ones(1, 5, 5, 8)
        out = mask(grid, x)
        center = out[0, 2, 2, :]
        torch.testing.assert_close(center, torch.ones(8), atol=1e-6, rtol=0)

    def test_no_weight_decay_flag(self):
        mask = ExponentialModulationND(data_dim=1, num_channels=4)
        for p in mask.parameters():
            assert hasattr(p, "_no_weight_decay") and p._no_weight_decay
