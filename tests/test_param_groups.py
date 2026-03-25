"""Tests for _build_param_groups and _get_layer_index (LLRD + WD grouping)."""

import warnings

import pytest
import torch
import torch.nn as nn

from experiments.lightning_wrappers.base_lightning_wrapper import (
    _build_param_groups,
    _get_layer_index,
)


# ---------------------------------------------------------------------------
# Helpers: minimal model that mirrors ViT5 param naming under "network.*"
# ---------------------------------------------------------------------------


class _FakeBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)


class _FakeViT(nn.Module):
    """Mimics the ViT5ClassificationNet naming convention."""

    def __init__(self, dim: int = 8, num_blocks: int = 4):
        super().__init__()
        self.patch_embed = nn.Linear(dim, dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.cls_token._no_weight_decay = True

        self.pos_embed = nn.Parameter(torch.zeros(1, 4, dim))
        self.pos_embed._no_weight_decay = True

        self.blocks = nn.ModuleList([_FakeBlock(dim) for _ in range(num_blocks)])

        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, 10)


class _FakeWrapper(nn.Module):
    """Wraps _FakeViT under `self.network` to match lightning wrapper naming."""

    def __init__(self, **kwargs):
        super().__init__()
        self.network = _FakeViT(**kwargs)


# ---------------------------------------------------------------------------
# _get_layer_index
# ---------------------------------------------------------------------------


class TestGetLayerIndex:
    def test_embedding_params(self):
        for prefix in ("patch_embed", "cls_token", "pos_embed", "reg_token"):
            assert _get_layer_index(f"network.{prefix}.weight", num_blocks=4) == 0

    def test_block_params(self):
        assert _get_layer_index("network.blocks.0.norm.weight", num_blocks=4) == 1
        assert _get_layer_index("network.blocks.3.linear.bias", num_blocks=4) == 4

    def test_head_params(self):
        for prefix in ("out_norm", "out_proj"):
            assert _get_layer_index(f"network.{prefix}.weight", num_blocks=4) == 5

    def test_unknown_params_map_to_head(self):
        assert _get_layer_index("network.some_new_module.weight", num_blocks=4) == 5


# ---------------------------------------------------------------------------
# _build_param_groups — weight decay
# ---------------------------------------------------------------------------


class TestBuildParamGroupsWD:
    def test_default_wd(self):
        model = _FakeWrapper(num_blocks=2)
        groups = _build_param_groups(model, default_weight_decay=0.05)

        wd_values = {g["weight_decay"] for g in groups}
        assert 0.05 in wd_values, "default WD group should exist"
        assert 0.0 in wd_values, "no-WD group should exist (cls_token, pos_embed)"

    def test_no_weight_decay_attribute(self):
        model = _FakeWrapper(num_blocks=2)
        groups = _build_param_groups(model, default_weight_decay=0.1)

        no_wd_params = []
        for g in groups:
            if g["weight_decay"] == 0.0:
                no_wd_params.extend(g["params"])

        no_wd_ids = {id(p) for p in no_wd_params}
        assert id(model.network.cls_token) in no_wd_ids
        assert id(model.network.pos_embed) in no_wd_ids

    def test_custom_weight_decay_attribute(self):
        model = _FakeWrapper(num_blocks=2)
        model.network.out_proj.weight._weight_decay = 0.01
        groups = _build_param_groups(model, default_weight_decay=0.05)

        wd_values = {g["weight_decay"] for g in groups}
        assert 0.01 in wd_values

    def test_all_params_assigned_once(self):
        model = _FakeWrapper(num_blocks=4)
        groups = _build_param_groups(model, default_weight_decay=0.05)

        total_in_groups = sum(len(g["params"]) for g in groups)
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert total_in_groups == total_trainable


# ---------------------------------------------------------------------------
# _build_param_groups — 1D parameter warnings
# ---------------------------------------------------------------------------


class TestBuildParamGroups1DWarning:
    def test_warns_for_unflagged_1d_param(self):
        """A 1D param without _no_weight_decay receiving WD > 0 triggers a warning."""
        model = _FakeWrapper(num_blocks=2)
        # patch_embed.bias is 1D and has no _no_weight_decay flag
        assert not getattr(model.network.patch_embed.bias, "_no_weight_decay", False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_param_groups(model, default_weight_decay=0.05)

        bias_warnings = [w for w in caught if "patch_embed.bias" in str(w.message)]
        assert len(bias_warnings) >= 1, "Expected warning for unflagged patch_embed.bias"

    def test_no_warning_for_flagged_1d_param(self):
        """A 1D param with _no_weight_decay=True should NOT trigger a warning."""
        model = _FakeWrapper(num_blocks=2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_param_groups(model, default_weight_decay=0.05)

        flagged_warnings = [w for w in caught if "cls_token" in str(w.message) or "pos_embed" in str(w.message)]
        assert len(flagged_warnings) == 0, "Should not warn for params with _no_weight_decay"

    def test_no_warning_when_default_wd_is_zero(self):
        """No warnings when default_weight_decay=0 (all params get WD=0)."""
        model = _FakeWrapper(num_blocks=2)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_param_groups(model, default_weight_decay=0.0)

        ndim_warnings = [w for w in caught if "ndim <= 1" in str(w.message)]
        assert len(ndim_warnings) == 0


# ---------------------------------------------------------------------------
# _build_param_groups — LLRD
# ---------------------------------------------------------------------------


class TestBuildParamGroupsLLRD:
    def test_raises_when_num_blocks_is_none(self):
        model = _FakeWrapper(num_blocks=2)
        with pytest.raises(ValueError, match="num_blocks must be set"):
            _build_param_groups(model, default_weight_decay=0.05, layer_decay=0.75, num_blocks=None)

    def test_no_error_when_both_none(self):
        model = _FakeWrapper(num_blocks=2)
        groups = _build_param_groups(model, default_weight_decay=0.05, layer_decay=None, num_blocks=None)
        assert len(groups) > 0
        assert all("lr_scale" not in g for g in groups)

    def test_lr_scale_present_with_llrd(self):
        model = _FakeWrapper(num_blocks=4)
        groups = _build_param_groups(model, default_weight_decay=0.05, layer_decay=0.75, num_blocks=4)
        assert all("lr_scale" in g for g in groups)

    def test_head_gets_highest_lr_scale(self):
        model = _FakeWrapper(num_blocks=4)
        groups = _build_param_groups(model, default_weight_decay=0.05, layer_decay=0.75, num_blocks=4)
        # Head layer index = num_blocks + 1 = 5; num_layers = 6
        # lr_scale = 0.75^(6-1-5) = 0.75^0 = 1.0
        lr_scales = [g["lr_scale"] for g in groups]
        assert max(lr_scales) == pytest.approx(1.0)

    def test_embedding_gets_lowest_lr_scale(self):
        num_blocks = 4
        decay = 0.75
        model = _FakeWrapper(num_blocks=num_blocks)
        groups = _build_param_groups(model, default_weight_decay=0.05, layer_decay=decay, num_blocks=num_blocks)

        # Embedding layer index = 0; num_layers = 6
        # lr_scale = 0.75^(6-1-0) = 0.75^5
        expected_min = decay ** (num_blocks + 2 - 1)
        lr_scales = [g["lr_scale"] for g in groups]
        assert min(lr_scales) == pytest.approx(expected_min)

    def test_lr_scales_monotonically_increase(self):
        model = _FakeWrapper(num_blocks=4)
        groups = _build_param_groups(model, default_weight_decay=0.0, layer_decay=0.75, num_blocks=4)
        # With wd=0 for all params, groups differ only by lr_scale
        lr_scales = sorted(g["lr_scale"] for g in groups)
        for i in range(len(lr_scales) - 1):
            assert lr_scales[i] <= lr_scales[i + 1]

    def test_no_lr_scale_without_llrd(self):
        model = _FakeWrapper(num_blocks=4)
        groups = _build_param_groups(model, default_weight_decay=0.05, layer_decay=None, num_blocks=4)
        for g in groups:
            assert "lr_scale" not in g
