"""Tests for ``_build_param_groups`` and ``construct_optimizer``.

Covers:
- Weight-decay grouping (default / ``_no_weight_decay`` / ``_weight_decay``).
- 1D-parameter weight-decay warnings.
- Per-parameter ``_lr_scale`` overrides + their effect on
  ``construct_optimizer``'s per-group ``lr``.
"""

import warnings

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from experiments.lightning_wrappers.base_lightning_wrapper import (
    _build_param_groups,
    construct_optimizer,
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
# _build_param_groups — per-parameter _lr_scale
# ---------------------------------------------------------------------------


class TestBuildParamGroupsLrScale:
    def test_no_lr_scale_by_default(self):
        """Without any ``_lr_scale`` attribute, no group emits an ``lr_scale`` key."""
        model = _FakeWrapper(num_blocks=4)
        groups = _build_param_groups(model, default_weight_decay=0.05)
        assert all("lr_scale" not in g for g in groups)

    def test_unit_lr_scale_is_omitted(self):
        """``_lr_scale = 1.0`` is treated as the default and not emitted."""
        model = _FakeWrapper(num_blocks=2)
        model.network.patch_embed.weight._lr_scale = 1.0
        groups = _build_param_groups(model, default_weight_decay=0.05)
        assert all("lr_scale" not in g for g in groups)

    def test_custom_lr_scale_creates_separate_group(self):
        """A non-trivial ``_lr_scale`` produces its own group with ``lr_scale``."""
        model = _FakeWrapper(num_blocks=2)
        # Use a non-default weight-decay value to also avoid the 1D warning,
        # since this parameter (Linear weight) is 2D.
        model.network.patch_embed.weight._lr_scale = 0.25
        groups = _build_param_groups(model, default_weight_decay=0.05)

        scaled = [g for g in groups if "lr_scale" in g]
        assert len(scaled) == 1
        assert scaled[0]["lr_scale"] == pytest.approx(0.25)
        assert id(model.network.patch_embed.weight) in {id(p) for p in scaled[0]["params"]}

    def test_lr_scale_groups_split_by_value(self):
        """Different ``_lr_scale`` values land in different groups."""
        model = _FakeWrapper(num_blocks=2)
        model.network.patch_embed.weight._lr_scale = 0.25
        model.network.out_proj.weight._lr_scale = 4.0
        groups = _build_param_groups(model, default_weight_decay=0.05)

        scaled = sorted([g["lr_scale"] for g in groups if "lr_scale" in g])
        assert scaled == pytest.approx([0.25, 4.0])

    def test_negative_or_zero_lr_scale_raises(self):
        model = _FakeWrapper(num_blocks=2)
        model.network.patch_embed.weight._lr_scale = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            _build_param_groups(model, default_weight_decay=0.05)


# ---------------------------------------------------------------------------
# construct_optimizer — _lr_scale propagates to per-group lr
# ---------------------------------------------------------------------------


class TestConstructOptimizerLrScale:
    def _adamw_cfg(self, lr: float, weight_decay: float):
        return OmegaConf.create(
            {
                "__target__": "torch.optim.AdamW",
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )

    def test_lr_scale_applies_base_lr_multiplier(self):
        model = _FakeWrapper(num_blocks=2)
        model.network.patch_embed.weight._lr_scale = 0.25

        base_lr = 1e-3
        optim = construct_optimizer(model, self._adamw_cfg(lr=base_lr, weight_decay=0.05))

        scaled_groups = [g for g in optim.param_groups if g["lr"] != base_lr]
        assert len(scaled_groups) == 1
        assert scaled_groups[0]["lr"] == pytest.approx(base_lr * 0.25)
        assert id(model.network.patch_embed.weight) in {id(p) for p in scaled_groups[0]["params"]}

    def test_no_lr_scale_inherits_base_lr(self):
        model = _FakeWrapper(num_blocks=2)

        base_lr = 1e-3
        optim = construct_optimizer(model, self._adamw_cfg(lr=base_lr, weight_decay=0.05))

        for g in optim.param_groups:
            assert g["lr"] == pytest.approx(base_lr)
