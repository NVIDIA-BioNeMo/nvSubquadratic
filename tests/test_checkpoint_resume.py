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

"""Tests for checkpoint resume fixes and per-phase checkpoint uploads.

Covers:
1. Best-metric persistence across save/load checkpoint cycles.
2. ``current_model_state`` key remapping for EMA + torch.compile mismatch.
3. ``ResumableSequentialLR`` round-trip for cosine and WSD schedules.
4. ``WandbSelectiveCheckpointUploader`` phase determination and per-phase best tracking.

.. TODO(@dwromero/dwessels/dmknigge): Expand resume coverage
    - ``TestBestMetricsPersistence`` only covers ``ClassificationWrapper``.
      Add equivalent round-trip tests for ``RegressionWrapper``,
      ``AutoregressiveWrapper``, and ``DiffusionWrapper`` once they
      implement ``on_save_checkpoint`` / ``on_load_checkpoint``.
    - Add an integration test that runs a short train loop, saves a
      checkpoint, resumes, and verifies: (a) LR schedule continuity,
      (b) best-metric values, (c) EMA model weights match pre-resume
      state, (d) ``_val_metric_suffix`` is preserved.
    - Add a test for ``DiffusionWrapper``'s manual EMA save/restore
      once that is implemented.
"""

from typing import ClassVar

import pytest
import torch
import torch.nn as nn

from experiments.utils.checkpointing import WandbSelectiveCheckpointUploader, align_compiled_keys
from nvsubquadratic.modules.schedulers import ResumableSequentialLR


# ---------------------------------------------------------------------------
# Minimal stubs — just enough to exercise save/load without a full Trainer
# ---------------------------------------------------------------------------
class _TinyNet(nn.Module):
    """Two-param network with an ``out_proj`` (required by ClassificationWrapper)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.out_proj = nn.Linear(4, 10)

    def forward(self, x):
        return {"logits": self.out_proj(self.linear(x["input"]))}


def _make_wrapper():
    """Construct a ClassificationWrapper around ``_TinyNet`` with minimal config."""
    from dataclasses import fields

    from experiments.default_cfg import ExperimentConfig
    from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
    from nvsubquadratic.lazy_config import LazyConfig

    cfg = ExperimentConfig()
    cfg.optimizer = LazyConfig(torch.optim.Adam)(lr=1e-3, weight_decay=0.0)

    # Fill remaining PLACEHOLDER fields so the wrapper doesn't crash
    for f in fields(cfg):
        if getattr(cfg, f.name) is None and f.name not in ("optimizer",):
            pass  # leave as-is; not needed for this test

    net = _TinyNet()
    wrapper = ClassificationWrapper(network=net, cfg=cfg)
    return wrapper


# ---------------------------------------------------------------------------
# Test 1: best metrics are saved and restored
# ---------------------------------------------------------------------------
class TestBestMetricsPersistence:
    def test_round_trip(self):
        wrapper = _make_wrapper()

        # Simulate training progress
        wrapper.best_train_acc = 0.85
        wrapper.best_train_loss = 0.32
        wrapper.best_val_acc = 0.82
        wrapper.best_val_loss = 0.45

        # Save
        checkpoint: dict = {"state_dict": wrapper.state_dict()}
        wrapper.on_save_checkpoint(checkpoint)

        assert "best_metrics" in checkpoint
        assert checkpoint["best_metrics"]["best_val_acc"] == 0.82
        assert checkpoint["best_metrics"]["best_val_loss"] == 0.45
        assert checkpoint["best_metrics"]["best_train_acc"] == 0.85
        assert checkpoint["best_metrics"]["best_train_loss"] == 0.32

        # Create a fresh wrapper (simulates process restart)
        wrapper2 = _make_wrapper()
        assert wrapper2.best_val_acc == 0.0  # default
        assert wrapper2.best_val_loss == 1e9

        # Load
        wrapper2.on_load_checkpoint(checkpoint)
        assert wrapper2.best_train_acc == 0.85
        assert wrapper2.best_train_loss == 0.32
        assert wrapper2.best_val_acc == 0.82
        assert wrapper2.best_val_loss == 0.45

    def test_missing_best_metrics_is_safe(self):
        """Loading a legacy checkpoint without best_metrics should not crash."""
        wrapper = _make_wrapper()
        checkpoint: dict = {"state_dict": wrapper.state_dict()}
        wrapper.on_load_checkpoint(checkpoint)
        assert wrapper.best_val_acc == 0.0
        assert wrapper.best_val_loss == 1e9


# ---------------------------------------------------------------------------
# Test 2: current_model_state key remapping for EMA + compile
# ---------------------------------------------------------------------------
class TestEMAKeyRemap:
    @staticmethod
    def _add_orig_mod(state_dict: dict) -> dict:
        """Simulate keys produced by torch.compile (adds ``._orig_mod.`` segment)."""
        return {k.replace("network.", "network._orig_mod.", 1): v for k, v in state_dict.items()}

    def test_state_dict_remap_compiled_to_plain(self):
        """Checkpoint from compiled model loaded into non-compiled model."""
        wrapper = _make_wrapper()
        plain_keys = set(wrapper.state_dict().keys())

        compiled_sd = self._add_orig_mod(wrapper.state_dict())
        checkpoint = {"state_dict": compiled_sd}

        wrapper.on_load_checkpoint(checkpoint)
        assert set(checkpoint["state_dict"].keys()) == plain_keys

    def test_current_model_state_remap(self):
        """EMA's ``current_model_state`` is remapped just like ``state_dict``."""
        wrapper = _make_wrapper()
        plain_keys = set(wrapper.state_dict().keys())

        compiled_sd = self._add_orig_mod(wrapper.state_dict())
        compiled_cms = self._add_orig_mod(wrapper.state_dict())

        checkpoint = {
            "state_dict": compiled_sd,
            "current_model_state": compiled_cms,
        }

        wrapper.on_load_checkpoint(checkpoint)
        assert set(checkpoint["state_dict"].keys()) == plain_keys
        assert set(checkpoint["current_model_state"].keys()) == plain_keys

    def test_no_remap_when_keys_match(self):
        """When keys already match, the dicts should be left unchanged."""
        wrapper = _make_wrapper()
        sd = wrapper.state_dict()
        checkpoint = {
            "state_dict": dict(sd),
            "current_model_state": dict(sd),
        }
        wrapper.on_load_checkpoint(checkpoint)
        assert set(checkpoint["state_dict"].keys()) == set(sd.keys())
        assert set(checkpoint["current_model_state"].keys()) == set(sd.keys())

    def test_plain_to_compiled_remap(self):
        """Checkpoint from non-compiled model loaded into compiled model.

        We simulate a compiled model by manually checking that the reverse
        direction (plain -> compiled-prefix) also works via align_compiled_keys.
        """
        wrapper = _make_wrapper()
        plain_sd = wrapper.state_dict()
        compiled_keys = set(self._add_orig_mod(plain_sd).keys())

        remapped = align_compiled_keys(plain_sd, compiled_keys)
        assert set(remapped.keys()) == compiled_keys


# ---------------------------------------------------------------------------
# Test 3: ResumableSequentialLR state_dict round-trip
# ---------------------------------------------------------------------------
def _make_optimizer(lr=0.004, num_groups=1):
    """Create an SGD optimizer with *num_groups* parameter groups.

    When ``num_groups > 1``, each group gets a distinct base LR (lr, lr/2,
    lr/3, ...) so that tests verify per-group LR restoration, not just a
    single shared value.
    """
    params = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": lr / (i + 1)} for i in range(num_groups)]
    return torch.optim.SGD(params)


def _make_cosine_scheduler(lr=0.004, warmup=50, total=500, eta_min=0.0, num_groups=1):
    """Create a warmup+cosine scheduler matching construct_scheduler's cosine path."""
    opt = _make_optimizer(lr=lr, num_groups=num_groups)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=total - warmup,
        eta_min=eta_min,
    )
    seq = ResumableSequentialLR(opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])
    return opt, seq


def _make_wsd_scheduler(lr=0.004, warmup=50, stable=200, total=500, eta_min=0.0, num_groups=1):
    """Create a warmup+stable+decay scheduler matching construct_scheduler's WSD path."""
    opt = _make_optimizer(lr=lr, num_groups=num_groups)
    decay_iters = total - warmup - stable
    end_factor = max(eta_min / lr, 1e-8) if lr > 0 else 1e-8
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup,
    )
    stable_sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=stable)
    decay_sched = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=decay_iters,
    )
    milestones = [warmup, warmup + stable]
    seq = ResumableSequentialLR(
        opt,
        schedulers=[warmup_sched, stable_sched, decay_sched],
        milestones=milestones,
    )
    return opt, seq


def _roundtrip_at(make_fn, resume_step, check_steps=50, **kwargs):
    """Step to *resume_step*, save/load, then verify next *check_steps* match.

    Checks ALL param groups, not just the first.
    """
    opt1, seq1 = make_fn(**kwargs)
    for _ in range(resume_step):
        seq1.step()
    lrs_before = [pg["lr"] for pg in opt1.param_groups]
    sd = seq1.state_dict()

    opt2, seq2 = make_fn(**kwargs)
    seq2.load_state_dict(sd)
    lrs_after = [pg["lr"] for pg in opt2.param_groups]

    for g, (before, after) in enumerate(zip(lrs_before, lrs_after)):
        assert before == pytest.approx(after, abs=1e-12), (
            f"LR mismatch in group {g} at resume step {resume_step}: {before} vs {after}"
        )

    max_diffs = [0.0] * len(opt1.param_groups)
    for _ in range(check_steps):
        seq1.step()
        seq2.step()
        for g in range(len(opt1.param_groups)):
            diff = abs(opt1.param_groups[g]["lr"] - opt2.param_groups[g]["lr"])
            max_diffs[g] = max(max_diffs[g], diff)

    for g, md in enumerate(max_diffs):
        assert md < 1e-12, f"LR diverged in group {g} after resume at step {resume_step}: max_diff={md}"


class TestResumableSequentialLR:
    """Verify that ResumableSequentialLR correctly restores LR on load_state_dict."""

    @pytest.mark.parametrize("resume_step", [0, 10, 25, 49, 50, 51, 100, 250, 400, 499])
    def test_cosine_roundtrip(self, resume_step):
        _roundtrip_at(_make_cosine_scheduler, resume_step)

    @pytest.mark.parametrize("resume_step", [0, 25, 49, 50, 51, 150, 249, 250, 251, 350, 450, 499])
    def test_wsd_roundtrip(self, resume_step):
        _roundtrip_at(_make_wsd_scheduler, resume_step)

    def test_cosine_no_warmup(self):
        _roundtrip_at(_make_cosine_scheduler, resume_step=100, warmup=0, total=500)

    def test_wsd_no_warmup(self):
        _roundtrip_at(_make_wsd_scheduler, resume_step=100, warmup=0, stable=200, total=500)

    def test_wsd_no_stable(self):
        _roundtrip_at(_make_wsd_scheduler, resume_step=100, warmup=50, stable=0, total=500)

    def test_nonzero_eta_min(self):
        _roundtrip_at(_make_cosine_scheduler, resume_step=300, eta_min=1e-5)
        _roundtrip_at(_make_wsd_scheduler, resume_step=300, eta_min=1e-5)

    # --- Multiple param groups (mirrors real training with wd / no-wd groups) ---

    @pytest.mark.parametrize("resume_step", [25, 100, 300])
    def test_cosine_multi_group_roundtrip(self, resume_step):
        _roundtrip_at(_make_cosine_scheduler, resume_step, num_groups=2)

    @pytest.mark.parametrize("resume_step", [25, 150, 350])
    def test_wsd_multi_group_roundtrip(self, resume_step):
        _roundtrip_at(_make_wsd_scheduler, resume_step, num_groups=2)

    def test_cosine_many_groups(self):
        """Stress-test with 4 param groups (e.g. backbone / head / wd / no-wd)."""
        _roundtrip_at(_make_cosine_scheduler, resume_step=200, num_groups=4)

    def test_wsd_many_groups(self):
        """Stress-test with 4 param groups."""
        _roundtrip_at(_make_wsd_scheduler, resume_step=300, num_groups=4)

    def test_vanilla_sequential_lr_is_broken(self):
        """Confirm the PyTorch bug exists — vanilla SequentialLR does NOT restore LR."""
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        w = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=10)
        c = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=90, eta_min=0.001)
        seq = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[w, c], milestones=[10])

        for _ in range(50):
            seq.step()
        expected_lr = opt.param_groups[0]["lr"]
        sd = seq.state_dict()

        opt2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        w2 = torch.optim.lr_scheduler.LinearLR(opt2, start_factor=0.01, total_iters=10)
        c2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=90, eta_min=0.001)
        seq2 = torch.optim.lr_scheduler.SequentialLR(opt2, schedulers=[w2, c2], milestones=[10])
        seq2.load_state_dict(sd)

        restored_lr = opt2.param_groups[0]["lr"]
        assert restored_lr != pytest.approx(expected_lr, abs=1e-6), (
            "If this fails, PyTorch fixed the SequentialLR bug — ResumableSequentialLR can be removed"
        )


# ---------------------------------------------------------------------------
# Test 4: WandbSelectiveCheckpointUploader — phase determination & tracking
# ---------------------------------------------------------------------------
def _make_uploader(phase_boundaries=None, mode="max", **kwargs):
    """Create a WandbSelectiveCheckpointUploader for unit testing."""
    return WandbSelectiveCheckpointUploader(
        upload_best=True,
        upload_last=True,
        phase_boundaries=phase_boundaries,
        mode=mode,
        **kwargs,
    )


class TestSchedulerPhaseBoundaries:
    """Verify ``_scheduler_phase_boundaries`` helper in trainer.py."""

    def test_wsd_boundaries(self):
        from experiments.default_cfg import ExperimentConfig, SchedulerConfig
        from experiments.trainer import _scheduler_phase_boundaries

        cfg = ExperimentConfig()
        cfg.scheduler = SchedulerConfig(
            name="wsd",
            warmup_iterations_percentage=0.1,
            stable_iterations_percentage=0.6,
            total_iterations=10000,
            mode="max",
        )
        boundaries = _scheduler_phase_boundaries(cfg)
        assert boundaries == {"stable": (1000, 7000), "decay": (7000, 10000)}

    def test_cosine_boundaries(self):
        from experiments.default_cfg import ExperimentConfig, SchedulerConfig
        from experiments.trainer import _scheduler_phase_boundaries

        cfg = ExperimentConfig()
        cfg.scheduler = SchedulerConfig(
            name="cosine",
            warmup_iterations_percentage=0.05,
            total_iterations=20000,
            mode="max",
        )
        boundaries = _scheduler_phase_boundaries(cfg)
        assert boundaries == {"cosine": (1000, 20000)}

    def test_constant_boundaries(self):
        from experiments.default_cfg import ExperimentConfig, SchedulerConfig
        from experiments.trainer import _scheduler_phase_boundaries

        cfg = ExperimentConfig()
        cfg.scheduler = SchedulerConfig(
            name="constant",
            warmup_iterations_percentage=0.02,
            total_iterations=5000,
            mode="max",
        )
        boundaries = _scheduler_phase_boundaries(cfg)
        assert boundaries == {"constant": (100, 5000)}

    def test_unknown_scheduler_returns_empty(self):
        from experiments.default_cfg import ExperimentConfig, SchedulerConfig
        from experiments.trainer import _scheduler_phase_boundaries

        cfg = ExperimentConfig()
        cfg.scheduler = SchedulerConfig(
            name="polynomial",
            total_iterations=1000,
            mode="max",
        )
        assert _scheduler_phase_boundaries(cfg) == {}


class TestUploaderPhaseDetermination:
    """Verify ``_current_phase`` correctly maps global_step to phase name."""

    WSD_BOUNDARIES: ClassVar[dict] = {"stable": (100, 800), "decay": (800, 1000)}

    def test_warmup_returns_none(self):
        up = _make_uploader(self.WSD_BOUNDARIES)
        assert up._current_phase(0) is None
        assert up._current_phase(50) is None
        assert up._current_phase(99) is None

    def test_stable_phase(self):
        up = _make_uploader(self.WSD_BOUNDARIES)
        assert up._current_phase(100) == "stable"
        assert up._current_phase(500) == "stable"
        assert up._current_phase(799) == "stable"

    def test_decay_phase(self):
        up = _make_uploader(self.WSD_BOUNDARIES)
        assert up._current_phase(800) == "decay"
        assert up._current_phase(900) == "decay"
        assert up._current_phase(999) == "decay"

    def test_beyond_total_returns_none(self):
        up = _make_uploader(self.WSD_BOUNDARIES)
        assert up._current_phase(1000) is None
        assert up._current_phase(5000) is None

    def test_cosine_single_phase(self):
        up = _make_uploader({"cosine": (50, 500)})
        assert up._current_phase(49) is None
        assert up._current_phase(50) == "cosine"
        assert up._current_phase(499) == "cosine"
        assert up._current_phase(500) is None

    def test_no_boundaries_always_none(self):
        up = _make_uploader()
        assert up._current_phase(0) is None
        assert up._current_phase(500) is None


class TestUploaderPhaseTracking:
    """Verify per-phase best metric tracking."""

    WSD_BOUNDARIES: ClassVar[dict] = {"stable": (100, 800), "decay": (800, 1000)}

    def test_max_mode_improvement(self):
        up = _make_uploader(self.WSD_BOUNDARIES, mode="max")
        assert up._phase_best == {"stable": float("-inf"), "decay": float("-inf")}

        assert up._is_phase_improvement("stable", 0.5)
        up._phase_best["stable"] = 0.5
        assert up._is_phase_improvement("stable", 0.6)
        assert not up._is_phase_improvement("stable", 0.5)
        assert not up._is_phase_improvement("stable", 0.4)

    def test_min_mode_improvement(self):
        up = _make_uploader(self.WSD_BOUNDARIES, mode="min")
        assert up._phase_best == {"stable": float("inf"), "decay": float("inf")}

        assert up._is_phase_improvement("stable", 1.0)
        up._phase_best["stable"] = 1.0
        assert up._is_phase_improvement("stable", 0.8)
        assert not up._is_phase_improvement("stable", 1.0)
        assert not up._is_phase_improvement("stable", 1.5)

    def test_phases_tracked_independently(self):
        """Improving in stable does not affect decay tracking."""
        up = _make_uploader(self.WSD_BOUNDARIES, mode="max")
        up._phase_best["stable"] = 0.9
        assert up._is_phase_improvement("decay", 0.1)
        up._phase_best["decay"] = 0.1
        assert not up._is_phase_improvement("stable", 0.8)

    def test_empty_boundaries_has_no_tracking(self):
        up = _make_uploader()
        assert up._phase_best == {}

    def test_initial_score_always_improves(self):
        """First score in any phase is always an improvement."""
        up = _make_uploader(self.WSD_BOUNDARIES, mode="max")
        assert up._is_phase_improvement("stable", -1000.0)

        up_min = _make_uploader(self.WSD_BOUNDARIES, mode="min")
        assert up_min._is_phase_improvement("decay", 1e12)
