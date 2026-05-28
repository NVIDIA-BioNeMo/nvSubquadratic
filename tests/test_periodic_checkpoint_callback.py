# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``TrainerConfig.periodic_save_every_n_steps``.

A second ``ModelCheckpoint`` callback was added so users can retain the
full periodic snapshot history (``save_top_k=-1``) for offline FID / model
selection, on top of the rolling ``last.ckpt`` produced by the main
checkpoint callback.

These tests pin:

- the config field exists with a ``None`` default (backwards-compatible),
- when ``None`` no extra callback is attached,
- when set, exactly one extra ``ModelCheckpoint`` is attached with the
  expected retention / cadence / output-dir settings.
"""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning.callbacks as pl_callbacks


def _read_trainer_callbacks(tmp_path: Path, periodic_save_every_n_steps: int | None):
    """Build the project's trainer callbacks list and return ``(main_ckpt, periodic_ckpts)``."""
    import importlib
    import sys

    # We don't run a real Trainer; just call ``construct_trainer`` and read the
    # callbacks list off the returned object.
    construct_trainer = importlib.import_module("experiments.trainer").construct_trainer

    # Build a minimal ExperimentConfig with whatever construct_trainer needs.
    from experiments.default_cfg import (
        ExperimentConfig,
        SchedulerConfig,
        TrainConfig,
        TrainerConfig,
    )

    cfg = ExperimentConfig()
    cfg.train = TrainConfig(iterations=10, batch_size=1, precision="32-true")
    cfg.scheduler = SchedulerConfig(name="constant", warmup_iterations_percentage=0.0, total_iterations=10, mode="min")
    cfg.trainer = TrainerConfig(
        checkpoint_every_n_steps=5,
        periodic_save_every_n_steps=periodic_save_every_n_steps,
        wandb_checkpoint_upload=False,  # avoid wandb dependency in the test
    )

    # ``construct_trainer`` writes into runs/<run_name>/checkpoints; use a tmpdir.
    # It returns ``(trainer, checkpoint_callback)``; we only need the trainer's
    # full callback list to inspect what got attached.
    sys.path.insert(0, str(tmp_path))
    try:
        trainer, _ckpt = construct_trainer(
            cfg=cfg,
            wandb_logger=None,  # logger not exercised in this codepath
            run_name="periodic_ckpt_test",
            experiment_dir=tmp_path / "run_dir",
            num_nodes=1,
        )
    finally:
        sys.path.pop(0)

    ckpt_cbs = [cb for cb in trainer.callbacks if isinstance(cb, pl_callbacks.ModelCheckpoint)]
    return ckpt_cbs


def test_periodic_save_default_is_none() -> None:
    """``TrainerConfig.periodic_save_every_n_steps`` defaults to ``None``."""
    from experiments.default_cfg import TrainerConfig

    cfg = TrainerConfig()
    assert cfg.periodic_save_every_n_steps is None


def test_no_periodic_callback_when_unset(tmp_path: Path) -> None:
    """With ``periodic_save_every_n_steps=None`` only the main ``ModelCheckpoint`` is attached."""
    ckpt_cbs = _read_trainer_callbacks(tmp_path, periodic_save_every_n_steps=None)
    assert len(ckpt_cbs) == 1
    only = ckpt_cbs[0]
    # Main callback keeps best + rolling last.
    assert only.save_top_k == 1
    assert only.save_last is True


def test_periodic_callback_attached_when_set(tmp_path: Path) -> None:
    """Setting the field attaches a second callback with ``save_top_k=-1``."""
    n = 100
    ckpt_cbs = _read_trainer_callbacks(tmp_path, periodic_save_every_n_steps=n)
    assert len(ckpt_cbs) == 2, "expected main + periodic ModelCheckpoint, got: " + str(ckpt_cbs)

    # Identify the periodic one — it's the one with save_top_k=-1.
    periodic = next((cb for cb in ckpt_cbs if cb.save_top_k == -1), None)
    assert periodic is not None, "no periodic checkpoint callback found (save_top_k=-1)"

    # Periodic settings.
    assert periodic._every_n_train_steps == n
    assert periodic.save_last in (False, None)  # Lightning may normalise; either is fine
    # Lands in a separate ``periodic/`` subdir so it doesn't shadow best/last.
    assert Path(periodic.dirpath).name == "periodic"
    # Filename pattern includes step + epoch for unique snapshot names.
    assert "step=" in periodic.filename
    assert "epoch=" in periodic.filename


def test_main_callback_unaffected_by_periodic_flag(tmp_path: Path) -> None:
    """The original ``ModelCheckpoint`` keeps its best+last contract regardless."""
    ckpt_cbs = _read_trainer_callbacks(tmp_path, periodic_save_every_n_steps=100)
    main = next(cb for cb in ckpt_cbs if cb.save_top_k != -1)
    assert main.save_top_k == 1
    assert main.save_last is True
    assert main._every_n_train_steps == 5  # = TrainConfig.checkpoint_every_n_steps used above


# =============================================================================
# Regression: monitor=None mode (the "no monitor" sentinel)
# =============================================================================


def _read_main_callback_with_monitor(tmp_path: Path, checkpoint_monitor):
    """Construct the trainer with a specific ``checkpoint_monitor`` value and
    return only the main (non-periodic) ``ModelCheckpoint`` callback."""
    import importlib
    import sys

    construct_trainer = importlib.import_module("experiments.trainer").construct_trainer
    from experiments.default_cfg import (
        ExperimentConfig,
        SchedulerConfig,
        TrainConfig,
        TrainerConfig,
    )

    cfg = ExperimentConfig()
    cfg.train = TrainConfig(iterations=10, batch_size=1, precision="32-true")
    cfg.scheduler = SchedulerConfig(name="constant", warmup_iterations_percentage=0.0, total_iterations=10, mode="min")
    cfg.trainer = TrainerConfig(
        checkpoint_every_n_steps=5,
        checkpoint_monitor=checkpoint_monitor,
        wandb_checkpoint_upload=False,
    )

    sys.path.insert(0, str(tmp_path))
    try:
        trainer, _ckpt = construct_trainer(
            cfg=cfg,
            wandb_logger=None,
            run_name="monitor_test",
            experiment_dir=tmp_path / "run_dir",
            num_nodes=1,
        )
    finally:
        sys.path.pop(0)

    ckpt_cbs = [cb for cb in trainer.callbacks if isinstance(cb, pl_callbacks.ModelCheckpoint)]
    return next(cb for cb in ckpt_cbs if cb.save_top_k != -1)


def test_default_monitor_is_auto_derived_from_scheduler_mode(tmp_path: Path) -> None:
    """``checkpoint_monitor=None`` -> ``"val/loss"`` (scheduler.mode='min' here)."""
    main = _read_main_callback_with_monitor(tmp_path, checkpoint_monitor=None)
    assert main.monitor == "val/loss"


def test_explicit_monitor_string_is_used_verbatim(tmp_path: Path) -> None:
    """A non-empty ``checkpoint_monitor`` string is forwarded directly."""
    main = _read_main_callback_with_monitor(tmp_path, checkpoint_monitor="train/loss_step")
    assert main.monitor == "train/loss_step"


def test_empty_string_monitor_disables_metric_gating(tmp_path: Path) -> None:
    """``checkpoint_monitor=""`` (opt-out) sets ``monitor=None`` on the callback.

    Regression test for the bug where diffusion runs with
    ``check_val_every_n_epoch=40`` produced no checkpoints for the first
    50K steps because Lightning silently skipped saves whose monitor
    metric had never been logged.  With ``monitor=None`` the main
    callback saves unconditionally on ``every_n_train_steps`` and
    ``save_last=True`` produces a rolling ``last.ckpt``.
    """
    main = _read_main_callback_with_monitor(tmp_path, checkpoint_monitor="")
    assert main.monitor is None
    # save_last + every_n_train_steps must still be wired so rolling last works.
    assert main.save_last is True
    assert main._every_n_train_steps == 5


# =============================================================================
# End-to-end: run Lightning Trainer.fit() and prove the ckpt actually appears
# =============================================================================
#
# Unit tests above only check the callback is constructed with the right args.
# These tests run a real ``Trainer.fit()`` on a CPU-only trivial model and
# assert that ``last.ckpt`` (and ``best.ckpt`` / step-snapshot ckpts) actually
# materialise on disk at the expected save trigger.  This is the only way to
# catch Lightning behaviours like "monitor metric missing -> silent skip".


class _TrivialLM(__import__("pytorch_lightning").LightningModule):
    """Minimal trainable LightningModule (one learnable scalar, MSE loss)."""

    def __init__(self) -> None:
        super().__init__()
        import torch

        self.p = torch.nn.Parameter(torch.zeros(1))

    def training_step(self, batch, batch_idx):

        loss = (self.p - batch).pow(2).mean()
        # Log a step-frequency metric so the "explicit monitor on a train
        # metric" path has something to compare against.
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def configure_optimizers(self):
        import torch

        return torch.optim.SGD(self.parameters(), lr=1e-2)


def _run_short_trainer(
    tmp_path: Path,
    *,
    checkpoint_monitor,
    max_steps: int = 30,
    every_n_train_steps: int = 10,
):
    """Run a 30-step ``Trainer.fit()`` and return the checkpoint dir contents."""
    import importlib
    import sys

    import pytorch_lightning as pl
    import torch

    construct_trainer = importlib.import_module("experiments.trainer").construct_trainer
    from experiments.default_cfg import (
        ExperimentConfig,
        SchedulerConfig,
        TrainConfig,
        TrainerConfig,
    )

    cfg = ExperimentConfig()
    cfg.train = TrainConfig(iterations=max_steps, batch_size=4, precision="32-true")
    cfg.scheduler = SchedulerConfig(
        name="constant", warmup_iterations_percentage=0.0, total_iterations=max_steps, mode="min"
    )
    cfg.trainer = TrainerConfig(
        checkpoint_every_n_steps=every_n_train_steps,
        checkpoint_monitor=checkpoint_monitor,
        wandb_checkpoint_upload=False,
    )

    sys.path.insert(0, str(tmp_path))
    try:
        trainer, _ckpt_cb = construct_trainer(
            cfg=cfg,
            wandb_logger=None,
            run_name="e2e_ckpt_test",
            experiment_dir=tmp_path / "run_dir",
            num_nodes=1,
        )
    finally:
        sys.path.pop(0)

    # Override the trainer for a strictly CPU-only short fit (sidesteps GPU /
    # DDP / fp16 plumbing inside ``construct_trainer`` for this unit test).
    # Only forward the ModelCheckpoint callbacks (the others — progress bar,
    # LR monitor, timer — would conflict with ``enable_progress_bar=False``
    # or pollute the unit test with noisy output).
    ckpt_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, pl_callbacks.ModelCheckpoint)]
    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="cpu",
        devices=1,
        callbacks=ckpt_callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        check_val_every_n_epoch=10_000,  # never validate -> reproduces the bug
    )

    model = _TrivialLM()
    dataset = [torch.tensor([float(i)]) for i in range(max_steps * 2)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    trainer.fit(model, train_dataloaders=loader)

    ckpt_dir = tmp_path / "run_dir" / "checkpoints"
    return sorted(p.name for p in ckpt_dir.glob("*.ckpt"))


def test_e2e_empty_string_monitor_writes_last_ckpt(tmp_path: Path) -> None:
    """Fix verification: with ``checkpoint_monitor=''`` Lightning saves the
    rolling ``last.ckpt`` on every ``every_n_train_steps`` trigger, even
    when no validation has ever run.

    This is the contract the JiT base config now relies on for crash
    recovery during the ~50K-step gap before the first val epoch.

    Note: we explicitly do *not* try to also reproduce the original bug
    here (``monitor='val/loss'`` failing to save).  On CPU + a trivial
    LightningModule the bug does not reproduce (Lightning DOES write
    ``last.ckpt`` in that path), so any such test would be a false
    negative.  In the live DDP+wandb+DALI environment the same config
    DID fail to produce any ckpts; the root cause is environment-specific
    and outside the scope of this unit test.  The CRITICAL invariant we
    *can* prove on CPU is that the ``monitor=None`` path used by our fix
    saves ``last.ckpt`` correctly — that's what this test pins.
    """
    files = _run_short_trainer(tmp_path, checkpoint_monitor="", max_steps=30, every_n_train_steps=10)
    # ``save_top_k=1`` + ``save_last=True`` produces at least ``last.ckpt``
    # (and possibly a step-named "best" ckpt). The CRITICAL assertion is
    # that ``last.ckpt`` exists — that's what autoresume reads on a fresh
    # SLURM allocation.
    assert "last.ckpt" in files, (
        f"FIX FAILED: expected last.ckpt in checkpoints/ with monitor='' + every_n_train_steps=10, got {files}"
    )
