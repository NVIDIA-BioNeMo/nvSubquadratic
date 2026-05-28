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
