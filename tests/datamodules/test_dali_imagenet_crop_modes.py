# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``DALIImageNetFusedDataModule.train_crop_mode``.

A new ``train_crop_mode: Literal["random_resized", "center"]`` flag was
added so the JiT-exact diffusion configs can opt into the deterministic
resize-shorter-side + center-crop preprocessing JiT uses, instead of the
default classification-style ``RandomResizedCrop(scale=(0.08, 1.0))``.

DALI is imported at module level by ``dali_imagenet_fused``, so the whole
file is skipped on environments without ``nvidia.dali`` installed.  These
tests therefore run on the GPU/container build path (matching the project
convention for DALI-touching code) — see ``CLAUDE``-style notes in the
workspace rules under ``conda-env.mdc``.
"""

from __future__ import annotations

import pytest


# Skip the whole module if DALI isn't importable in this env (e.g. CPU CI box).
pytest.importorskip("nvidia.dali")

from experiments.datamodules.dali_imagenet_fused import (
    DALIImageNetFusedDataModule,
)


# =============================================================================
# Construction-time validation
# =============================================================================


def _make_dm(**overrides) -> DALIImageNetFusedDataModule:
    """Build a minimal DALI datamodule (does not trigger DALI graph build)."""
    kwargs = {
        "data_dir": "/tmp/nonexistent",
        "imagefolder_dir": "/tmp/nonexistent/imagefolder",
        "batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
        "seed": 0,
        "image_size": 64,
        "task": "generation",
        "channels_first": False,
    }
    kwargs.update(overrides)
    return DALIImageNetFusedDataModule(**kwargs)


def test_default_train_crop_mode_is_random_resized() -> None:
    """Backward compatibility: the legacy default is preserved for v5_hybrid users."""
    dm = _make_dm()
    assert dm.train_crop_mode == "random_resized"


def test_explicit_train_crop_mode_center() -> None:
    """The new ``"center"`` value is accepted and stored as an attribute."""
    dm = _make_dm(train_crop_mode="center")
    assert dm.train_crop_mode == "center"


def test_invalid_train_crop_mode_raises() -> None:
    """Unknown values fail fast at ``__init__`` with a clear message."""
    with pytest.raises(ValueError, match="train_crop_mode"):
        _make_dm(train_crop_mode="bogus")


# =============================================================================
# Pipeline-level: each mode produces a buildable DALI graph
# =============================================================================
#
# Pipeline construction calls into DALI's pipeline_def decorator which only
# requires DALI itself (not a GPU).  We don't ``.build()`` because that
# requires a real device + reader files; verifying the pipeline factory
# accepts the new kwarg and returns a Pipeline object is enough.


def test_train_pipeline_factory_accepts_both_modes() -> None:
    """``_train_pipeline_fused`` must accept ``train_crop_mode`` for both values."""
    from experiments.datamodules.dali_imagenet_fused import _train_pipeline_fused

    for mode in ("random_resized", "center"):
        pipe = _train_pipeline_fused(
            file_root="/tmp/nonexistent",
            image_size=64,
            final_image_size=64,
            norm_mean=(0.5, 0.5, 0.5),
            norm_std=(0.5, 0.5, 0.5),
            train_crop_mode=mode,
            shard_id=0,
            num_shards=1,
            batch_size=2,
            num_threads=1,
            device_id=0,
            seed=0,
        )
        # Returned object should be a DALI Pipeline instance.
        from nvidia.dali.pipeline import Pipeline

        assert isinstance(pipe, Pipeline), f"train_crop_mode={mode!r} did not yield a Pipeline"


def test_train_pipeline_factory_rejects_invalid_mode() -> None:
    """An unknown ``train_crop_mode`` propagates a ValueError out of the factory."""
    from experiments.datamodules.dali_imagenet_fused import _train_pipeline_fused

    # DALI's ``pipeline_def`` defers the body to ``Pipeline.build()`` for some
    # ops, but the ``raise ValueError(...)`` for an unknown mode lives in plain
    # Python and fires during pipeline construction.
    with pytest.raises(ValueError, match="train_crop_mode"):
        _train_pipeline_fused(
            file_root="/tmp/nonexistent",
            image_size=64,
            final_image_size=64,
            norm_mean=(0.5, 0.5, 0.5),
            norm_std=(0.5, 0.5, 0.5),
            train_crop_mode="bogus",
            shard_id=0,
            num_shards=1,
            batch_size=2,
            num_threads=1,
            device_id=0,
            seed=0,
        ).build()


# =============================================================================
# Integration: JiT base config uses DALI with train_crop_mode="center"
# =============================================================================


def test_jit_base_config_wires_dali_center_crop() -> None:
    """Smoke test the public surface — the JiT base config wires DALI properly."""
    from examples.imagenet_diffusion_jit._base_config import get_base_config
    from nvsubquadratic.networks.jit import JiT_B_16

    cfg = get_base_config(
        model_factory=JiT_B_16,
        image_size=256,
        batch_size=4,
        num_gpus=1,
        accumulate_grad_steps=1,
    )
    # Dataset target should now be DALI, not the deprecated HF backend.
    target = cfg.dataset["__target__"]
    assert target.endswith("DALIImageNetFusedDataModule"), (
        f"JiT base config should now build a DALI dataset; got target={target!r}"
    )
    assert cfg.dataset["train_crop_mode"] == "center"
    assert cfg.dataset["task"] == "generation"
    assert cfg.dataset["channels_first"] is False
    assert cfg.dataset["drop_labels"] is False
