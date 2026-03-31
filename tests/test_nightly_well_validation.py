"""Nightly validation tests for WELL benchmark models.

Verifies that our dataloader changes (persistent_workers, prefetch_factor,
shuffle=False, drop_last=False on eval) produce metrics matching the upstream
dataloader within floating-point tolerance.

Prerequisites (all provided by the SLURM container):
  - GPU with CUDA
  - WELL dataset at ``/shared/data/image_datasets/the_well/datasets``
  - ``WANDB_API_KEY`` environment variable

Run:
    source .env && PYTHONPATH=. python -m pytest tests/test_nightly_well_validation.py -m nightly -v -o addopts=""
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from experiments.utils.checkpointing import (
    StripCompiledPrefix,
    download_checkpoint,
    load_checkpoint_state_dict,
)
from nvsubquadratic.lazy_config import instantiate


# ─── Constants ────────────────────────────────────────────────────────────────

WANDB_ENTITY = "implicit-long-convs"
WANDB_PROJECT = "nvsubquadratic"
WELL_BASE_PATH = "/shared/data/image_datasets/the_well/datasets"
WELL_DATASET_NAME = "gray_scott_reaction_diffusion"

# Relative tolerance for metric comparison.
# Our changes vs upstream differ only by batch composition (drop_last removes
# up to batch_size-1 samples) and floating-point reduction order (shuffle).
# Empirically, the largest relative difference observed is < 0.02%.
RTOL = 5e-3

# ─── Skip conditions ─────────────────────────────────────────────────────────

_SKIP_REASONS: list[tuple[bool, str]] = [
    (not torch.cuda.is_available(), "CUDA not available"),
    ("WANDB_API_KEY" not in os.environ, "WANDB_API_KEY not set (run `source .env`)"),
    (
        not os.path.isdir(os.path.join(WELL_BASE_PATH, WELL_DATASET_NAME)),
        f"WELL dataset not found at {WELL_BASE_PATH}/{WELL_DATASET_NAME}",
    ),
]

_skip_nightly = pytest.mark.skipif(
    any(cond for cond, _ in _SKIP_REASONS),
    reason="; ".join(reason for cond, reason in _SKIP_REASONS if cond),
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_loader(dataset, batch_size, num_workers, *, shuffle, drop_last, persistent_workers, prefetch_factor):
    """Build a DataLoader with explicit parameters."""
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and use_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
    )


def _run_eval_variant(trainer, model, datamodule, *, val_shuffle, val_drop_last, persistent_workers, prefetch_factor):
    """Monkey-patch datamodule loaders and run validation + test."""
    bs = datamodule.batch_size
    nw = datamodule.num_workers
    dm = datamodule._well_datamodule
    common = {"persistent_workers": persistent_workers, "prefetch_factor": prefetch_factor}

    datamodule.val_dataloader = lambda: _make_loader(
        dm.val_dataset, bs, nw, shuffle=val_shuffle, drop_last=val_drop_last, **common
    )
    datamodule.test_dataloader = lambda: _make_loader(
        dm.test_dataset, bs, nw, shuffle=False, drop_last=val_drop_last, **common
    )

    val_results = trainer.validate(model, datamodule=datamodule)
    test_results = trainer.test(model, datamodule=datamodule)
    return val_results[0], test_results[0]


def _load_config_and_model(wandb_run_id: str, batch_size: int = 64, num_workers: int = 4):
    """Load config, download checkpoint, and return (model, datamodule)."""
    from examples.well.gray_scott_reaction_diffusion.cfg_hyena_gaussian_mask import (
        get_config,
    )

    config = get_config()
    config.dataset.batch_size = batch_size
    config.dataset.num_workers = num_workers
    config.dataset.local_staging_dir = None  # always read from /shared
    config.dataset.well_base_path = WELL_BASE_PATH
    config.compile = False
    config.debug = True

    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if getattr(config, "mp_sharing_strategy", None):
        torch.multiprocessing.set_sharing_strategy(config.mp_sharing_strategy)

    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup()

    network = instantiate(config.net)
    wrapper_kwargs = {"network": network, "cfg": config}
    if hasattr(datamodule, "metadata"):
        wrapper_kwargs["metadata"] = datamodule.metadata
    if hasattr(datamodule, "normalization"):
        wrapper_kwargs["normalization"] = datamodule.normalization
    model = instantiate(config.lightning_wrapper_class, **wrapper_kwargs)

    # Download and load checkpoint
    run_path = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{wandb_run_id}"
    ckpt_path = download_checkpoint(run_path=run_path, alias="latest")
    state_dict = load_checkpoint_state_dict(ckpt_path)
    strip = StripCompiledPrefix()
    state_dict = strip(state_dict=state_dict, model=model)
    model.load_state_dict(state_dict, strict=True)

    return model, datamodule, config


# ─── Nightly tests ───────────────────────────────────────────────────────────


@pytest.mark.nightly
@_skip_nightly
def test_well_dataloader_parity_gray_scott() -> None:
    """Verify our dataloader fixes produce metrics matching upstream within tolerance.

    Downloads the Gray-Scott Hyena checkpoint from W&B (run vjetobiy) and runs
    validation + test twice:
      1. "Ours":     shuffle=False, drop_last=False, persistent_workers=True, prefetch_factor=4
      2. "Upstream": shuffle=True (val), drop_last=True, persistent_workers=False, prefetch_factor=2

    Asserts all metrics agree within RTOL.

    Config: gray_scott_reaction_diffusion/cfg_hyena_gaussian_mask
    W&B run: vjetobiy
    """
    model, datamodule, config = _load_config_and_model(
        wandb_run_id="vjetobiy",
        batch_size=64,
        num_workers=4,
    )

    logger = WandbLogger(
        project="nvsubquadratic-nightly",
        save_dir="/tmp/nightly_well_validate",
        offline=True,
        name="nightly-well-parity",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=config.train.precision,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )

    # Variant 1: ours
    pl.seed_everything(config.seed, workers=True)
    ours_val, ours_test = _run_eval_variant(
        trainer,
        model,
        datamodule,
        val_shuffle=False,
        val_drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Variant 2: upstream-like
    pl.seed_everything(config.seed, workers=True)
    upstream_val, upstream_test = _run_eval_variant(
        trainer,
        model,
        datamodule,
        val_shuffle=True,
        val_drop_last=True,
        persistent_workers=False,
        prefetch_factor=2,
    )

    # Compare all metrics
    for split_name, ours_d, upstream_d in [
        ("val", ours_val, upstream_val),
        ("test", ours_test, upstream_test),
    ]:
        shared_keys = sorted(set(ours_d.keys()) & set(upstream_d.keys()))
        assert shared_keys, f"No shared keys for {split_name}!"

        for key in shared_keys:
            ov = ours_d[key]
            uv = upstream_d[key]
            if isinstance(ov, (int, float)) and isinstance(uv, (int, float)):
                denom = max(abs(uv), 1e-12)
                rel_diff = abs(ov - uv) / denom
                assert rel_diff < RTOL, (
                    f"{split_name}/{key}: ours={ov}, upstream={uv}, rel_diff={rel_diff:.6f} exceeds tolerance {RTOL}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "nightly", "-o", "addopts="])
