"""Tests for ARCDataModule — verifies setup(), eval-split validation, and batch format."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.datamodules.arc import IGNORE_INDEX, PAD_INDEX, ARCDataModule


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_task(path: Path, n_train: int = 2, n_test: int = 1) -> None:
    """Write a minimal ARC task JSON with small 2×2 grids."""
    example = {"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]}
    task = {
        "train": [example] * n_train,
        "test": [example] * n_test,
    }
    path.write_text(json.dumps(task))


@pytest.fixture()
def arc_data_dir(tmp_path: Path) -> Path:
    """Minimal ARC directory: 5 training tasks and 3 evaluation tasks."""
    (tmp_path / "training").mkdir()
    (tmp_path / "evaluation").mkdir()
    for i in range(5):
        _write_task(tmp_path / "training" / f"task_{i:03d}.json")
    for i in range(3):
        _write_task(tmp_path / "evaluation" / f"eval_{i:03d}.json")
    return tmp_path


@pytest.fixture()
def datamodule(arc_data_dir: Path) -> ARCDataModule:
    dm = ARCDataModule(
        data_dir=str(arc_data_dir),
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        seed=0,
        max_size=16,
        num_color_permutations=0,
    )
    dm.setup()
    return dm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_eval_task_files_populated(datamodule: ARCDataModule) -> None:
    """_eval_task_files must be set and contain all evaluation tasks after setup()."""
    assert datamodule._eval_task_files is not None
    assert len(datamodule._eval_task_files) == 3


def test_eval_task_ids_offset(datamodule: ARCDataModule) -> None:
    """Evaluation task IDs must start after the last training task ID."""
    n_train = 5
    for task_id, _ in datamodule._eval_task_files:
        assert task_id >= n_train, f"Eval task_id {task_id} overlaps training range"


def test_train_uses_all_training_tasks(datamodule: ARCDataModule) -> None:
    """All 5 training tasks must appear in the training dataset (no holdout)."""
    seen_ids = {int(datamodule.train_dataset[i]["task_id"]) for i in range(len(datamodule.train_dataset))}
    assert len(seen_ids) == 5


def test_val_uses_eval_tasks(datamodule: ARCDataModule) -> None:
    """Val dataset must be drawn from evaluation-split tasks (task_id >= 5)."""
    for i in range(len(datamodule.val_dataset)):
        task_id = int(datamodule.val_dataset[i]["task_id"])
        assert task_id >= 5, f"Val sample has training task_id {task_id}"


def test_num_tasks(datamodule: ARCDataModule) -> None:
    """num_tasks must equal total unique task IDs across both splits."""
    assert datamodule.num_tasks == 8  # 5 training + 3 evaluation


def test_batch_format(datamodule: ARCDataModule) -> None:
    """on_before_batch_transfer must produce the standard {input, label, condition} layout."""
    loader = datamodule.train_dataloader()
    raw_batch = next(iter(loader))
    batch = datamodule.on_before_batch_transfer(raw_batch, dataloader_idx=0)

    assert set(batch.keys()) == {"input", "label", "condition"}
    assert set(batch["condition"].keys()) == {"task_id", "attention_mask"}

    b = batch["input"].shape[0]
    h = w = datamodule.max_size
    assert batch["input"].shape == (b, h, w)
    assert batch["label"].shape == (b, h, w)
    assert batch["condition"]["task_id"].shape == (b,)
    assert batch["condition"]["attention_mask"].shape == (b, h, w)


def test_input_values_in_range(datamodule: ARCDataModule) -> None:
    """Input pixels must be in [0, num_colors-1] ∪ {IGNORE_INDEX}; labels also allow PAD_INDEX."""
    sample = datamodule.train_dataset[0]
    inp = sample["input"]
    label = sample["label"]

    valid_input = (inp >= 0) & (inp <= IGNORE_INDEX)
    assert valid_input.all(), "Input contains out-of-range values"

    valid_label = (label >= 0) & (label <= PAD_INDEX)
    assert valid_label.all(), "Label contains out-of-range values"


def test_attention_mask_binary(datamodule: ARCDataModule) -> None:
    """Attention mask must be binary (0 or 1) and match input padding."""
    sample = datamodule.train_dataset[0]
    mask = sample["attention_mask"]
    assert ((mask == 0) | (mask == 1)).all()
    # Mask should be 1 exactly where input != IGNORE_INDEX
    assert (mask == (sample["input"] != IGNORE_INDEX).long()).all()
