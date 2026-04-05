"""Tests for ARCTTTValidationCallback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from experiments.callbacks.arc_ttt_validation import ARCTTTValidationCallback
from experiments.datamodules.arc import IGNORE_INDEX


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write_task(path: Path) -> None:
    """Write a minimal ARC task JSON with a 2×2 grid (fits any max_size >= 4)."""
    example = {"input": [[0, 1], [2, 3]], "output": [[1, 0], [3, 2]]}
    path.write_text(json.dumps({"train": [example], "test": [example]}))


@pytest.fixture()
def task_file(tmp_path: Path) -> tuple[int, Path]:
    p = tmp_path / "task_000.json"
    _write_task(p)
    return (0, p)


@pytest.fixture()
def many_task_files(tmp_path: Path) -> list[tuple[int, Path]]:
    """30 minimal task files for sampling tests."""
    files = []
    for i in range(30):
        p = tmp_path / f"task_{i:03d}.json"
        _write_task(p)
        files.append((i, p))
    return files


class _MinimalNet(nn.Module):
    """Minimal mock network that satisfies the TTT callback's interface.

    The logits are made to depend on task_token_embed so that loss.backward()
    has a valid gradient path (required for the TTT optimiser step).
    """

    NUM_COLORS = 12

    def __init__(self, num_tasks: int = 10) -> None:
        super().__init__()
        self.task_token_embed = nn.Embedding(num_tasks, 16)

    def forward(self, batch: dict) -> dict:
        b, h, w = batch["input"].shape
        task_ids = batch["condition"]["task_id"]
        # Route task token through the computation so gradients flow back to the embedding.
        token_bias = self.task_token_embed(task_ids).sum(dim=-1)  # [B]
        logits = token_bias[:, None, None, None].expand(b, self.NUM_COLORS, h, w)
        return {"logits": logits.contiguous()}


# ---------------------------------------------------------------------------
# _build_batch
# ---------------------------------------------------------------------------


def test_build_batch_shape(task_file: tuple[int, Path]) -> None:
    """_build_batch must return tensors with the expected shapes."""
    cb = ARCTTTValidationCallback()
    task_id, path = task_file
    task_json = json.loads(path.read_text())
    examples = task_json["train"]

    result = cb._build_batch(examples, task_id=task_id, device=torch.device("cpu"))
    assert result is not None

    b = len(examples)
    assert result["input"].shape == (b, 32, 32)
    assert result["label"].shape == (b, 32, 32)
    assert result["condition"]["task_id"].shape == (b,)
    assert result["condition"]["attention_mask"].shape == (b, 32, 32)


def test_build_batch_no_augmentation(task_file: tuple[int, Path]) -> None:
    """Grids must be placed at offset (1,1) with scale=1 (VARC TTT protocol)."""
    cb = ARCTTTValidationCallback()
    task_id, path = task_file
    examples = json.loads(path.read_text())["train"]

    result = cb._build_batch(examples, task_id=task_id, device=torch.device("cpu"))
    inp = result["input"][0]

    # The 2×2 grid should be placed at (1,1) — row 0 and col 0 are IGNORE_INDEX
    assert (inp[0, :] == IGNORE_INDEX).all(), "Top border should be IGNORE_INDEX"
    assert (inp[:, 0] == IGNORE_INDEX).all(), "Left border should be IGNORE_INDEX"


def test_build_batch_oversized_skipped(tmp_path: Path) -> None:
    """Examples whose grid exceeds max_size-2 must be skipped."""
    # max_size default is 32; a 31×31 grid can't fit (max_grid_dim = 30)
    big_example = {"input": [[0] * 31] * 31, "output": [[0] * 31] * 31}
    path = tmp_path / "big.json"
    path.write_text(json.dumps({"train": [big_example], "test": [big_example]}))

    cb = ARCTTTValidationCallback()
    result = cb._build_batch([big_example], task_id=0, device=torch.device("cpu"))
    assert result is None, "Oversized example should cause _build_batch to return None"


def test_build_batch_returns_none_for_empty(task_file: tuple[int, Path]) -> None:
    """_build_batch must return None when the example list is empty."""
    cb = ARCTTTValidationCallback()
    result = cb._build_batch([], task_id=0, device=torch.device("cpu"))
    assert result is None


# ---------------------------------------------------------------------------
# Fixed-seed task sampling
# ---------------------------------------------------------------------------


def test_fixed_seed_same_tasks_each_call(many_task_files: list[tuple[int, Path]]) -> None:
    """The same 20 tasks must be selected on every call (reproducible validation)."""
    cb = ARCTTTValidationCallback(ttt_val_tasks=20, seed=42)

    import random

    def _sample(files):
        rng = random.Random(cb.seed)
        return rng.sample(files, min(cb.ttt_val_tasks, len(files)))

    first = _sample(many_task_files)
    second = _sample(many_task_files)
    assert first == second, "Same seed must produce identical task samples"


def test_different_seeds_different_tasks(many_task_files: list[tuple[int, Path]]) -> None:
    """Different seeds should (almost certainly) produce different samples."""
    import random

    def _sample(seed):
        rng = random.Random(seed)
        return rng.sample(many_task_files, 20)

    assert _sample(0) != _sample(1)


# ---------------------------------------------------------------------------
# _run_ttt_task — token restoration
# ---------------------------------------------------------------------------


def test_run_ttt_task_restores_token(task_file: tuple[int, Path]) -> None:
    """Original task token must be restored after _run_ttt_task completes."""
    task_id, path = task_file
    net = _MinimalNet(num_tasks=10)
    original_weight = net.task_token_embed.weight[task_id].clone()

    cb = ARCTTTValidationCallback(ttt_steps=3)
    cb._run_ttt_task(net, task_id=task_id, task_path=path, device=torch.device("cpu"))

    assert torch.allclose(net.task_token_embed.weight[task_id], original_weight), (
        "task_token_embed weight must be restored to its original value after TTT"
    )


def test_run_ttt_task_returns_metrics(task_file: tuple[int, Path]) -> None:
    """_run_ttt_task must return a dict with exact_match and pixel_acc keys."""
    task_id, path = task_file
    net = _MinimalNet(num_tasks=10)

    cb = ARCTTTValidationCallback(ttt_steps=3)
    result = cb._run_ttt_task(net, task_id=task_id, task_path=path, device=torch.device("cpu"))

    assert result is not None
    assert "exact_match" in result
    assert "pixel_acc" in result
    assert 0.0 <= result["exact_match"] <= 1.0
    assert 0.0 <= result["pixel_acc"] <= 1.0


def test_run_ttt_task_other_tokens_unchanged(task_file: tuple[int, Path]) -> None:
    """TTT must not modify task tokens belonging to other tasks."""
    task_id, path = task_file
    net = _MinimalNet(num_tasks=10)
    other_id = 5  # different from task_id=0
    original_other = net.task_token_embed.weight[other_id].clone()

    cb = ARCTTTValidationCallback(ttt_steps=3)
    cb._run_ttt_task(net, task_id=task_id, task_path=path, device=torch.device("cpu"))

    assert torch.allclose(net.task_token_embed.weight[other_id], original_other), (
        "TTT must not modify token embeddings of other tasks"
    )
