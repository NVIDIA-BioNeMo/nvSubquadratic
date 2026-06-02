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

"""Verify DataLoader parameters between our WellDataModule and the upstream BaseWellDataModule.

Compares every user-facing DataLoader attribute across all five loader methods
(train, val, test, rollout_val, rollout_test) and reports matches/mismatches.

This test does NOT require the actual dataset on disk — it mocks the
underlying datasets so the DataLoaders can be constructed without I/O.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from torch.utils.data import DataLoader, Dataset


BATCH_SIZE = 24
NUM_WORKERS = 12

# ── Attributes we compare between the two DataLoaders ─────────────────────
COMPARED_ATTRS = [
    "batch_size",
    "num_workers",
    "pin_memory",
    "drop_last",
    "prefetch_factor",
    "persistent_workers",
]


def _get_shuffle(loader: DataLoader) -> bool:
    """Infer whether a DataLoader shuffles from its sampler type."""
    from torch.utils.data import RandomSampler, SequentialSampler

    sampler = loader.sampler
    if isinstance(sampler, RandomSampler):
        return True
    if isinstance(sampler, SequentialSampler):
        return False
    return None


class _FakeDataset(Dataset):
    """Tiny fake dataset so DataLoader construction succeeds."""

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        return idx


def _make_fake_base_datamodule():
    """Return a mock BaseWellDataModule with fake datasets attached."""
    base = MagicMock()
    base.train_dataset = _FakeDataset()
    base.val_dataset = _FakeDataset()
    base.test_dataset = _FakeDataset()
    base.rollout_val_dataset = _FakeDataset()
    base.rollout_test_dataset = _FakeDataset()
    base.data_workers = NUM_WORKERS
    base.batch_size = BATCH_SIZE
    base.world_size = 1
    base.rank = 1

    metadata = SimpleNamespace(
        n_fields=2,
        n_constant_fields=0,
    )
    base.train_dataset.metadata = metadata
    base.train_dataset.norm = MagicMock()
    base.train_dataset.use_normalization = True

    return base


def _build_ours():
    """Build our WellDataModule with fake datasets (skip real I/O)."""
    from experiments.datamodules.pde.well import WellDataModule

    dm = WellDataModule(
        well_base_path="/fake",
        well_dataset_name="fake_dataset",
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_normalization=False,
        prefetch_factor=4,
        persistent_workers=True,
    )
    dm._well_datamodule = _make_fake_base_datamodule()
    return dm


def _build_upstream():
    """Build the upstream BaseWellDataModule with fake datasets (skip real I/O)."""
    base = _make_fake_base_datamodule()

    from the_well.data.datamodule import WellDataModule as BaseWellDataModule

    upstream = BaseWellDataModule.__new__(BaseWellDataModule)
    upstream.train_dataset = base.train_dataset
    upstream.val_dataset = base.val_dataset
    upstream.test_dataset = base.test_dataset
    upstream.rollout_val_dataset = base.rollout_val_dataset
    upstream.rollout_test_dataset = base.rollout_test_dataset
    upstream.data_workers = NUM_WORKERS
    upstream.batch_size = BATCH_SIZE
    upstream.world_size = 1
    upstream.rank = 1
    return upstream


LOADER_METHODS = ["train_dataloader", "val_dataloader", "test_dataloader"]
LOADER_PROPERTIES = ["rollout_val_dataloader", "rollout_test_dataloader"]


def _get_loader(obj, name):
    attr = getattr(obj, name)
    return attr() if callable(attr) else attr


@pytest.fixture(scope="module")
def loaders():
    """Build both sets of DataLoaders once for all tests."""
    ours_dm = _build_ours()
    upstream_dm = _build_upstream()

    result = {}
    for name in LOADER_METHODS + LOADER_PROPERTIES:
        ours_loader = _get_loader(ours_dm, name)
        upstream_loader = _get_loader(upstream_dm, name)
        result[name] = (ours_loader, upstream_loader)
    return result


# ── Parametrized comparison across all loaders and attributes ─────────────


@pytest.mark.parametrize("loader_name", LOADER_METHODS + LOADER_PROPERTIES)
def test_dataloader_comparison(loaders, loader_name):
    """Print a full comparison table for a given loader, fail on unexpected mismatches."""
    ours_loader, upstream_loader = loaders[loader_name]

    ours_shuffle = _get_shuffle(ours_loader)
    upstream_shuffle = _get_shuffle(upstream_loader)

    rows = []
    for attr in COMPARED_ATTRS:
        ours_val = getattr(ours_loader, attr, "N/A")
        upstream_val = getattr(upstream_loader, attr, "N/A")
        match = "✓" if ours_val == upstream_val else "✗"
        rows.append((attr, ours_val, upstream_val, match))

    rows.append(("shuffle", ours_shuffle, upstream_shuffle, "✓" if ours_shuffle == upstream_shuffle else "✗"))

    header = f"\n{'─' * 70}\n  {loader_name}\n{'─' * 70}"
    lines = [header, f"  {'Attribute':<25} {'Ours':<15} {'Upstream':<15} {'Match'}"]
    lines.append(f"  {'─' * 25} {'─' * 15} {'─' * 15} {'─' * 5}")
    for attr, ours_val, upstream_val, match in rows:
        lines.append(f"  {attr:<25} {str(ours_val):<15} {str(upstream_val):<15} {match}")

    print("\n".join(lines))

    # ── Intentional differences we expect ────────────────────────────
    # We intentionally diverge from upstream on these:
    #   - persistent_workers: True (upstream: False) — avoid worker respawn overhead
    #   - prefetch_factor: 4 (upstream: None/2) — keep GPU fed
    #   - val/test shuffle: False (upstream: True for val) — reproducible evaluation
    #   - val/test drop_last: False (upstream: True) — evaluate all samples
    #
    # Any OTHER mismatch is unexpected and should be investigated.

    EXPECTED_IMPROVEMENTS = {"persistent_workers", "prefetch_factor"}

    EXPECTED_FIXES = set()
    is_eval = loader_name in (
        "val_dataloader",
        "test_dataloader",
        "rollout_val_dataloader",
        "rollout_test_dataloader",
    )
    if is_eval:
        EXPECTED_FIXES |= {"shuffle", "drop_last"}

    allowed_mismatches = EXPECTED_IMPROVEMENTS | EXPECTED_FIXES

    unexpected = []
    for attr, ours_val, upstream_val, match in rows:
        if match == "✗" and attr not in allowed_mismatches:
            unexpected.append(f"{attr}: ours={ours_val}, upstream={upstream_val}")

    if unexpected:
        pytest.fail(
            f"Unexpected dataloader parameter mismatches in {loader_name}:\n"
            + "\n".join(f"  - {u}" for u in unexpected)
        )
