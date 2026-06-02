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

"""Test DALI repeated-augmentation epoch transitions.

Background
----------
DALI's ``fn.external_source`` with ``parallel=True`` spawns worker
processes that each hold an independent copy of the callable source.
When workers raise ``StopIteration`` to signal end-of-epoch, DALI kills
them; on pipeline reset for the next epoch, it must re-spawn workers,
which fails with ``StopIteration`` leaking through the generator wrapper
(PEP 479 → ``RuntimeError``), or — if caught — silently returns empty
epochs forever.

The fix avoids ``StopIteration`` entirely:

1. ``_RepeatedAugSource.__call__`` wraps around with modular indexing
   instead of raising ``StopIteration``.  Epoch length is controlled by
   ``DALIGenericIterator(size=num_selected)``, not by the source.
2. ``fn.external_source`` is called with ``cycle="quiet"`` so DALI
   never expects the source to signal end-of-data.
3. Epoch reshuffling is triggered by checking ``sample_info.epoch_idx``.

The test classes are:

* **TestRepeatedAugSource** — CPU-only unit tests for the callable source:
  wrap-around indexing, epoch reshuffling, and boundary behavior.
* **TestDALIRepeatedAugEndToEnd** — GPU tests (skipped without CUDA/DALI):
  builds a real DALI pipeline **with ``parallel=True``** (matching
  production) and iterates through 3 epochs, verifying correct batch
  counts for both ``num_repeats=3`` and ``num_repeats=1`` code paths.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.datamodules.dali_imagenet_fused import (
    _DALILoaderWrapper,
    _RepeatedAugSource,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_imagefolder(root: Path, num_classes: int = 3, imgs_per_class: int = 10):
    """Create a minimal ImageFolder tree with tiny 4x4 JPEG files.

    The resulting directory has the standard structure::

        root/class_000/img_0000.jpeg
        root/class_000/img_0001.jpeg
        ...
        root/class_001/img_0000.jpeg
    """
    for c in range(num_classes):
        class_dir = root / f"class_{c:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(imgs_per_class):
            img = Image.fromarray(np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8))
            img.save(class_dir / f"img_{i:04d}.jpeg")


@dataclass
class FakeSampleInfo:
    """Mimics ``nvidia.dali.types.SampleInfo`` for CPU-only unit tests.

    DALI passes a ``SampleInfo`` to ``external_source`` callables.  We only
    need the four fields that ``_RepeatedAugSource.__call__`` inspects.
    """

    idx_in_epoch: int
    idx_in_batch: int
    iteration: int
    epoch_idx: int


# ---------------------------------------------------------------------------
# Tests for _RepeatedAugSource
# ---------------------------------------------------------------------------


class TestRepeatedAugSource:
    """CPU-only unit tests for ``_RepeatedAugSource``.

    ``_RepeatedAugSource`` is the per-sample callable passed to DALI's
    ``fn.external_source(parallel=True)``.  It implements DeiT-style
    repeated augmentation by repeat-interleaving shuffled dataset indices
    and sharding across GPUs.

    The source **never** raises ``StopIteration``.  Instead it wraps
    around with modular indexing — epoch length is controlled by the
    ``DALIGenericIterator(size=num_selected)`` wrapper, not the source.
    """

    @pytest.fixture()
    def tiny_folder(self, tmp_path):
        """Create an ImageFolder with 300 images (10 classes x 30).

        ``_RepeatedAugSource`` computes ``num_selected = floor(N // 256 * 256 / W)``,
        so we need N >= 256 for any samples to be selected.
        """
        _make_tiny_imagefolder(tmp_path, num_classes=10, imgs_per_class=30)
        return str(tmp_path)

    def test_epoch0_returns_data(self, tiny_folder):
        """The first sample of epoch 0 returns valid (jpeg_bytes, label)."""
        src = _RepeatedAugSource(tiny_folder, num_repeats=3, shard_id=0, num_shards=1)
        info = FakeSampleInfo(idx_in_epoch=0, idx_in_batch=0, iteration=0, epoch_idx=0)
        jpeg, label = src(info)
        assert isinstance(jpeg, np.ndarray)
        assert label.shape == (1,)

    def test_wraps_around_past_boundary(self, tiny_folder):
        """Requesting ``idx_in_epoch >= num_selected`` wraps around (no StopIteration).

        The ``DALIGenericIterator(size=num_selected)`` stops the epoch;
        the source itself must stay alive for the parallel workers.
        """
        src = _RepeatedAugSource(tiny_folder, num_repeats=3, shard_id=0, num_shards=1)
        n = src.num_selected

        # Request one past the boundary — should wrap to index 0, not crash
        info = FakeSampleInfo(idx_in_epoch=n, idx_in_batch=0, iteration=0, epoch_idx=0)
        jpeg, label = src(info)
        assert isinstance(jpeg, np.ndarray)

        # Same data as idx_in_epoch=0
        info_first = FakeSampleInfo(idx_in_epoch=0, idx_in_batch=0, iteration=0, epoch_idx=0)
        _jpeg_first, label_first = src(info_first)
        np.testing.assert_array_equal(label, label_first)

    def test_epoch_transition_reshuffles(self, tiny_folder):
        """When ``epoch_idx`` advances, indices are reshuffled with a new seed.

        The reshuffle uses ``seed + epoch_idx``, so epoch 0 and epoch 1
        must produce different orderings.
        """
        src = _RepeatedAugSource(tiny_folder, num_repeats=3, shard_id=0, num_shards=1)
        indices_epoch0 = list(src._indices)

        info = FakeSampleInfo(idx_in_epoch=0, idx_in_batch=0, iteration=100, epoch_idx=1)
        src(info)
        indices_epoch1 = list(src._indices)

        assert src._epoch == 1
        assert len(indices_epoch0) == len(indices_epoch1)
        assert indices_epoch0 != indices_epoch1, "Indices should differ between epochs"

    def test_epoch_transition_after_boundary(self, tiny_folder):
        """After wrapping past the boundary, the next epoch's ``idx_in_epoch=0`` succeeds.

        Regression test: before the fix, the source raised ``StopIteration``
        at the boundary, which killed parallel workers and broke resets.
        """
        src = _RepeatedAugSource(tiny_folder, num_repeats=3, shard_id=0, num_shards=1)

        # Go past boundary in epoch 0 (simulating DALI over-fetching)
        info_past = FakeSampleInfo(idx_in_epoch=src.num_selected + 5, idx_in_batch=0, iteration=0, epoch_idx=0)
        jpeg, _ = src(info_past)
        assert isinstance(jpeg, np.ndarray)

        # First sample of epoch 1 must work
        info_new = FakeSampleInfo(idx_in_epoch=0, idx_in_batch=0, iteration=0, epoch_idx=1)
        jpeg, _label = src(info_new)
        assert isinstance(jpeg, np.ndarray)

    def test_multiple_epoch_transitions(self, tiny_folder):
        """Source correctly reshuffles across 5 consecutive epoch transitions.

        Each epoch should produce a distinct ordering (different seed).
        """
        src = _RepeatedAugSource(tiny_folder, num_repeats=3, shard_id=0, num_shards=1)
        prev_indices = list(src._indices)

        for epoch in range(1, 5):
            info = FakeSampleInfo(idx_in_epoch=0, idx_in_batch=0, iteration=0, epoch_idx=epoch)
            src(info)
            assert src._epoch == epoch
            curr_indices = list(src._indices)
            assert curr_indices != prev_indices
            prev_indices = curr_indices


# ---------------------------------------------------------------------------
# End-to-end DALI pipeline test (requires GPU + DALI)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DALI end-to-end test requires GPU",
)
class TestDALIRepeatedAugEndToEnd:
    """GPU integration tests: build a real DALI pipeline and iterate 3 epochs.

    These tests exercise the full data-loading path that
    ``DALIImageNetFusedDataModule`` uses in production, **including
    ``parallel=True``** for the external_source path.  A tiny synthetic
    ImageFolder keeps them fast (~5 s).

    Two code paths exist in ``train_dataloader()``:

    * **``num_repeats > 1``** — uses ``fn.external_source(source=...,
      parallel=True, cycle="quiet")`` with ``size=num_selected``.
    * **``num_repeats == 1``** — uses ``fn.readers.file`` with
      ``reader_name="reader"``.

    Both are tested over 3 epochs (not just 2) to confirm stable cycling.
    """

    @pytest.fixture()
    def tiny_folder(self, tmp_path):
        """300 images (3 classes x 100) — enough for ``num_selected > 0``."""
        _make_tiny_imagefolder(tmp_path, num_classes=3, imgs_per_class=100)
        return str(tmp_path)

    def test_three_epoch_repeated_aug_parallel(self, tiny_folder):
        """``external_source(parallel=True, cycle="quiet")`` path survives 3 epochs.

        This matches the production configuration.  Before the fix, epoch 1
        would either crash (``RuntimeError`` from PEP 479) or silently
        return empty epochs forever.

        With 300 images, ``num_selected = floor(300 // 256 * 256 / 1) = 256``.
        At batch_size=8 with DROP policy: ``256 // 8 = 32`` batches/epoch.
        """
        try:
            from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
        except ImportError:
            pytest.skip("nvidia.dali not installed")

        from experiments.datamodules.dali_imagenet_fused import _train_pipeline_fused

        num_repeats = 3
        batch_size = 8

        ra_source = _RepeatedAugSource(
            file_root=tiny_folder,
            num_repeats=num_repeats,
            shard_id=0,
            num_shards=1,
        )
        expected_batches = ra_source.num_selected // batch_size

        pipe = _train_pipeline_fused(
            file_root=tiny_folder,
            image_size=32,
            final_image_size=32,
            norm_mean=(0.485, 0.456, 0.406),
            norm_std=(0.229, 0.224, 0.225),
            batch_size=batch_size,
            num_threads=2,
            device_id=0,
            seed=42,
            ra_source=ra_source,
            py_start_method="spawn",
        )
        pipe.build()

        dali_iter = DALIGenericIterator(
            pipe,
            output_map=["images", "labels"],
            size=ra_source.num_selected,
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )
        wrapper = _DALILoaderWrapper(dali_iter)

        for epoch in range(3):
            count = 0
            total_samples = 0
            for images, labels in wrapper:
                count += 1
                total_samples += images.shape[0]
            assert count == expected_batches, f"Epoch {epoch}: expected {expected_batches} batches, got {count}"
            assert total_samples == expected_batches * batch_size

    def test_three_epoch_no_repeated_aug(self, tiny_folder):
        """``fn.readers.file`` path (num_repeats=1) iterates all images across 3 epochs.

        Sanity check that the cycle="quiet" change doesn't affect the
        standard reader-based path.

        With 300 images at batch_size=8 and DROP policy: ``300 // 8 = 37``
        batches/epoch (296 images served, 4 dropped).
        """
        try:
            from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
        except ImportError:
            pytest.skip("nvidia.dali not installed")

        from experiments.datamodules.dali_imagenet_fused import _train_pipeline_fused

        num_images = 300  # 3 classes x 100
        batch_size = 8
        expected_batches = num_images // batch_size  # 37

        pipe = _train_pipeline_fused(
            file_root=tiny_folder,
            image_size=32,
            final_image_size=32,
            norm_mean=(0.485, 0.456, 0.406),
            norm_std=(0.229, 0.224, 0.225),
            batch_size=batch_size,
            num_threads=2,
            device_id=0,
            seed=42,
            ra_source=None,
        )
        pipe.build()

        dali_iter = DALIGenericIterator(
            pipe,
            output_map=["images", "labels"],
            reader_name="reader",
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        )
        wrapper = _DALILoaderWrapper(dali_iter)

        for epoch in range(3):
            count = 0
            total_samples = 0
            for images, labels in wrapper:
                count += 1
                total_samples += images.shape[0]
            assert count == expected_batches, f"Epoch {epoch}: expected {expected_batches} batches, got {count}"
            assert total_samples == expected_batches * batch_size
