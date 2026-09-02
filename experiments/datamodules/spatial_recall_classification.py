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

"""Classification variant of the spatial recall task.

Extends :class:`~experiments.datamodules.spatial_recall_dataset.SpatialRecallDataset`
to return the integer class label from the base dataset instead of the pixel
readout region.  This is the correct label for
:class:`~experiments.lightning_wrappers.classification_wrapper.ClassificationWrapper`.

The regression-flavoured :class:`~experiments.datamodules.spatial_recall_dataset.SpatialRecallDataModule`
discards the base-dataset class label and returns a pixel crop as the target
(for copy / recall losses).  Classification requires the integer digit class
(0–9 for MNIST) instead.

Design:
    ``SpatialRecallClassificationDataset`` is a thin subclass that overrides
    ``__getitem__`` to return ``(canvas, class_label)`` where ``class_label``
    is the long-integer class from the base dataset.

    ``SpatialRecallClassificationDataModule`` mirrors the structure of
    :class:`~experiments.datamodules.spatial_recall_dataset.SpatialRecallDataModule`
    but wires up ``SpatialRecallClassificationDataset`` and converts the batch
    to ``{"input": [B, H, W, C], "label": [B], "condition": None}``.

Usage::

    from experiments.datamodules.spatial_recall_classification import (
        SpatialRecallClassificationDataModule,
    )
    from experiments.datamodules.mnist import MNISTDataModule
    from nvsubquadratic.lazy_config import LazyConfig

    dm = SpatialRecallClassificationDataModule(
        base_datamodule_cfg=LazyConfig(MNISTDataModule)(
            data_dir=".data/mnist",
            batch_size=128,
            data_type="image",
            num_workers=4,
            pin_memory=True,
            use_deterministic_worker_init=False,
            seed=42,
            task="classification",
        ),
        target_size=14,
        canvas_size=56,
        placement="fixed",
        with_mask=False,
        readout_value=-1.0,
    )
"""

from typing import Literal, Tuple

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from experiments.datamodules.spatial_recall_dataset import SpatialRecallDataset
from nvsubquadratic.lazy_config import instantiate


class SpatialRecallClassificationDataset(SpatialRecallDataset):
    """Spatial recall canvas paired with the base dataset's integer class label.

    Drop-in replacement for
    :class:`~experiments.datamodules.spatial_recall_dataset.SpatialRecallDataset`
    that returns ``(canvas [C, H, W], class_label: int)`` instead of
    ``(canvas, pixel_readout_region)``.  Everything else — canvas construction,
    placement logic, readout-region masking — is inherited unchanged.

    Args:
        base_dataset: Base dataset providing ``(image [C, H, W], class_label)``
            pairs.  ``class_label`` must be an integer (e.g. MNIST 0–9).
        target_size: Side length of the digit patch after rescaling.
        canvas_size: Side length of the output square canvas.
        generator: :class:`torch.Generator` for reproducible random placement.
        placement: ``"fixed"`` (top-left corner) or ``"random"``.
        with_mask: If ``True``, append a binary mask channel marking the digit
            position to the canvas.
        readout_value: Scalar value used to fill the bottom-right readout region
            of the canvas.  Set to ``-1.0`` to give the model an explicit
            spatial cue; ``0.0`` means no filling.
    """

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Return ``(canvas [C, H, W], class_label)`` for the given index.

        The canvas is constructed by the parent class logic (placement,
        readout region masking, optional mask channel).  The class label is
        the integer target from the base dataset, not a pixel crop.

        Args:
            idx: Dataset index.

        Returns:
            Tuple of (canvas tensor ``[C, H, W]``, integer class label).
        """
        # Access class label before the parent discards it.
        _, class_label = self.base_dataset[idx]
        # Parent builds the canvas (discarding base label internally).
        canvas, _ = super().__getitem__(idx)
        return canvas, int(class_label)


class SpatialRecallClassificationDataModule(pl.LightningDataModule):
    """Lightning DataModule for the classification spatial recall task.

    Wraps ``SpatialRecallClassificationDataset`` around a base datamodule
    (MNIST, EMNIST, …) and returns batches as::

        {
            "input":     [B, H, W, C],   # channels-last canvas
            "label":     [B],             # long integer class labels (0–9)
            "condition": None,
        }

    This format is consumed directly by
    :class:`~experiments.lightning_wrappers.classification_wrapper.ClassificationWrapper`.

    Args:
        base_datamodule_cfg: LazyConfig for the base datamodule (e.g.
            :class:`~experiments.datamodules.mnist.MNISTDataModule`).  The
            base datamodule must expose ``train_dataset``, ``val_dataset``, and
            optionally ``test_dataset`` attributes after ``setup()``.
        target_size: Side length of the digit patch.
        canvas_size: Side length of the square canvas.  Must satisfy
            ``canvas_size >= 2 * target_size`` when ``placement="random"``.
        placement: ``"fixed"`` or ``"random"``.
        with_mask: Append a binary location-mask channel to the canvas.
        readout_value: Value to fill the bottom-right readout corner.
        batch_size: Batch size.  Defaults to the base datamodule's
            ``batch_size`` if the base config exposes it; pass explicitly to
            override.
        num_workers: DataLoader worker count.
        pin_memory: Pin memory for CUDA transfers.
        seed: Base seed; generators for train/val/test are offset by 0/1000/2000.
    """

    def __init__(
        self,
        base_datamodule_cfg,
        target_size: int,
        canvas_size: int,
        placement: Literal["fixed", "random"] = "fixed",
        with_mask: bool = False,
        readout_value: float = -1.0,
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        """Initialise and store configuration; datasets created in ``setup``."""
        super().__init__()
        self._base_dm_cfg = base_datamodule_cfg
        self.target_size = target_size
        self.canvas_size = canvas_size
        self.placement = placement
        self.with_mask = with_mask
        self.readout_value = readout_value
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self._base_dm = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        """Delegate to the base datamodule for data download."""
        dm = instantiate(self._base_dm_cfg)
        dm.prepare_data()

    def setup(self, stage=None) -> None:
        """Instantiate base datamodule and wrap datasets with classification variant."""
        self._base_dm = instantiate(self._base_dm_cfg)
        self._base_dm.setup(stage)

        def _wrap(base_ds, seed_offset: int):
            return SpatialRecallClassificationDataset(
                base_dataset=base_ds,
                target_size=self.target_size,
                canvas_size=self.canvas_size,
                generator=torch.Generator().manual_seed(self.seed + seed_offset),
                placement=self.placement,
                with_mask=self.with_mask,
                readout_value=self.readout_value,
            )

        if stage in ("fit", None):
            self.train_dataset = _wrap(self._base_dm.train_dataset, seed_offset=0)
            self.val_dataset = _wrap(self._base_dm.val_dataset, seed_offset=1000)
        if stage in ("test", None) and self._base_dm.test_dataset is not None:
            self.test_dataset = _wrap(self._base_dm.test_dataset, seed_offset=2000)

    def _build_loader(self, dataset: Dataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return self._build_loader(self.test_dataset, shuffle=False)

    def on_before_batch_transfer(self, batch, dataloader_idx) -> dict:
        """Convert ``(canvas, label)`` batch to channels-last dict format.

        Transforms the canvas from ``[B, C, H, W]`` (PyTorch default
        channels-first) to ``[B, H, W, C]`` (channels-last, expected by
        ViT-5 networks).  Class labels are stacked into a ``[B]`` long tensor.

        Args:
            batch: Tuple ``(canvas [B, C, H, W], label [B])``.
            dataloader_idx: Ignored.

        Returns:
            Dict with keys ``"input" [B, H, W, C]``, ``"label" [B]``,
            ``"condition" None``.
        """
        x, y = batch
        x = rearrange(x, "b c h w -> b h w c")
        return {"input": x, "label": y, "condition": None}
