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


"""CIFAR-10 DataModule backed by Hugging Face datasets (no Toronto mirror dependency)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from timm.data import Mixup
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from experiments.datamodules._deprecated.ref_imagenet import ThreeAugment


# Copied from dali_imagenet_fused to avoid the nvidia.dali import at module level.
@dataclass
class MixupConfig:
    """Mixup/CutMix strengths, probability, mode and label smoothing."""

    mixup: float = 0.0
    cutmix: float = 0.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"
    smoothing: float = 0.1


@dataclass
class AugmentConfig:
    """Optional ThreeAugment, colour jitter and RandAugment settings."""

    use_three_augment: bool = False
    color_jitter: float = 0.4
    rand_augment: Optional[str] = None
    random_erasing_prob: float = 0.0
    random_erasing_mode: str = "pixel"
    num_repeats: int = 1


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
CIFAR10_NUM_CLASSES = 10
CIFAR10_IMAGE_SIZE = 32
CIFAR10_TRAIN_SIZE = 50_000
HF_DATASET_NAME = "uoft-cs/cifar10"


class _CIFAR10HFDataset(Dataset):
    """Thin wrapper around a HuggingFace CIFAR-10 split."""

    def __init__(self, hf_split, transform: transforms.Compose, drop_labels: bool) -> None:
        self.data = hf_split
        self.transform = transform
        self.drop_labels = drop_labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.data[idx]
        image = self.transform(example["img"].convert("RGB"))
        label = torch.tensor(-1 if self.drop_labels else example["label"], dtype=torch.long)
        return image, label


class CIFAR10DataModule(pl.LightningDataModule):
    """Lightning DataModule for CIFAR-10 via HuggingFace datasets.

    Data is cached at ``data_dir``; pass the same path across runs.
    Images are returned channels-last (B, H, W, C) to match the pipeline.

    Args:
        data_dir: Hugging Face dataset cache directory.
        batch_size: Number of images per batch.
        num_workers: Number of DataLoader workers.
        pin_memory: Pin batch memory for device transfer.
        seed: Experiment seed metadata; loader/augmentation RNGs follow PyTorch's global seed.
        image_size: Random-crop size before optional resizing.
        final_image_size: Output spatial size; defaults to image_size.
        drop_labels: Replace class labels with -1.
        num_classes: Number of classes for Mixup/CutMix targets.
        task: Classification task identifier.
        mixup_cfg: Optional Mixup/CutMix and label-smoothing settings.
        augment_cfg: Optional ThreeAugment and RandAugment settings.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        seed: int = 42,
        image_size: int = CIFAR10_IMAGE_SIZE,
        final_image_size: Optional[int] = None,
        drop_labels: bool = False,
        num_classes: int = CIFAR10_NUM_CLASSES,
        task: Literal["classification"] = "classification",
        mixup_cfg: Optional[MixupConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None,
    ) -> None:
        """Configure the HF source, transforms and optional training mixup."""
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.image_size = image_size
        self.final_image_size = final_image_size or image_size
        self.drop_labels = drop_labels
        self.num_classes = num_classes
        self.output_channels = num_classes
        self.task = task

        if isinstance(mixup_cfg, (dict, DictConfig)):
            base = OmegaConf.structured(MixupConfig)
            self.mixup_cfg = OmegaConf.to_object(OmegaConf.merge(base, mixup_cfg))
        else:
            self.mixup_cfg = mixup_cfg

        if isinstance(augment_cfg, (dict, DictConfig)):
            base = OmegaConf.structured(AugmentConfig)
            self.augment_cfg = OmegaConf.to_object(OmegaConf.merge(base, augment_cfg))
        else:
            self.augment_cfg = augment_cfg

        self.mixup_fn: Optional[Mixup] = None
        if self.mixup_cfg is not None and (self.mixup_cfg.mixup > 0 or self.mixup_cfg.cutmix > 0):
            self.mixup_fn = Mixup(
                mixup_alpha=self.mixup_cfg.mixup,
                cutmix_alpha=self.mixup_cfg.cutmix,
                prob=self.mixup_cfg.mixup_prob,
                switch_prob=self.mixup_cfg.mixup_switch_prob,
                mode=self.mixup_cfg.mixup_mode,
                label_smoothing=self.mixup_cfg.smoothing,
                num_classes=num_classes,
            )

        self.train_dataset: Optional[_CIFAR10HFDataset] = None
        self.val_dataset: Optional[_CIFAR10HFDataset] = None

    def _build_transform(self, *, train: bool) -> transforms.Compose:
        ops: list = []

        if self.image_size != CIFAR10_IMAGE_SIZE:
            ops.append(transforms.Resize(self.image_size))

        if train:
            ops.append(transforms.RandomCrop(self.image_size, padding=4))
            ops.append(transforms.RandomHorizontalFlip())

            if self.augment_cfg is not None and self.augment_cfg.use_three_augment:
                ops.append(
                    transforms.ColorJitter(
                        brightness=self.augment_cfg.color_jitter,
                        contrast=self.augment_cfg.color_jitter,
                        saturation=self.augment_cfg.color_jitter,
                    )
                )
                ops.append(ThreeAugment())

            if self.augment_cfg is not None and self.augment_cfg.rand_augment:
                ops.append(
                    rand_augment_transform(
                        config_str=self.augment_cfg.rand_augment,
                        hparams={"img_mean": tuple(int(x * 255) for x in CIFAR10_MEAN)},
                    )
                )

        if self.final_image_size != self.image_size:
            ops.append(transforms.Resize(self.final_image_size))

        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD))
        return transforms.Compose(ops)

    def prepare_data(self) -> None:
        """Download/cache the CIFAR-10 train and test splits."""
        load_dataset(HF_DATASET_NAME, cache_dir=str(self.data_dir), split="train")
        load_dataset(HF_DATASET_NAME, cache_dir=str(self.data_dir), split="test")

    def setup(self, stage: Optional[str] = None) -> None:
        """Construct transforms and datasets for fit, validate or test.

        Args:
            stage: Lightning stage; None constructs both splits.
        """
        if stage in ("fit", None):
            ds = load_dataset(HF_DATASET_NAME, cache_dir=str(self.data_dir))
            self.train_dataset = _CIFAR10HFDataset(ds["train"], self._build_transform(train=True), self.drop_labels)
            self.val_dataset = _CIFAR10HFDataset(ds["test"], self._build_transform(train=False), self.drop_labels)
        elif stage in ("validate", "test"):
            ds = load_dataset(HF_DATASET_NAME, cache_dir=str(self.data_dir), split="test")
            self.val_dataset = _CIFAR10HFDataset(ds, self._build_transform(train=False), self.drop_labels)

    def _build_loader(self, dataset: _CIFAR10HFDataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
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
        """Return the train loader of image/label pairs."""
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Return the val loader of image/label pairs."""
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test loader of image/label pairs."""
        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def on_before_batch_transfer(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        dataloader_idx: int,
    ) -> dict:
        """Apply training mixup and convert images to channels-last batches.

        Args:
            batch: Images [B, C, H, W] and integer labels [B].
            dataloader_idx: Lightning loader index; unused.

        Returns:
            Dict with input [B, H, W, C], labels and condition=None.
        """
        images, labels = batch

        if self.mixup_fn is not None and self.trainer.training:
            images, labels = self.mixup_fn(images, labels)

        # (B, C, H, W) → (B, H, W, C) channels-last
        images = images.permute(0, 2, 3, 1).contiguous()

        if labels.ndim == 1:
            labels = labels.view(-1)

        return {"input": images, "label": labels, "condition": None}
