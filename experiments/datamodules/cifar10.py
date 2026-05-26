"""CIFAR-10 Lightning DataModule.

Returns batches with the three-key format expected by ClassificationWrapper:
    {"input": [B, H, W, 3] float32 channels-last, "label": [B] long, "condition": None}

Standard augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize.
Optional Mixup/CutMix via timm (off by default for quick debug runs).
"""

from typing import Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from timm.data import Mixup
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 channel statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class _ChannelsLastCollate:
    """Collate CIFAR-10 (C, H, W) images into (B, H, W, C) channels-last tensors.

    Also injects the ``condition`` key (always None) expected by
    ClassificationWrapper.
    """

    def __init__(self, mixup_fn: Optional[Mixup] = None):
        self.mixup_fn = mixup_fn

    def __call__(self, batch):
        images, labels = zip(*batch)
        x = torch.stack(images)  # [B, C, H, W]
        y = torch.tensor(labels, dtype=torch.long)

        if self.mixup_fn is not None:
            x, y = self.mixup_fn(x, y)

        x = rearrange(x, "b c h w -> b h w c")  # channels-last
        return {"input": x, "label": y, "condition": None}


class CIFAR10DataModule(pl.LightningDataModule):
    """CIFAR-10 datamodule.

    Args:
        data_dir: Path where torchvision will download / find CIFAR-10.
        batch_size: Per-GPU batch size.
        num_workers: DataLoader workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
        image_size: Resize target.  CIFAR-10 is natively 32×32; resize only
            if you need a different resolution.
        mixup: Mixup alpha (0 = disabled).
        cutmix: CutMix alpha (0 = disabled).
        num_classes: Number of output classes (10).
    """

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 32,
        mixup: float = 0.0,
        cutmix: float = 0.0,
        num_classes: int = 10,
    ):
        """Initialize the CIFAR-10 datamodule (see class docstring for args)."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.num_classes = num_classes
        self.output_channels = num_classes

        # timm Mixup/CutMix (only active during training)
        use_mix = mixup > 0 or cutmix > 0
        self._mixup_fn = (
            Mixup(
                mixup_alpha=mixup,
                cutmix_alpha=cutmix,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                num_classes=num_classes,
            )
            if use_mix
            else None
        )

        train_tf = [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if image_size != 32:
            train_tf.insert(0, transforms.Resize(image_size))
        train_tf += [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
        self._train_transform = transforms.Compose(train_tf)

        val_tf = []
        if image_size != 32:
            val_tf.append(transforms.Resize(image_size))
        val_tf += [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
        self._val_transform = transforms.Compose(val_tf)

    def prepare_data(self):
        """Download train and test splits of CIFAR-10 if not already present."""
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Instantiate the train/val :class:`torchvision.datasets.CIFAR10` datasets."""
        self.train_ds = datasets.CIFAR10(self.data_dir, train=True, transform=self._train_transform)
        self.val_ds = datasets.CIFAR10(self.data_dir, train=False, transform=self._val_transform)

    def train_dataloader(self):
        """Return the training DataLoader with shuffling, drop_last, and optional Mixup/CutMix."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=_ChannelsLastCollate(self._mixup_fn),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        """Return the validation DataLoader (no shuffling, no Mixup, larger batch)."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            collate_fn=_ChannelsLastCollate(mixup_fn=None),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        """Return the test DataLoader — aliased to :meth:`val_dataloader` (see comment below)."""
        # CIFAR-10 has no separate held-out test split beyond the 10k test set,
        # which is already used for validation.  Reuse val_dataloader so that
        # trainer.test() reports the same test-set accuracy cleanly.
        return self.val_dataloader()
