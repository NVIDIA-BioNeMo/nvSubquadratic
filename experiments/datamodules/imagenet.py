# TODO: Add license header here


"""ImageNet datamodule aligned with the MNIST loader semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# Pre-computed statistics for diffusion-ready 32x32 crops (10k sample estimate).
IMAGENET_MEAN_STD_BY_SIZE = {
    32: (
        [0.48450482, 0.45589244, 0.40366766],
        [0.25668961, 0.24765739, 0.26173702],
    ),
}

DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]


class _ImageNetDataset(Dataset):
    """Hugging Face backed dataset that mirrors torchvision's tuple output."""

    def __init__(
        self,
        *,
        split: str,
        dataset_name: str,
        dataset_config: Optional[str],
        cache_dir: Path,
        hf_token: Optional[str],
        transform: transforms.Compose,
        drop_labels: bool,
    ) -> None:
        super().__init__()
        self.transform = transform
        self.drop_labels = drop_labels

        self.dataset = load_dataset(
            path=dataset_name,
            name=dataset_config,
            split=split,
            streaming=False,
            cache_dir=str(cache_dir),
            token=hf_token,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.dataset[index]

        image = self.transform(example["image"].convert("RGB"))  # (num_channels, height, width)

        label_value = example.get("label", -1)

        label = torch.as_tensor(label_value, dtype=torch.long)  # (1,)

        if self.drop_labels:
            label = torch.full_like(label, fill_value=-1)  # (1,)

        return image, label


class ImageNetDataModule(pl.LightningDataModule):
    """Lightning DataModule that outputs MNIST-style dict batches."""

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int,
        image_size: int = 256,
        final_image_size: Optional[int] = None,
        center_crop: bool = True,
        drop_labels: bool = True,
        hf_dataset_name: str = "imagenet-1k",
        hf_dataset_config: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.image_size = image_size
        self.final_image_size = final_image_size or image_size
        self.center_crop = center_crop
        self.drop_labels = drop_labels
        self.hf_dataset_name = hf_dataset_name
        self.hf_dataset_config = hf_dataset_config
        self.hf_auth_token = hf_auth_token
        self.input_channels = 3
        self.num_classes = num_classes
        self.output_channels = self.input_channels if drop_labels else num_classes

        self.train_dataset: Optional[_ImageNetDataset] = None
        self.val_dataset: Optional[_ImageNetDataset] = None


    def _build_transform(self, *, train: bool) -> transforms.Compose:
        mean, std = IMAGENET_MEAN_STD_BY_SIZE.get(
            self.final_image_size,
            (DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD),
        )

        ops: list[transforms.Compose | transforms.RandomCrop | transforms.CenterCrop | transforms.Resize | transforms.RandomHorizontalFlip | transforms.ToTensor] = [
            transforms.Resize(self.image_size + 32),
        ]

        if train:
            ops.append(transforms.RandomCrop(self.image_size))

        else:
            if self.center_crop:
                ops.append(transforms.CenterCrop(self.image_size))

            else:
                ops.append(transforms.Resize(self.image_size))

        if train:
            ops.append(transforms.RandomHorizontalFlip())

        if self.final_image_size != self.image_size:
            ops.append(
                transforms.Resize(
                    self.final_image_size,
                    interpolation=InterpolationMode.BICUBIC,
                )
            )

        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        return transforms.Compose(ops)

    def prepare_data(self) -> None:  # pragma: no cover
        load_dataset(
            path=self.hf_dataset_name,
            name=self.hf_dataset_config,
            split="train",
            streaming=False,
            cache_dir=str(self.data_dir),
            token=self.hf_auth_token,
        )

        load_dataset(
            path=self.hf_dataset_name,
            name=self.hf_dataset_config,
            split="validation",
            streaming=False,
            cache_dir=str(self.data_dir),
            token=self.hf_auth_token,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_dataset = _ImageNetDataset(
                split="train",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=True),
                drop_labels=self.drop_labels,
            )

            self.val_dataset = _ImageNetDataset(
                split="validation",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=False),
                drop_labels=self.drop_labels,
            )

        elif stage == "validate":
            self.val_dataset = _ImageNetDataset(
                split="validation",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=False),
                drop_labels=self.drop_labels,
            )

        elif stage == "test":
            self.val_dataset = _ImageNetDataset(
                split="validation",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=False),
                drop_labels=self.drop_labels,
            )

    def _build_loader(self, dataset: _ImageNetDataset, shuffle: bool, drop_last: bool) -> DataLoader:
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
        if self.train_dataset is None:
            raise RuntimeError("train_dataloader called before setup('fit')")

        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("val_dataloader called before setup")

        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("test_dataloader called before setup('test')")

        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def on_before_batch_transfer(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        dataloader_idx: int,
    ) -> dict[str, torch.Tensor]:
        images, labels = batch

        images = images.permute(0, 2, 3, 1).contiguous()  # (bsize, height, width, num_channels)

        labels = labels.view(-1)  # (bsize,)

        return {
            "input": images,
            "label": labels,
            "condition": None,
        }
