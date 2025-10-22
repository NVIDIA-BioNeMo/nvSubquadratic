# TODO: Add license header here


"""LightningDataModule exposing ImageNet through a shared Hugging Face cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from examples.imagenet_diffusion import deterministic_utils


@dataclass
class ImageNetTransforms:
    """Container bundling the preprocessing pipelines for ImageNet."""

    train: Callable
    eval: Callable


class _ImageNetDataset(Dataset):
    """Hugging Face-backed ImageNet dataset that materializes locally."""

    def __init__(
        self,
        *,
        split: str,
        transforms_fn: Callable,
        dataset_name: str,
        dataset_config: Optional[str],
        cache_dir: Path,
        drop_labels: bool,
        hf_token: Optional[str],
    ) -> None:
        super().__init__()
        self.transforms_fn = transforms_fn
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
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.dataset[index]
        image = self.transforms_fn(example["image"].convert("RGB"))
        # Convert to channels-last so diffusion network sees feature dimension last.
        if image.ndim == 3:
            image = image.permute(1, 2, 0).contiguous()
        output = {"input": image}
        if not self.drop_labels and "label" in example:
            output["label"] = torch.as_tensor(example["label"], dtype=torch.long)
        return output


class ImageNetDataModule(pl.LightningDataModule):
    """LightningDataModule exposing ImageNet via a shared Hugging Face cache."""

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int,
        image_size: int = 256,
        center_crop: bool = True,
        drop_labels: bool = True,
        use_deterministic_worker_init: bool = True,
        hf_dataset_name: str = "imagenet-1k",
        hf_dataset_config: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.image_size = image_size
        self.center_crop = center_crop
        self.drop_labels = drop_labels
        self.use_deterministic_worker_init = use_deterministic_worker_init
        self.hf_dataset_name = hf_dataset_name
        self.hf_dataset_config = hf_dataset_config
        self.hf_auth_token = hf_auth_token

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.worker_init_fn = (
            deterministic_utils.worker_init_fn if self.use_deterministic_worker_init else None
        )

        # Model IO metadata consumed downstream by example runner.
        self.input_channels = 3
        self.output_channels = 3

        self._transforms: Optional[ImageNetTransforms] = None
        self._train_dataset: Optional[_ImageNetDataset] = None
        self._val_dataset: Optional[_ImageNetDataset] = None

        deterministic_utils.set_base_seed(self.seed)

    def prepare_data(self) -> None:  # pragma: no cover - nothing to download eagerly.
        """Trigger a one-off download of the ImageNet splits."""
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

    def _build_transforms(self) -> ImageNetTransforms:
        resize_op = transforms.Resize(self.image_size + 32)
        crop_train = transforms.RandomCrop(self.image_size)
        crop_eval = transforms.CenterCrop(self.image_size) if self.center_crop else transforms.Resize(self.image_size)
        common_tail = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        train_transforms = transforms.Compose([resize_op, crop_train, transforms.RandomHorizontalFlip(), *common_tail])
        eval_transforms = transforms.Compose([resize_op, crop_eval, *common_tail])
        return ImageNetTransforms(train=train_transforms, eval=eval_transforms)

    def setup(self, stage: Optional[str] = None) -> None:
        if self._transforms is None:
            self._transforms = self._build_transforms()

        def _create_dataset(split: str, transforms_fn: Callable) -> _ImageNetDataset:
            return _ImageNetDataset(
                split=split,
                transforms_fn=transforms_fn,
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                drop_labels=self.drop_labels,
                hf_token=self.hf_auth_token,
            )

        if stage in ("fit", None):
            self._train_dataset = _create_dataset("train", self._transforms.train)
            self._val_dataset = _create_dataset("validation", self._transforms.eval)
        elif stage == "validate":
            self._val_dataset = _create_dataset("validation", self._transforms.eval)
        elif stage == "test":
            self._val_dataset = _create_dataset("validation", self._transforms.eval)

    def _build_dataloader(self, dataset: _ImageNetDataset, *, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            worker_init_fn=self.worker_init_fn,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("train_dataloader called before setup('fit')")
        return self._build_dataloader(self._train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("val_dataloader called before setup")
        return self._build_dataloader(self._val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("test_dataloader called before setup")
        return self._build_dataloader(self._val_dataset, shuffle=False, drop_last=False)
