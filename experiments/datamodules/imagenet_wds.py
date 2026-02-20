# TODO: Add license header here

"""WebDataset-backed ImageNet DataModule for Lightning.

This module provides ``ImageNetWebDataModule``, a drop-in replacement for
``ImageNetDataModule`` that reads from WebDataset TAR shards instead of
HuggingFace Arrow files.  Sequential TAR reads are dramatically faster
over NFS (the original Arrow random I/O achieved only 0.10 it/s on 8×
A5000; WebDataset should reach 3-4 it/s).

The shards are produced by ``scripts/convert_imagenet_to_webdataset.py``.
Expected layout::

    data/imagenet-wds/
    ├── train/
    │   ├── imagenet-train-000000.tar
    │   ├── imagenet-train-000001.tar
    │   ├── ...
    │   └── meta.json          # {"num_samples": 1281167, "num_shards": ...}
    └── validation/
        ├── imagenet-validation-000000.tar
        ├── ...
        └── meta.json          # {"num_samples": 50000, "num_shards": ...}
"""

import io
import json
import math
from pathlib import Path
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from timm.data import Mixup
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from experiments.datamodules.imagenet import (
    DEFAULT_IMAGENET_MEAN,
    DEFAULT_IMAGENET_STD,
    IMAGENET_MEAN_STD_BY_SIZE,
    AugmentConfig,
    MixupConfig,
    ThreeAugment,
)


def _decode_image(sample: dict) -> dict:
    """Decode JPEG bytes to a PIL Image."""
    sample["jpg"] = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
    return sample


def _decode_label(sample: dict) -> dict:
    """Decode label bytes to an integer."""
    # WebDataset stores cls as raw bytes of the string representation
    sample["cls"] = int(sample["cls"])
    return sample


class ImageNetWebDataModule(pl.LightningDataModule):
    """Lightning DataModule backed by WebDataset TAR shards.

    This is API-compatible with ``ImageNetDataModule`` — it produces the
    same ``{"input": ..., "label": ..., "condition": None}`` dict batches
    and supports the same augmentation / mixup configuration.
    """

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
        num_classes: int = 1000,
        task: Literal["classification", "generation"],
        # Augmentations
        mixup_cfg: Optional[MixupConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None,
        # WebDataset-specific — not used, kept for API compat
        hf_dataset_name: str = "ILSVRC/imagenet-1k",
        hf_dataset_config: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
    ) -> None:
        """Initialize the WebDataset-backed ImageNet datamodule."""
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
        self.task = task

        # Handle mixup_cfg (same logic as ImageNetDataModule)
        if isinstance(mixup_cfg, (dict, DictConfig)):
            base_cfg = OmegaConf.structured(MixupConfig)
            merged_cfg = OmegaConf.merge(base_cfg, mixup_cfg)
            self.mixup_cfg = OmegaConf.to_object(merged_cfg)
        else:
            self.mixup_cfg = mixup_cfg

        # Handle augment_cfg
        if isinstance(augment_cfg, (dict, DictConfig)):
            base_cfg = OmegaConf.structured(AugmentConfig)
            merged_cfg = OmegaConf.merge(base_cfg, augment_cfg)
            self.augment_cfg = OmegaConf.to_object(merged_cfg)
        else:
            self.augment_cfg = augment_cfg

        self.input_channels = 3
        if task == "classification":
            self.output_channels = num_classes
        elif task == "generation":
            self.output_channels = self.input_channels
        else:
            raise ValueError(f"Unsupported task: {task}")
        self.num_classes = num_classes

        self.normalization_mean = [0.5, 0.5, 0.5]
        self.normalization_std = [0.5, 0.5, 0.5]

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

        # Read metadata for epoch length calculation
        self._train_num_samples = self._read_num_samples("train")
        self._val_num_samples = self._read_num_samples("validation")

    def _read_num_samples(self, split: str) -> int:
        """Read the number of samples from the meta.json file."""
        meta_path = self.data_dir / split / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            return meta["num_samples"]
        # Fallback defaults
        return 1_281_167 if split == "train" else 50_000

    def _shard_pattern(self, split: str) -> str:
        """Return the WebDataset shard URL pattern for a split."""
        shard_dir = self.data_dir / split
        tars = sorted(shard_dir.glob("*.tar"))
        if not tars:
            raise FileNotFoundError(
                f"No TAR shards found in {shard_dir}. "
                f"Run scripts/convert_imagenet_to_webdataset.py first."
            )
        # Build brace pattern: /path/to/train/imagenet-train-{000000..000267}.tar
        first = tars[0].stem.split("-")[-1]  # e.g. "000000"
        last = tars[-1].stem.split("-")[-1]  # e.g. "000267"
        prefix = "-".join(tars[0].stem.split("-")[:-1])  # e.g. "imagenet-train"
        return str(shard_dir / f"{prefix}-{{{first}..{last}}}.tar")

    def _build_transform(self, *, train: bool) -> transforms.Compose:
        """Build image transform pipeline (same logic as ImageNetDataModule)."""
        mean, std = IMAGENET_MEAN_STD_BY_SIZE.get(
            self.final_image_size,
            (DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD),
        )

        if self.task == "generation":
            mean = self.normalization_mean
            std = self.normalization_std

        ops: list[transforms.Transform] = []

        if train:
            ops.append(transforms.Resize(self.image_size + 32, interpolation=InterpolationMode.BICUBIC))
            ops.append(transforms.RandomCrop(self.image_size))
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
                        hparams={"img_mean": tuple([int(x * 255) for x in mean])},
                    )
                )
        else:
            ops.append(transforms.Resize(self.image_size + 32, interpolation=InterpolationMode.BICUBIC))
            if self.center_crop:
                ops.append(transforms.CenterCrop(self.image_size))
            else:
                ops.append(transforms.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC))

        if self.final_image_size != self.image_size:
            ops.append(
                transforms.Resize(
                    self.final_image_size,
                    interpolation=InterpolationMode.BICUBIC,
                )
            )

        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(ops)

    def _make_pipeline(self, *, split: str, train: bool) -> wds.WebDataset:
        """Build a WebDataset pipeline for the given split."""
        url = self._shard_pattern(split)
        transform = self._build_transform(train=train)

        dataset = wds.WebDataset(
            url,
            shardshuffle=train,
            nodesplitter=wds.split_by_node,
        )

        if train:
            dataset = dataset.shuffle(5000)

        dataset = (
            dataset
            .map(_decode_image)
            .map(_decode_label)
            .map_dict(jpg=transform)
            .to_tuple("jpg", "cls")
        )

        if self.drop_labels:
            dataset = dataset.map(lambda x: (x[0], torch.tensor(-1, dtype=torch.long)))
        else:
            dataset = dataset.map(lambda x: (x[0], torch.tensor(x[1], dtype=torch.long)))

        return dataset

    def prepare_data(self) -> None:
        """Verify that WebDataset shards exist."""
        for split in ("train", "validation"):
            shard_dir = self.data_dir / split
            if not shard_dir.exists() or not list(shard_dir.glob("*.tar")):
                raise FileNotFoundError(
                    f"WebDataset shards not found in {shard_dir}. "
                    f"Run: python scripts/convert_imagenet_to_webdataset.py "
                    f"--src data/imagenet --dst {self.data_dir}"
                )

    def setup(self, stage: Optional[str] = None) -> None:
        """Construct the dataset pipelines for the requested stage."""
        if stage in ("fit", None):
            self.train_dataset = self._make_pipeline(split="train", train=True)
            self.val_dataset = self._make_pipeline(split="validation", train=False)
        elif stage == "validate":
            self.val_dataset = self._make_pipeline(split="validation", train=False)
        elif stage == "test":
            self.val_dataset = self._make_pipeline(split="validation", train=False)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        # WebDataset is IterableDataset; we set an epoch length explicitly
        # so Lightning knows when an epoch ends.
        num_batches = math.ceil(self._train_num_samples / self.batch_size)

        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
        # Set epoch length for Lightning
        loader = wds.WebLoader(loader, length=num_batches)
        return loader

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        num_batches = math.ceil(self._val_num_samples / self.batch_size)

        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )
        loader = wds.WebLoader(loader, length=num_batches)
        return loader

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return self.val_dataloader()

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Revert the normalization applied in the dataset pipeline."""
        mean = torch.as_tensor(self.normalization_mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.normalization_std, dtype=tensor.dtype, device=tensor.device)
        channels = mean.numel()

        if tensor.ndim == 4:
            if tensor.shape[1] == channels:
                reshape = (1, channels, 1, 1)
            elif tensor.shape[-1] == channels:
                reshape = (1, 1, 1, channels)
            else:
                raise ValueError("Unsupported tensor shape for unnormalization.")
        elif tensor.ndim == 3:
            if tensor.shape[0] == channels:
                reshape = (channels, 1, 1)
            elif tensor.shape[-1] == channels:
                reshape = (1, 1, channels)
            else:
                raise ValueError("Unsupported tensor shape for unnormalization.")
        else:
            raise ValueError("Tensor ndim must be 3 or 4 for unnormalization.")

        mean = mean.view(reshape)
        std = std.view(reshape)
        return torch.clamp(tensor * std + mean, 0.0, 1.0)

    def on_before_batch_transfer(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        dataloader_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Convert tuple batches to dict batches expected by the wrappers."""
        images, labels = batch

        if self.mixup_fn is not None and self.trainer.training:
            images, labels = self.mixup_fn(images, labels)

        images = images.permute(0, 2, 3, 1).contiguous()  # (bsize, height, width, num_channels)

        if len(labels.shape) == 1:
            labels = labels.view(-1)  # (bsize,)

        return {
            "input": images,
            "label": labels,
            "condition": None,
        }
