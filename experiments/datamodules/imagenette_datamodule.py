from pathlib import Path
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from timm.data import Mixup
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from experiments.datamodules.imagenet import AugmentConfig, MixupConfig, ThreeAugment
from timm.data.auto_augment import rand_augment_transform


# Imagenette statistics (using standard ImageNet stats as a proxy or if known)
# Standard ImageNet stats:
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _ImagenetteDataset(Dataset):
    """Hugging Face backed dataset for Imagenette."""

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


class ImagenetteDataModule(pl.LightningDataModule):
    """Lightning DataModule for Imagenette."""

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int,
        image_size: int = 160,
        final_image_size: Optional[int] = None,
        center_crop: bool = True,
        drop_labels: bool = False,
        hf_dataset_name: str = "Sijuade/ImageNette",
        hf_dataset_config: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
        num_classes: int = 10,
        task: Literal["classification", "generation"] = "classification",
        # Augmentations
        mixup_cfg: Optional[MixupConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None,
    ) -> None:
        """Initialize the Imagenette datamodule."""
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
        self.task = task

        # Handle mixup_cfg
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

        self.normalization_mean = (0.5, 0.5, 0.5)
        self.normalization_std = (0.5, 0.5, 0.5)

        self.train_dataset: Optional[_ImagenetteDataset] = None
        self.val_dataset: Optional[_ImagenetteDataset] = None

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

    def _build_transform(self, *, train: bool) -> transforms.Compose:
        mean = IMAGENET_MEAN
        std = IMAGENET_STD

        if self.task == "generation":
            mean = self.normalization_mean
            std = self.normalization_std

        ops: list[transforms.Transform] = []

        # Resize shortest edge to target size + padding if training, or just target size
        if train:
            ops.append(transforms.RandomResizedCrop(self.image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC))
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
                        hparams={"img_mean": tuple([int(x * 255) for x in IMAGENET_MEAN])},
                    )
                )

        else:
            # Traditional validation transform: resize to 256 then center crop to 224
            # For 160px base, we'll resize to int(1.14 * image_size) then center crop
            resize_size = int((256 / 224) * self.image_size)
            ops.append(transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC))
            if self.center_crop:
                ops.append(transforms.CenterCrop(self.image_size))

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

    def prepare_data(self) -> None:
        """Download the train/validation splits."""
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
        """Construct the datasets."""
        if stage in ("fit", None):
            self.train_dataset = _ImagenetteDataset(
                split="train",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=True),
                drop_labels=self.drop_labels,
            )

            self.val_dataset = _ImagenetteDataset(
                split="validation",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=False),
                drop_labels=self.drop_labels,
            )

        elif stage == "validate":
            self.val_dataset = _ImagenetteDataset(
                split="validation",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=False),
                drop_labels=self.drop_labels,
            )

        elif stage == "test":
            self.val_dataset = _ImagenetteDataset(
                split="validation",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=False),
                drop_labels=self.drop_labels,
            )

    def _build_loader(self, dataset: _ImagenetteDataset, shuffle: bool, drop_last: bool) -> DataLoader:
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

        if self.mixup_fn is not None and self.trainer.training:
            images, labels = self.mixup_fn(images, labels)

        images = images.permute(0, 2, 3, 1).contiguous()  # (bsize, height, width, num_channels)

        if len(labels.shape) == 1:
            labels = labels.view(-1)

        return {
            "input": images,
            "label": labels,
            "condition": None,
        }
