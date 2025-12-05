from pathlib import Path
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# TinyImageNet statistics
TINYIMAGENET_MEAN = [0.4802, 0.4481, 0.3975]
TINYIMAGENET_STD = [0.2302, 0.2265, 0.2262]


class _TinyImageNetDataset(Dataset):
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


class TinyImageNetDataModule(pl.LightningDataModule):
    """Lightning DataModule for TinyImageNet."""

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        seed: int,
        image_size: int = 64,
        final_image_size: Optional[int] = None,
        center_crop: bool = False,  # TinyImageNet is already 64x64, usually no crop needed unless aug
        drop_labels: bool = False,
        hf_dataset_name: str = "zh-plus/tiny-imagenet",
        hf_dataset_config: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
        num_classes: int = 200,
        task: Literal["classification", "generation"] = "classification",
    ) -> None:
        """Initialize the TinyImageNet datamodule."""
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

        self.input_channels = 3
        if task == "classification":
            self.output_channels = num_classes
        elif task == "generation":
            self.output_channels = self.input_channels
        else:
            raise ValueError(f"Unsupported task: {task}")
        self.num_classes = num_classes

        # Diffusion experiments consume inputs scaled to [-1, 1].
        self.normalization_mean = (0.5, 0.5, 0.5)
        self.normalization_std = (0.5, 0.5, 0.5)

        self.train_dataset: Optional[_TinyImageNetDataset] = None
        self.val_dataset: Optional[_TinyImageNetDataset] = None

    def _build_transform(self, *, train: bool) -> transforms.Compose:
        mean = TINYIMAGENET_MEAN
        std = TINYIMAGENET_STD

        # For generation tasks, we always use the diffusion normalization.
        # mapping images to [-1, 1].
        if self.task == "generation":
            mean = self.normalization_mean
            std = self.normalization_std

        ops: list[transforms.Transform] = []

        # TinyImageNet images are 64x64.
        # If image_size != 64, we might resize.
        if self.image_size != 64:
            ops.append(transforms.Resize(self.image_size))

        if train:
            # Standard augmentation
            ops.append(transforms.RandomCrop(self.image_size, padding=4))
            ops.append(transforms.RandomHorizontalFlip())
        else:
            if self.center_crop and self.image_size < 64:
                ops.append(transforms.CenterCrop(self.image_size))

        if self.final_image_size != self.image_size:
            ops.append(
                transforms.Resize(
                    self.final_image_size,
                    interpolation=InterpolationMode.BICUBIC,
                )
            )

        # Final preprocessing
        ops.append(transforms.ToTensor())
        if self.task == "generation":
            ops.append(transforms.Lambda(self._uniform_dequantize))
        ops.append(transforms.Normalize(mean=mean, std=std))
        return transforms.Compose(ops)

    def prepare_data(self) -> None:
        """Download the train/validation splits if they are not already cached locally."""
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
            split="valid",  # zh-plus/tiny-imagenet uses 'valid'
            streaming=False,
            cache_dir=str(self.data_dir),
            token=self.hf_auth_token,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Construct the datasets for the requested stage."""
        if stage in ("fit", None):
            self.train_dataset = _TinyImageNetDataset(
                split="train",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=True),
                drop_labels=self.drop_labels,
            )

            self.val_dataset = _TinyImageNetDataset(
                split="valid",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=False),
                drop_labels=self.drop_labels,
            )

        elif stage == "validate":
            self.val_dataset = _TinyImageNetDataset(
                split="valid",
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=False),
                drop_labels=self.drop_labels,
            )

        elif stage == "test":
            self.val_dataset = _TinyImageNetDataset(
                split="valid",  # TinyImageNet test set has no labels usually, so we use valid for testing if needed
                dataset_name=self.hf_dataset_name,
                dataset_config=self.hf_dataset_config,
                cache_dir=self.data_dir,
                hf_token=self.hf_auth_token,
                transform=self._build_transform(train=False),
                drop_labels=self.drop_labels,
            )

    def _build_loader(self, dataset: _TinyImageNetDataset, shuffle: bool, drop_last: bool) -> DataLoader:
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
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataloader called before setup('fit')")

        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataloader called before setup")

        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("test_dataloader called before setup('test')")

        return self._build_loader(self.val_dataset, shuffle=False, drop_last=False)

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
        """Convert tuple batches to dict batches expected by the diffusion wrappers."""
        images, labels = batch

        images = images.permute(0, 2, 3, 1).contiguous()  # (bsize, height, width, num_channels)

        labels = labels.view(-1)  # (bsize,)

        return {
            "input": images,
            "label": labels,
            "condition": None,
        }

    @staticmethod
    def _uniform_dequantize(tensor: torch.Tensor) -> torch.Tensor:
        """Add uniform noise in [0, 1/256) after quantised pixels and keep the result in [0, 1]."""
        noise = torch.rand_like(tensor)
        return (tensor * 255.0 + noise) / 256.0
