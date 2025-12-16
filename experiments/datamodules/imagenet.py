from pathlib import Path
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data import create_transform, Mixup


# Pre-computed statistics for diffusion-ready 32x32 crops (10k sample estimate).
IMAGENET_MEAN_STD_BY_SIZE = {
    32: (
        [0.48450482, 0.45589244, 0.40366766],
        [0.25668961, 0.24765739, 0.26173702],
    ),
    64: (
        [0.48453078, 0.45592377, 0.40370297],
        [0.26425716, 0.25516447, 0.26875198],
    ),
}

DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]


from dataclasses import dataclass, field

@dataclass
class MixupConfig:
    mixup: float = 0.0
    cutmix: float = 0.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"
    smoothing: float = 0.1
    
@dataclass
class AugmentConfig:
    use_three_augment: bool = False
    color_jitter: float = 0.4
    # Future expansion: use_simple_random_crop, etc.


class ThreeAugment(torch.nn.Module):
    """DeiT III 3-Augment: Grayscale, Solarization, Gaussian Blur."""
    
    def __init__(self, prob: float = 1.0):
        super().__init__()
        self.prob = prob
        self.transforms = [
            transforms.RandomGrayscale(p=1.0),
            transforms.RandomSolarize(threshold=0.5, p=1.0), 
            transforms.GaussianBlur(kernel_size=3) # approx default, sigma random 0.1-2.0 usually
        ]

    def forward(self, img):
        if torch.rand(1) > self.prob:
            return img
            
        # Select one augmentation with equal probability
        idx = torch.randint(0, 3, (1,)).item()
        
        # Apply specific logic per transform if needed
        if idx == 2: # GaussianBlur
             # ConvNeXt/DeiT III might use specific sigma logic, using standard range [0.1, 2.0]
             sigma = torch.rand(1).item() * 1.9 + 0.1
             return transforms.GaussianBlur(kernel_size=5, sigma=sigma)(img)
             
        return self.transforms[idx](img)


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
        task: Literal["classification", "generation"],
        # Augmentations
        mixup_cfg: Optional[MixupConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None,
    ) -> None:
        """Initialize the ImageNet datamodule and cache configuration values."""
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
        
        # Default configs if not provided
        self.mixup_cfg = mixup_cfg
        self.augment_cfg = augment_cfg

        self.normalization_mean = [0.5, 0.5, 0.5]
        self.normalization_std = [0.5, 0.5, 0.5]

        self.mixup_fn: Optional[Mixup] = None
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

        self.train_dataset: Optional[_ImageNetDataset] = None
        self.val_dataset: Optional[_ImageNetDataset] = None

    def _build_transform(self, *, train: bool) -> transforms.Compose:
        mean, std = IMAGENET_MEAN_STD_BY_SIZE.get(
            self.final_image_size,
            (DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD),
        )

        # For generation tasks, we always use the diffusion normalization.
        # mapping images to [-1, 1].
        if self.task == "generation":
            mean = self.normalization_mean
            std = self.normalization_std

        # Initialize ops with Simple Random Crop logic: Resize -> RandomCrop
        # For SRC, we typically resize to slightly larger than crop size (e.g. 256 for 224 crop) or
        # resize shortest edge to target size.
        # Original code used Resize(image_size + 32). This is standard SRC.
        ops: list[transforms.Transform] = []

        if train:
            # Simple Random Crop
            ops.append(transforms.Resize(self.image_size + 32, interpolation=InterpolationMode.BICUBIC))
            ops.append(transforms.RandomCrop(self.image_size))
            
            # Manual augmentation pipeline matching DeiT III / User request
            # "ColorJitter Grayscale Gaussian Blur Solarization"
            ops.append(transforms.RandomHorizontalFlip())
            
            if self.augment_cfg is not None and self.augment_cfg.use_three_augment:
                 # Standard Color Jitter
                ops.append(transforms.ColorJitter(
                    brightness=self.augment_cfg.color_jitter, 
                    contrast=self.augment_cfg.color_jitter, 
                    saturation=self.augment_cfg.color_jitter
                ))
                # 3-Augment (Gray, Solar, Blur)
                ops.append(ThreeAugment())
        else:
            # Validation: Resize + CenterCrop or Resize
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
            split="validation",
            streaming=False,
            cache_dir=str(self.data_dir),
            token=self.hf_auth_token,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Construct the datasets for the requested stage."""
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

    @staticmethod
    def _uniform_dequantize(tensor: torch.Tensor) -> torch.Tensor:
        """Add uniform noise in [0, 1/256) after quantised pixels and keep the result in [0, 1]."""
        noise = torch.rand_like(tensor)
        return (tensor * 255.0 + noise) / 256.0
