from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from timm.data import Mixup
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets as tv_datasets, transforms
from torchvision.io import decode_jpeg, read_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2 as transforms_v2
from timm.data.auto_augment import rand_augment_transform


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


@dataclass
class MixupConfig:
    """Configuration for mixup."""

    mixup: float = 0.0
    cutmix: float = 0.0
    mixup_prob: float = 1.0
    mixup_switch_prob: float = 0.5
    mixup_mode: str = "batch"
    smoothing: float = 0.1


@dataclass
class AugmentConfig:
    """Configuration for augmentations."""

    use_three_augment: bool = False
    color_jitter: float = 0.4
    rand_augment: Optional[str] = None  # e.g., 'rand-m9-n3-mstd0.5'


class ThreeAugment(torch.nn.Module):
    """DeiT III 3-Augment: Grayscale, Solarization, Gaussian Blur."""

    def __init__(self, prob: float = 1.0):
        """Initialize the 3-augment with probability `prob`."""
        super().__init__()
        self.prob = prob
        self.transforms = [
            transforms.RandomGrayscale(p=1.0),
            transforms.RandomSolarize(threshold=0.5, p=1.0),
            transforms.GaussianBlur(kernel_size=3),  # approx default, sigma random 0.1-2.0 usually
        ]

    def forward(self, img):
        """Apply 3-augment with probability `self.prob`."""
        if torch.rand(1) > self.prob:
            return img

        # Select one augmentation with equal probability
        idx = torch.randint(0, 3, (1,)).item()

        # For GaussianBlur, use DeiT III logic
        # DeiT III uses specific sigma logic, using standard range [0.1, 2.0]
        if idx == 2:
            sigma = torch.rand(1).item() * 1.9 + 0.1
            return transforms.GaussianBlur(kernel_size=5, sigma=sigma)(img)

        return self.transforms[idx](img)


class _ImageNetRawBytesDataset(Dataset):
    """Reads raw JPEG bytes from an ImageFolder layout for GPU-side decoding."""

    def __init__(self, root: Path, split: str) -> None:
        super().__init__()
        folder = "train" if split == "train" else "val"
        ds = tv_datasets.ImageFolder(root / folder)
        self.samples = ds.samples  # list of (path, class_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        raw_bytes = read_file(path)
        return raw_bytes, label


def _raw_bytes_collate(batch):
    """Collate that keeps variable-length byte tensors as a list."""
    bytes_list = [b for b, _ in batch]
    labels = torch.tensor([l for _, l in batch], dtype=torch.long)
    return bytes_list, labels


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

        from datasets import load_dataset
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
    """Lightning DataModule for ImageNet.

    Supports two backends:
    - **ImageFolder** (preferred): set ``imagefolder_dir`` to a directory
      containing ``train/`` and ``val/`` subdirectories in the standard
      torchvision ImageFolder layout.  Much faster than HF datasets.
    - **HuggingFace datasets** (fallback): uses Arrow-backed storage via
      ``data_dir``.  Slower due to per-sample Arrow deserialization + PIL
      decode overhead.
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
        hf_dataset_name: str = "imagenet-1k",
        hf_dataset_config: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
        num_classes: int = 1000,
        task: Literal["classification", "generation"],
        imagefolder_dir: Optional[str] = None,
        gpu_decode: bool = False,
        prefetch_factor: int = 4,
        # Augmentations
        mixup_cfg: Optional[MixupConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None,
    ) -> None:
        """Initialize the ImageNet datamodule and cache configuration values."""
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        self.imagefolder_dir = Path(imagefolder_dir) if imagefolder_dir else None
        self.gpu_decode = gpu_decode
        self.prefetch_factor = prefetch_factor
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

        self.task = task

        # Handle mixup_cfg
        if isinstance(mixup_cfg, (dict, DictConfig)):
            # Merge provided dict with default MixupConfig to ensure all keys like 'mixup_prob' exist
            base_cfg = OmegaConf.structured(MixupConfig)
            merged_cfg = OmegaConf.merge(base_cfg, mixup_cfg)
            self.mixup_cfg = OmegaConf.to_object(merged_cfg)
        else:
            self.mixup_cfg = mixup_cfg

        # Handle augment_cfg
        if isinstance(augment_cfg, (dict, DictConfig)):
            # Merge provided dict with default AugmentConfig
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

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        if self.gpu_decode:
            self._gpu_train_per_img, self._gpu_train_batch = self._build_gpu_transforms(train=True)
            self._gpu_val_per_img, self._gpu_val_batch = self._build_gpu_transforms(train=False)

    def _build_gpu_transforms(self, *, train: bool):
        """Build split GPU transform pipelines.

        Returns (per_image_ops, batch_ops) where per_image_ops handles
        variable-size inputs (decode → resize → crop) and batch_ops handles
        augmentations on a stacked [B,C,H,W] tensor for better GPU utilisation.
        """
        mean, std = IMAGENET_MEAN_STD_BY_SIZE.get(
            self.final_image_size,
            (DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD),
        )
        if self.task == "generation":
            mean = self.normalization_mean
            std = self.normalization_std

        per_img: list = []
        batch: list = []

        if train:
            per_img.append(transforms_v2.Resize(self.image_size + 32, interpolation=InterpolationMode.BICUBIC, antialias=True))
            per_img.append(transforms_v2.RandomCrop(self.image_size))

            batch.append(transforms_v2.RandomHorizontalFlip())

            if self.augment_cfg is not None and self.augment_cfg.use_three_augment:
                batch.append(transforms_v2.ColorJitter(
                    brightness=self.augment_cfg.color_jitter,
                    contrast=self.augment_cfg.color_jitter,
                    saturation=self.augment_cfg.color_jitter,
                ))
                batch.append(transforms_v2.RandomChoice([
                    transforms_v2.RandomGrayscale(p=1.0),
                    transforms_v2.RandomSolarize(threshold=128, p=1.0),
                    transforms_v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                ]))

            if self.augment_cfg is not None and self.augment_cfg.rand_augment:
                batch.append(transforms_v2.RandAugment())
        else:
            per_img.append(transforms_v2.Resize(self.image_size + 32, interpolation=InterpolationMode.BICUBIC, antialias=True))
            if self.center_crop:
                per_img.append(transforms_v2.CenterCrop(self.image_size))
            else:
                per_img.append(transforms_v2.Resize(self.image_size, interpolation=InterpolationMode.BICUBIC, antialias=True))

        if self.final_image_size != self.image_size:
            per_img.append(transforms_v2.Resize(self.final_image_size, interpolation=InterpolationMode.BICUBIC, antialias=True))

        batch.append(transforms_v2.ToDtype(torch.float32, scale=True))
        batch.append(transforms_v2.Normalize(mean=mean, std=std))

        return transforms_v2.Compose(per_img), transforms_v2.Compose(batch)

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
                ops.append(
                    transforms.ColorJitter(
                        brightness=self.augment_cfg.color_jitter,
                        contrast=self.augment_cfg.color_jitter,
                        saturation=self.augment_cfg.color_jitter,
                    )
                )
                # 3-Augment (Gray, Solar, Blur)
                ops.append(ThreeAugment())

            if self.augment_cfg is not None and self.augment_cfg.rand_augment:
                # RandAugment
                ops.append(
                    rand_augment_transform(
                        config_str=self.augment_cfg.rand_augment,
                        hparams={"img_mean": tuple([int(x * 255) for x in mean])},
                    )
                )
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
        if self.imagefolder_dir is not None:
            return

        from datasets import load_dataset
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

    def _make_folder_dataset(self, split: str, train: bool) -> Dataset:
        folder = "train" if split == "train" else "val"
        root = self.imagefolder_dir / folder
        return tv_datasets.ImageFolder(root, transform=self._build_transform(train=train))

    def _make_hf_dataset(self, split: str, train: bool) -> Dataset:
        return _ImageNetDataset(
            split=split,
            dataset_name=self.hf_dataset_name,
            dataset_config=self.hf_dataset_config,
            cache_dir=self.data_dir,
            hf_token=self.hf_auth_token,
            transform=self._build_transform(train=train),
            drop_labels=self.drop_labels,
        )

    def _make_raw_bytes_dataset(self, split: str) -> Dataset:
        return _ImageNetRawBytesDataset(self.imagefolder_dir, split)

    def setup(self, stage: Optional[str] = None) -> None:
        """Construct the datasets for the requested stage."""
        use_folder = self.imagefolder_dir is not None

        if stage in ("fit", None):
            if self.gpu_decode:
                self.train_dataset = self._make_raw_bytes_dataset("train")
                self.val_dataset = self._make_raw_bytes_dataset("validation")
            elif use_folder:
                self.train_dataset = self._make_folder_dataset("train", train=True)
                self.val_dataset = self._make_folder_dataset("validation", train=False)
            else:
                self.train_dataset = self._make_hf_dataset("train", train=True)
                self.val_dataset = self._make_hf_dataset("validation", train=False)

        elif stage in ("validate", "test"):
            if self.gpu_decode:
                self.val_dataset = self._make_raw_bytes_dataset("validation")
            elif use_folder:
                self.val_dataset = self._make_folder_dataset("validation", train=False)
            else:
                self.val_dataset = self._make_hf_dataset("validation", train=False)

    def _build_loader(self, dataset: Dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        kwargs = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory if not self.gpu_decode else False,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )
        if self.gpu_decode:
            kwargs["collate_fn"] = _raw_bytes_collate
        return DataLoader(dataset, **kwargs)

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
        batch,
        dataloader_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Convert tuple batches to dict batches expected by the diffusion wrappers.

        When gpu_decode is active, raw JPEG bytes arrive here; we decode on
        GPU, apply augmentations, then continue with the usual Mixup / permute.
        """
        if self.gpu_decode:
            return batch

        images, labels = batch

        if self.mixup_fn is not None and self.trainer.training:
            images, labels = self.mixup_fn(images, labels)

        images = images.permute(0, 2, 3, 1).contiguous()

        if len(labels.shape) == 1:
            labels = labels.view(-1)

        return {
            "input": images,
            "label": labels,
            "condition": None,
        }

    def on_after_batch_transfer(
        self,
        batch,
        dataloader_idx: int,
    ) -> dict[str, torch.Tensor]:
        """GPU-side decode + transform when gpu_decode is active.

        Uses a split pipeline: per-image ops (decode → resize → crop) produce
        uniform-size tensors that are stacked, then batch ops (augmentation,
        dtype conversion, normalisation) run on the full [B,C,H,W] tensor for
        much better GPU utilisation (~2x faster than per-image augmentation).
        """
        if not self.gpu_decode:
            return batch

        bytes_list, labels = batch
        device = labels.device if isinstance(labels, torch.Tensor) else torch.device("cuda")

        if self.trainer.training:
            per_img_ops, batch_ops = self._gpu_train_per_img, self._gpu_train_batch
        else:
            per_img_ops, batch_ops = self._gpu_val_per_img, self._gpu_val_batch

        cropped = []
        for raw in bytes_list:
            img = decode_jpeg(raw, device=device)
            cropped.append(per_img_ops(img))
        images = batch_ops(torch.stack(cropped))

        if self.mixup_fn is not None and self.trainer.training:
            images, labels = self.mixup_fn(images, labels)

        images = images.permute(0, 2, 3, 1).contiguous()

        if len(labels.shape) == 1:
            labels = labels.view(-1)

        return {
            "input": images,
            "label": labels,
            "condition": None,
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Only move labels to GPU; raw byte tensors stay on CPU for nvJPEG decode."""
        if self.gpu_decode and isinstance(batch, (list, tuple)) and len(batch) == 2:
            bytes_list, labels = batch
            if isinstance(bytes_list, list):
                return bytes_list, labels.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    @staticmethod
    def _uniform_dequantize(tensor: torch.Tensor) -> torch.Tensor:
        """Add uniform noise in [0, 1/256) after quantised pixels and keep the result in [0, 1]."""
        noise = torch.rand_like(tensor)
        return (tensor * 255.0 + noise) / 256.0
