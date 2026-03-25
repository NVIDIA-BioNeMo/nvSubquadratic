"""Optimized DALI-accelerated ImageNet DataModule.

Same functionality as ``dali_imagenet.DALIImageNetDataModule`` with performance
optimizations in the post-DALI augmentation pipeline:

1. Module-level imports (no per-forward ``gaussian_blur`` import)
2. Device-cached normalization tensors via ``register_buffer``
3. ``torch.where``-based blending instead of boolean-index scatter/gather
4. Vectorised random permutations (``argsort`` instead of Python loop)
5. Fused uint8→float + normalisation for the validation path
6. ``torch.compile``-friendly augmentation modules
7. Optional NCHW output (``channels_first=True``) to skip redundant permute

Requires: ``pip install nvidia-dali-cuda120``
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal, Optional

import pytorch_lightning as pl
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from omegaconf import DictConfig, OmegaConf
from timm.data import Mixup
from torchvision.transforms.v2.functional import gaussian_blur

from experiments.datamodules.dali_imagenet_fused import (
    DEFAULT_IMAGENET_MEAN,
    DEFAULT_IMAGENET_STD,
    IMAGENET_MEAN_STD_BY_SIZE,
    AugmentConfig,
    MixupConfig,
)


# ---------------------------------------------------------------------------
# DALI pipelines (identical to dali_imagenet.py)
# ---------------------------------------------------------------------------


@pipeline_def
def _train_pipeline(
    file_root: str,
    image_size: int,
    final_image_size: int,
    shard_id: int = 0,
    num_shards: int = 1,
):
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        random_shuffle=True,
        name="reader",
        shard_id=shard_id,
        num_shards=num_shards,
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.random_resized_crop(
        images,
        size=(image_size, image_size),
        random_area=(0.08, 1.0),
        interp_type=types.INTERP_CUBIC,
    )
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
    if final_image_size != image_size:
        images = fn.resize(
            images,
            size=(final_image_size, final_image_size),
            interp_type=types.INTERP_CUBIC,
        )
    return images, labels


@pipeline_def
def _val_pipeline(
    file_root: str,
    image_size: int,
    final_image_size: int,
    eval_crop_ratio: float,
    shard_id: int = 0,
    num_shards: int = 1,
):
    jpegs, labels = fn.readers.file(
        file_root=file_root,
        random_shuffle=False,
        name="reader",
        shard_id=shard_id,
        num_shards=num_shards,
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    eval_size = int(image_size / eval_crop_ratio)
    images = fn.resize(images, resize_shorter=eval_size, interp_type=types.INTERP_CUBIC)
    images = fn.crop(images, crop=(image_size, image_size))
    if final_image_size != image_size:
        images = fn.resize(
            images,
            size=(final_image_size, final_image_size),
            interp_type=types.INTERP_CUBIC,
        )
    return images, labels


# ---------------------------------------------------------------------------
# Thin wrapper so DALI iterators look like PyTorch DataLoaders to Lightning
# ---------------------------------------------------------------------------


class _DALILoaderWrapper:
    """Wraps ``DALIGenericIterator`` to yield ``(images, labels)`` tuples."""

    def __init__(self, dali_iterator: DALIGenericIterator):
        self._iter = dali_iterator

    def __iter__(self):
        for batch in self._iter:
            data = batch[0]
            images = data["images"]  # (B, H, W, C) uint8 GPU
            labels = data["labels"].squeeze(-1).long()  # (B,)
            yield images, labels

    def __len__(self):
        return len(self._iter)


# ---------------------------------------------------------------------------
# Batched PyTorch augmentations — compile-friendly versions
# ---------------------------------------------------------------------------


class _BatchThreeAugment(torch.nn.Module):
    """Batched DeiT III 3-Augment on (B, C, H, W) uint8 tensors.

    Uses ``torch.where`` for grayscale/solarize (no boolean-index
    scatter/gather). Blur still uses masked indexing since the convolution
    is too expensive to run on the full batch.
    """

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        dev = images.device
        choice = torch.randint(0, 3, (B,), device=dev)

        grey_mask = (choice == 0).view(B, 1, 1, 1)
        grey = images.float().mean(dim=1, keepdim=True).to(images.dtype).expand_as(images)
        images = torch.where(grey_mask, grey, images)

        solar_mask = (choice == 1).view(B, 1, 1, 1)
        solarized = torch.where(images >= 128, 255 - images, images)
        images = torch.where(solar_mask, solarized, images)

        blur_mask = choice == 2
        if blur_mask.any():
            sigma = torch.rand(1).item() * 1.9 + 0.1
            images[blur_mask] = gaussian_blur(
                images[blur_mask],
                kernel_size=[5, 5],
                sigma=sigma,
            )

        return images


class _BatchColorJitter(torch.nn.Module):
    """Per-image color jitter on a (B, C, H, W) float batch.

    Matches ``torchvision.transforms.ColorJitter`` behaviour:
    - Luminance-weighted grayscale (BT.601: 0.2989R + 0.587G + 0.114B)
    - Random application order per image (permutation of 3 ops)

    Optimised for ``torch.compile``: uses ``torch.where``-based blending
    and vectorised ``argsort`` permutations instead of boolean-index
    scatter/gather and Python-loop ``randperm``.
    """

    def __init__(self, brightness: float = 0.0, contrast: float = 0.0, saturation: float = 0.0):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.register_buffer("luma", torch.tensor([0.2989, 0.587, 0.114]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        dev = images.device

        b_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * self.brightness
        c_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * self.contrast
        s_factor = 1.0 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * self.saturation

        perms = torch.rand(B, 3, device=dev).argsort(dim=1)

        for step in range(3):
            grey = (images * self.luma).sum(dim=1, keepdim=True)
            grey_mean = grey.mean(dim=(2, 3), keepdim=True)

            brightness_out = images * b_factor
            contrast_out = grey_mean + (images - grey_mean) * c_factor
            saturation_out = grey + (images - grey) * s_factor

            sel = perms[:, step].view(B, 1, 1, 1)
            images = torch.where(
                sel == 0,
                brightness_out,
                torch.where(sel == 1, contrast_out, saturation_out),
            )

        return images.clamp_(0.0, 1.0)


# ---------------------------------------------------------------------------
# Compiled helpers
# ---------------------------------------------------------------------------


@torch.compile
def _fused_val_normalize(images: torch.Tensor, inv_std_255: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """uint8 NCHW → float32 normalised NCHW in a single fused kernel."""
    return images.float() * inv_std_255 + bias


# ---------------------------------------------------------------------------
# Main DataModule
# ---------------------------------------------------------------------------


class DALIImageNetOptimizedDataModule(pl.LightningDataModule):
    """Optimised DALI-accelerated ImageNet datamodule.

    Drop-in replacement for ``DALIImageNetDataModule`` with faster post-DALI
    augmentations via ``torch.compile``-friendly ops and fused normalisation.

    Set ``channels_first=True`` to output NCHW tensors directly, avoiding the
    final NHWC permute (requires the model to skip its own NHWC→NCHW rearrange).
    """

    def __init__(
        self,
        *,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        seed: int = 42,
        image_size: int = 224,
        final_image_size: Optional[int] = None,
        center_crop: bool = True,
        drop_labels: bool = False,
        hf_dataset_name: str = "imagenet-1k",
        hf_dataset_config: Optional[str] = None,
        hf_auth_token: Optional[str] = None,
        num_classes: int = 1000,
        task: Literal["classification", "generation"] = "classification",
        imagefolder_dir: Optional[str] = None,
        prefetch_factor: int = 2,
        eval_crop_ratio: float = 1.0,
        mixup_cfg: Optional[MixupConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None,
        device_id: int = 0,
        channels_first: bool = False,
        local_staging_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._local_staging_dir = Path(local_staging_dir) if local_staging_dir is not None else None

        if imagefolder_dir is None:
            raise ValueError(
                "DALIImageNetOptimizedDataModule requires imagefolder_dir "
                "(directory with train/ and val/ in ImageFolder layout)"
            )

        self.imagefolder_dir = Path(imagefolder_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.image_size = image_size
        self.final_image_size = final_image_size or image_size
        self.eval_crop_ratio = eval_crop_ratio
        self.device_id = device_id
        self.task = task
        self.num_classes = num_classes
        self.drop_labels = drop_labels
        self.prefetch_factor = prefetch_factor
        self.input_channels = 3
        self.output_channels = num_classes if task == "classification" else 3
        self.channels_first = channels_first

        # ── Normalization ────────────────────────────────────────────────────
        mean, std = IMAGENET_MEAN_STD_BY_SIZE.get(
            self.final_image_size,
            (DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD),
        )
        if task == "generation":
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.normalization_mean = mean
        self.normalization_std = std

        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self._norm_mean = mean_t
        self._norm_std = std_t
        self._inv_std_255 = 1.0 / (std_t * 255.0)
        self._neg_mean_over_std = -(mean_t / std_t)
        self._norm_cached_device: Optional[torch.device] = None

        # ── Augment config ───────────────────────────────────────────────────
        if isinstance(augment_cfg, (dict, DictConfig)):
            base = OmegaConf.structured(AugmentConfig)
            augment_cfg = OmegaConf.to_object(OmegaConf.merge(base, augment_cfg))
        self.augment_cfg = augment_cfg

        # ── Mixup config ─────────────────────────────────────────────────────
        if isinstance(mixup_cfg, (dict, DictConfig)):
            base = OmegaConf.structured(MixupConfig)
            mixup_cfg = OmegaConf.to_object(OmegaConf.merge(base, mixup_cfg))
        self.mixup_cfg = mixup_cfg

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

        # ── Batched augmentation modules ─────────────────────────────────────
        self._three_augment: Optional[_BatchThreeAugment] = None
        self._color_jitter: Optional[_BatchColorJitter] = None
        if self.augment_cfg is not None:
            if self.augment_cfg.use_three_augment:
                self._three_augment = _BatchThreeAugment()
            if self.augment_cfg.color_jitter > 0:
                cj = self.augment_cfg.color_jitter
                self._color_jitter = torch.compile(
                    _BatchColorJitter(brightness=cj, contrast=cj, saturation=cj),
                )

        self._train_pipe = None
        self._val_pipe = None

    # ------------------------------------------------------------------
    # Norm tensor caching
    # ------------------------------------------------------------------

    def _ensure_norm_on_device(self, device: torch.device) -> None:
        """Move normalisation tensors and augmentation modules to *device* once."""
        if self._norm_cached_device == device:
            return
        self._norm_mean = self._norm_mean.to(device)
        self._norm_std = self._norm_std.to(device)
        self._inv_std_255 = self._inv_std_255.to(device)
        self._neg_mean_over_std = self._neg_mean_over_std.to(device)
        if self._color_jitter is not None:
            self._color_jitter = self._color_jitter.to(device)
        self._norm_cached_device = device

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Optionally stage data to fast local storage before training.

        When ``local_staging_dir`` is set, copy the ImageFolder data there
        (idempotent — skips files that already exist).  On success,
        ``self.imagefolder_dir`` is updated to point to the local copy.
        Raises on failure so training does not silently use slow storage.
        """
        if self._local_staging_dir is None:
            return

        src = self.imagefolder_dir
        dst = self._local_staging_dir

        print(f"[data-staging] local_staging_dir={dst}, checking ...", flush=True)

        # Check destination is usable
        try:
            dst.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(dst).free
            min_bytes = 160 * (1024**3)  # 160 GB headroom for ImageNet
            if free_bytes < min_bytes:
                raise RuntimeError(
                    f"[data-staging] {dst} has only "
                    f"{free_bytes / (1024**3):.1f} GB free (need {min_bytes / (1024**3):.0f} GB)"
                )
        except OSError as exc:
            raise RuntimeError(f"[data-staging] Cannot access {dst}: {exc}") from exc

        sentinel = dst / ".staging_complete"
        if sentinel.is_file():
            print(f"[data-staging] {dst} already staged (sentinel found), skipping copy.", flush=True)
            self.imagefolder_dir = dst
            return

        print(f"[data-staging] Copying {src} → {dst} (this may take 10-20 min) ...", flush=True)
        subprocess.run(
            ["cp", "-a", "--no-clobber", "-r", str(src / "train"), str(src / "val"), str(dst)],
            check=True,
            timeout=3600,
        )
        sentinel.write_text("ok\n")
        self.imagefolder_dir = dst
        print(f"[data-staging] Done. Using local path: {dst}", flush=True)

    def setup(self, stage: Optional[str] = None) -> None:
        train_root = str(self.imagefolder_dir / "train")
        val_root = str(self.imagefolder_dir / "val")

        if self.trainer is not None:
            local_rank = self.trainer.local_rank
            world_size = self.trainer.world_size
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", self.device_id))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

        if stage in ("fit", None):
            self._train_pipe = _train_pipeline(
                file_root=train_root,
                image_size=self.image_size,
                final_image_size=self.final_image_size,
                shard_id=local_rank,
                num_shards=world_size,
                batch_size=self.batch_size,
                num_threads=self.num_workers,
                device_id=local_rank,
                seed=self.seed,
                prefetch_queue_depth=self.prefetch_factor,
            )
            self._train_pipe.build()

        if stage in ("fit", "validate", "test", None):
            self._val_pipe = _val_pipeline(
                file_root=val_root,
                image_size=self.image_size,
                final_image_size=self.final_image_size,
                eval_crop_ratio=self.eval_crop_ratio,
                shard_id=local_rank,
                num_shards=world_size,
                batch_size=self.batch_size,
                num_threads=self.num_workers,
                device_id=local_rank,
                seed=self.seed,
            )
            self._val_pipe.build()

    def train_dataloader(self):
        return _DALILoaderWrapper(
            DALIGenericIterator(
                self._train_pipe,
                output_map=["images", "labels"],
                reader_name="reader",
                last_batch_policy=LastBatchPolicy.DROP,
                auto_reset=True,
            )
        )

    def val_dataloader(self):
        return _DALILoaderWrapper(
            DALIGenericIterator(
                self._val_pipe,
                output_map=["images", "labels"],
                reader_name="reader",
                last_batch_policy=LastBatchPolicy.PARTIAL,
                auto_reset=True,
            )
        )

    def test_dataloader(self):
        return self.val_dataloader()

    # ------------------------------------------------------------------
    # Batch hooks — all data arrives on GPU from DALI
    # ------------------------------------------------------------------

    def on_before_batch_transfer(self, batch, dataloader_idx):
        images, labels = batch  # (B, H, W, C) uint8 GPU, (B,) long
        labels = labels.to(device=images.device)

        images = images.permute(0, 3, 1, 2).contiguous()  # → NCHW

        self._ensure_norm_on_device(images.device)

        is_train = self.trainer is not None and self.trainer.training
        if is_train:
            if self._three_augment is not None:
                images = self._three_augment(images)
            images = images.float().div_(255.0)
            if self._color_jitter is not None:
                images = self._color_jitter(images)
            images.sub_(self._norm_mean).div_(self._norm_std)
        else:
            images = _fused_val_normalize(images, self._inv_std_255, self._neg_mean_over_std)

        if self.mixup_fn is not None and is_train:
            images, labels = self.mixup_fn(images, labels)

        if not self.channels_first:
            images = images.permute(0, 2, 3, 1).contiguous()  # → NHWC
        if labels.ndim == 1:
            labels = labels.view(-1)

        return {"input": images, "label": labels, "condition": None}

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Revert normalization."""
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
