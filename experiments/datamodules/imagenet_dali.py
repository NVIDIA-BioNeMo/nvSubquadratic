"""NVIDIA DALI-backed ImageNet DataModule for PyTorch Lightning.

Replaces the CPU-bound torchvision decode+augment pipeline with
DALI's GPU-accelerated JPEG decoding (nvJPEG) and GPU augmentations.

Three components:
1. **DALI Pipelines** — training & validation, defined via ``@pipeline_def``.
2. **DALIIteratorWrapper** — thin wrapper around ``DALIGenericIterator``
   that yields ``(images, labels)`` tuples compatible with Lightning.
3. **ImageNetDALIDataModule** — drop-in ``pl.LightningDataModule`` with the
   same attribute interface as ``ImageNetDataModule``.

Requires: ``nvidia-dali-cuda120>=1.28.0`` (install via
``pip install nvidia-dali-cuda120``).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from timm.data import Mixup

# Re-export config dataclasses for convenience
from experiments.datamodules.imagenet import (
    AugmentConfig,
    DEFAULT_IMAGENET_MEAN,
    DEFAULT_IMAGENET_STD,
    MixupConfig,
)

# ---------------------------------------------------------------------------
# Graceful fallback when DALI is not installed
# ---------------------------------------------------------------------------
try:
    from nvidia.dali import fn, pipeline_def, types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [m * 255.0 for m in DEFAULT_IMAGENET_MEAN]
_IMAGENET_STD = [s * 255.0 for s in DEFAULT_IMAGENET_STD]


# ═══════════════════════════════════════════════════════════════════════════
# 1. DALI Pipelines
# ═══════════════════════════════════════════════════════════════════════════

def _training_pipeline(
    image_dir: str,
    image_size: int,
    final_image_size: int,
    shard_id: int,
    num_shards: int,
    seed: int,
    use_three_augment: bool = False,
    color_jitter: float = 0.3,
    mean: list[float] | None = None,
    std: list[float] | None = None,
):
    """Build a DALI training pipeline.

    Must be called within a ``@pipeline_def`` context or used as the body
    of a pipeline_def-decorated function (see ``_build_train_pipeline``).
    """
    if mean is None:
        mean = _IMAGENET_MEAN
    if std is None:
        std = _IMAGENET_STD

    # --- Read & decode (CPU read, GPU decode via nvJPEG) -----------------
    jpegs, labels = fn.readers.file(
        file_root=image_dir,
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards,
        seed=seed,
        name="reader",
    )
    # Use mixed decode (nvJPEG on GPU) with hw_decoder_load=0.65 (default).
    # Some Photoshop-exported JPEGs with complex metadata can crash nvJPEG;
    # if that happens, switch to device="cpu" + images.gpu() as a fallback.
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    # --- Resize shorter side to image_size + 32 -------------------------
    # Matches torchvision SRC: Resize(image_size + 32) → RandomCrop(image_size)
    images = fn.resize(
        images,
        resize_shorter=image_size + 32,
        interp_type=types.INTERP_CUBIC,
    )

    # --- Random crop to ``image_size`` -----------------------------------
    images = fn.crop(
        images,
        crop=(image_size, image_size),
        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
    )

    # --- Random horizontal flip ------------------------------------------
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))

    # --- Color jitter (brightness / contrast / saturation) ---------------
    if use_three_augment and color_jitter > 0:
        # Brightness + contrast via brightness_contrast
        # torchvision ColorJitter(brightness=b) samples a multiplicative factor
        # from [max(0, 1-b), 1+b], so we use the `brightness` (multiplicative)
        # parameter here, NOT brightness_shift (additive).
        brightness_factor = fn.random.uniform(range=(1.0 - color_jitter, 1.0 + color_jitter))
        contrast_factor = fn.random.uniform(range=(1.0 - color_jitter, 1.0 + color_jitter))
        images = fn.brightness_contrast(
            images,
            brightness=brightness_factor,
            contrast_center=0.5,
            contrast=contrast_factor,
        )
        # Saturation
        sat_factor = fn.random.uniform(range=(1.0 - color_jitter, 1.0 + color_jitter))
        images = fn.saturation(images, saturation=sat_factor)

    # --- ThreeAugment (DeiT III: Grayscale | Solarize | GaussianBlur) ---
    if use_three_augment:
        # Select one of three augmentations with equal probability
        aug_selector = fn.random.uniform(range=(0.0, 3.0))

        # Grayscale branch
        gray = fn.saturation(images, saturation=0.0)

        # Solarize branch (threshold 128 on uint8)
        images_f = fn.cast(images, dtype=types.FLOAT)
        solarized = fn.cast(
            images_f * fn.cast(images_f < 128.0, dtype=types.FLOAT)
            + (255.0 - images_f) * fn.cast(images_f >= 128.0, dtype=types.FLOAT),
            dtype=types.UINT8,
        )

        # GaussianBlur branch
        sigma = fn.random.uniform(range=(0.1, 2.0))
        blurred = fn.gaussian_blur(images, sigma=sigma, window_size=5)

        # Select based on aug_selector: [0,1) -> gray, [1,2) -> solarize, [2,3) -> blur
        images = blurred  # default to blurred
        if DALI_AVAILABLE:
            # Use element-wise conditional selection
            # DALI 1.28+ supports if_then in @pipeline_def context
            # Fallback: weighted random selection via sequential overwrite
            use_gray = fn.cast(aug_selector < 1.0, dtype=types.FLOAT)
            use_solar = fn.cast((aug_selector >= 1.0) & (aug_selector < 2.0), dtype=types.FLOAT)
            use_blur = fn.cast(aug_selector >= 2.0, dtype=types.FLOAT)

            # Broadcast multiply and sum
            images_f_gray = fn.cast(gray, dtype=types.FLOAT)
            images_f_solar = fn.cast(solarized, dtype=types.FLOAT)
            images_f_blur = fn.cast(blurred, dtype=types.FLOAT)

            images = fn.cast(
                images_f_gray * use_gray + images_f_solar * use_solar + images_f_blur * use_blur,
                dtype=types.UINT8,
            )

    # --- Final resize to model input size (matches torchvision's extra Resize) --
    # Mirrors: if self.final_image_size != self.image_size: Resize(final_image_size)
    if image_size != final_image_size:
        images = fn.resize(
            images,
            resize_shorter=final_image_size,
            interp_type=types.INTERP_CUBIC,
        )

    # --- Final normalize + to float: produces [B, H, W, C] float32 ------
    images = fn.crop_mirror_normalize(
        images,
        mean=mean,
        std=std,
        output_layout="HWC",
        dtype=types.FLOAT,
    )

    labels = labels.gpu()
    return images, labels


def _validation_pipeline(
    image_dir: str,
    image_size: int,
    final_image_size: int,
    shard_id: int,
    num_shards: int,
    seed: int,
    mean: list[float] | None = None,
    std: list[float] | None = None,
):
    """Build a DALI validation pipeline (deterministic center crop)."""
    if mean is None:
        mean = _IMAGENET_MEAN
    if std is None:
        std = _IMAGENET_STD

    jpegs, labels = fn.readers.file(
        file_root=image_dir,
        random_shuffle=False,
        shard_id=shard_id,
        num_shards=num_shards,
        seed=seed,
        name="reader",
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)

    # Resize shorter to image_size + 32, then center crop to image_size
    images = fn.resize(
        images,
        resize_shorter=image_size + 32,
        interp_type=types.INTERP_CUBIC,
    )

    # Center crop to image_size
    images = fn.crop(
        images,
        crop=(image_size, image_size),
        crop_pos_x=0.5,
        crop_pos_y=0.5,
    )

    # Final resize to model input size (when image_size != final_image_size)
    if image_size != final_image_size:
        images = fn.resize(
            images,
            resize_shorter=final_image_size,
            interp_type=types.INTERP_CUBIC,
        )

    # Normalize -> HWC float
    images = fn.crop_mirror_normalize(
        images,
        mean=mean,
        std=std,
        output_layout="HWC",
        dtype=types.FLOAT,
    )

    labels = labels.gpu()
    return images, labels


def _build_train_pipeline(
    *,
    image_dir: str,
    batch_size: int,
    num_threads: int,
    device_id: int,
    image_size: int,
    final_image_size: int,
    shard_id: int,
    num_shards: int,
    seed: int,
    use_three_augment: bool,
    color_jitter: float,
    prefetch_queue_depth: int,
    mean: list[float] | None = None,
    std: list[float] | None = None,
):
    """Instantiate and build a DALI training pipeline."""

    @pipeline_def(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    def pipe():
        return _training_pipeline(
            image_dir=image_dir,
            image_size=image_size,
            final_image_size=final_image_size,
            shard_id=shard_id,
            num_shards=num_shards,
            seed=seed,
            use_three_augment=use_three_augment,
            color_jitter=color_jitter,
            mean=mean,
            std=std,
        )

    p = pipe()
    p.build()
    return p


def _build_val_pipeline(
    *,
    image_dir: str,
    batch_size: int,
    num_threads: int,
    device_id: int,
    image_size: int,
    final_image_size: int,
    shard_id: int,
    num_shards: int,
    seed: int,
    prefetch_queue_depth: int,
    mean: list[float] | None = None,
    std: list[float] | None = None,
):
    """Instantiate and build a DALI validation pipeline."""

    @pipeline_def(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=seed,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    def pipe():
        return _validation_pipeline(
            image_dir=image_dir,
            image_size=image_size,
            final_image_size=final_image_size,
            shard_id=shard_id,
            num_shards=num_shards,
            seed=seed,
            mean=mean,
            std=std,
        )

    p = pipe()
    p.build()
    return p


# ═══════════════════════════════════════════════════════════════════════════
# 2. DALIIteratorWrapper
# ═══════════════════════════════════════════════════════════════════════════

class DALIIteratorWrapper:
    """Wraps ``DALIGenericIterator`` to yield ``(images, labels)`` tuples.

    * ``images`` — GPU tensor ``[B, H, W, C]`` (channels-last, float32)
    * ``labels`` — GPU tensor ``[B]`` (int64)
    """

    def __init__(
        self,
        pipeline,
        size: int,
        batch_size: int,
        auto_reset: bool = True,
        drop_last: bool = False,
    ):
        self._size = size
        self._batch_size = batch_size
        policy = LastBatchPolicy.DROP if drop_last else LastBatchPolicy.PARTIAL
        self._iterator = DALIGenericIterator(
            [pipeline],
            output_map=["images", "labels"],
            size=size,
            auto_reset=auto_reset,
            last_batch_padded=True,
            last_batch_policy=policy,
        )

    def __iter__(self):
        for batch in self._iterator:
            data = batch[0]  # single pipeline -> first element
            images = data["images"]  # [B, H, W, C] float32 on GPU
            labels = data["labels"].squeeze(-1).long()  # [B] int64 on GPU
            yield images, labels

    def __len__(self) -> int:
        return math.ceil(self._size / self._batch_size)

    def reset(self):
        """Manually reset the iterator (usually auto_reset handles this)."""
        self._iterator.reset()


# ═══════════════════════════════════════════════════════════════════════════
# 3. ImageNetDALIDataModule
# ═══════════════════════════════════════════════════════════════════════════

class ImageNetDALIDataModule(pl.LightningDataModule):
    """DALI-accelerated ImageNet DataModule for Lightning.

    Drop-in replacement for ``ImageNetDataModule`` with GPU-decoded JPEGs
    and GPU-accelerated augmentations.
    """

    def __init__(
        self,
        *,
        imagefolder_dir: str,
        batch_size: int,
        num_threads: int,
        seed: int,
        image_size: int = 256,
        final_image_size: int = 224,
        num_classes: int = 1000,
        task: str = "classification",
        mixup_cfg: Optional[MixupConfig] = None,
        augment_cfg: Optional[AugmentConfig] = None,
        prefetch_queue_depth: int = 2,
    ) -> None:
        if not DALI_AVAILABLE:
            raise ImportError(
                "NVIDIA DALI is required for ImageNetDALIDataModule. "
                "Install with: pip install nvidia-dali-cuda120>=1.28.0"
            )
        super().__init__()
        self.imagefolder_dir = imagefolder_dir
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.seed = seed
        self.image_size = image_size
        self.final_image_size = final_image_size
        self.task = task
        self.prefetch_queue_depth = prefetch_queue_depth

        # --- Public attributes (match ImageNetDataModule interface) --------
        self.input_channels = 3
        self.num_classes = num_classes
        if task == "classification":
            self.output_channels = num_classes
        elif task == "generation":
            self.output_channels = self.input_channels
        else:
            raise ValueError(f"Unsupported task: {task}")

        self.normalization_mean = DEFAULT_IMAGENET_MEAN
        self.normalization_std = DEFAULT_IMAGENET_STD

        # --- Handle mixup_cfg (dict / DictConfig / dataclass) -------------
        if isinstance(mixup_cfg, (dict, DictConfig)):
            base_cfg = OmegaConf.structured(MixupConfig)
            merged_cfg = OmegaConf.merge(base_cfg, mixup_cfg)
            self.mixup_cfg = OmegaConf.to_object(merged_cfg)
        else:
            self.mixup_cfg = mixup_cfg

        # --- Handle augment_cfg -------------------------------------------
        if isinstance(augment_cfg, (dict, DictConfig)):
            base_cfg = OmegaConf.structured(AugmentConfig)
            merged_cfg = OmegaConf.merge(base_cfg, augment_cfg)
            self.augment_cfg = OmegaConf.to_object(merged_cfg)
        else:
            self.augment_cfg = augment_cfg

        # Determine augmentation flags
        self._use_three_augment = (
            self.augment_cfg is not None and getattr(self.augment_cfg, "use_three_augment", False)
        )
        self._color_jitter = (
            getattr(self.augment_cfg, "color_jitter", 0.3) if self.augment_cfg else 0.0
        )

        # --- Mixup / CutMix -----------------------------------------------
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

        # Will be set in setup()
        self.shard_id = 0
        self.num_shards = 1
        self.device_id = 0
        self._train_pipeline = None
        self._val_pipeline = None
        self._train_size = 0
        self._val_size = 0

    # ------------------------------------------------------------------
    # DDP-aware setup
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        """Build DALI pipelines with DDP-aware sharding."""
        # Determine DDP rank / world size
        if self.trainer is not None:
            self.shard_id = self.trainer.global_rank
            self.num_shards = self.trainer.world_size
            self.device_id = self.trainer.local_rank
        else:
            self.shard_id = int(os.environ.get("LOCAL_RANK", 0))
            self.num_shards = int(os.environ.get("WORLD_SIZE", 1))
            self.device_id = int(os.environ.get("LOCAL_RANK", 0))

        train_dir = os.path.join(self.imagefolder_dir, "train")
        val_dir = os.path.join(self.imagefolder_dir, "val")

        # Count samples for __len__ calculation
        self._train_size = sum(
            len(files) for _, _, files in os.walk(train_dir)
        ) // self.num_shards
        self._val_size = sum(
            len(files) for _, _, files in os.walk(val_dir)
        ) // self.num_shards

        # Normalization values scaled to [0, 255] range
        mean = [m * 255.0 for m in self.normalization_mean]
        std = [s * 255.0 for s in self.normalization_std]

        if stage in ("fit", None):
            self._train_pipeline = _build_train_pipeline(
                image_dir=train_dir,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                device_id=self.device_id,
                image_size=self.image_size,
                final_image_size=self.final_image_size,
                shard_id=self.shard_id,
                num_shards=self.num_shards,
                seed=self.seed + self.shard_id,
                use_three_augment=self._use_three_augment,
                color_jitter=self._color_jitter,
                prefetch_queue_depth=self.prefetch_queue_depth,
                mean=mean,
                std=std,
            )
            self._val_pipeline = _build_val_pipeline(
                image_dir=val_dir,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                device_id=self.device_id,
                image_size=self.image_size,
                final_image_size=self.final_image_size,
                shard_id=self.shard_id,
                num_shards=self.num_shards,
                seed=self.seed,
                prefetch_queue_depth=self.prefetch_queue_depth,
                mean=mean,
                std=std,
            )

        elif stage in ("validate", "test"):
            self._val_pipeline = _build_val_pipeline(
                image_dir=val_dir,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                device_id=self.device_id,
                image_size=self.image_size,
                final_image_size=self.final_image_size,
                shard_id=self.shard_id,
                num_shards=self.num_shards,
                seed=self.seed,
                prefetch_queue_depth=self.prefetch_queue_depth,
                mean=mean,
                std=std,
            )

    def prepare_data(self) -> None:
        """No-op: DALI reads directly from ImageFolder layout on disk."""

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        """Return training iterator wrapping the DALI pipeline."""
        if self._train_pipeline is None:
            raise RuntimeError("train_dataloader called before setup('fit')")
        return DALIIteratorWrapper(
            self._train_pipeline, size=self._train_size, batch_size=self.batch_size,
            auto_reset=True, drop_last=True,
        )

    def val_dataloader(self):
        """Return validation iterator wrapping the DALI pipeline."""
        if self._val_pipeline is None:
            raise RuntimeError("val_dataloader called before setup")
        return DALIIteratorWrapper(
            self._val_pipeline, size=self._val_size, batch_size=self.batch_size, auto_reset=True,
        )

    def test_dataloader(self):
        """Return test iterator (same as val)."""
        return self.val_dataloader()

    # ------------------------------------------------------------------
    # Batch transfer hook (Mixup/CutMix on GPU)
    # ------------------------------------------------------------------

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        """Unpack DALI batch and apply Mixup/CutMix on GPU tensors.

        DALI already produces GPU tensors in channels-last [B, H, W, C],
        so no permute is needed (unlike the torchvision path).
        """
        images, labels = batch

        if self.mixup_fn is not None and self.trainer.training:
            # timm Mixup expects NCHW — temporarily permute for compatibility
            images_nchw = images.permute(0, 3, 1, 2).contiguous()
            images_nchw, labels = self.mixup_fn(images_nchw, labels)
            images = images_nchw.permute(0, 2, 3, 1).contiguous()

        if len(labels.shape) == 1:
            labels = labels.view(-1)

        return {
            "input": images,
            "label": labels,
            "condition": None,
        }

    # ------------------------------------------------------------------
    # Unnormalize (for visualization / debugging)
    # ------------------------------------------------------------------

    def unnormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Revert the ImageNet normalization."""
        mean = torch.as_tensor(self.normalization_mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.normalization_std, dtype=tensor.dtype, device=tensor.device)
        channels = mean.numel()

        if tensor.ndim == 4:
            if tensor.shape[-1] == channels:
                reshape = (1, 1, 1, channels)  # NHWC
            elif tensor.shape[1] == channels:
                reshape = (1, channels, 1, 1)  # NCHW
            else:
                raise ValueError("Unsupported tensor shape for unnormalization.")
        elif tensor.ndim == 3:
            if tensor.shape[-1] == channels:
                reshape = (1, 1, channels)
            elif tensor.shape[0] == channels:
                reshape = (channels, 1, 1)
            else:
                raise ValueError("Unsupported tensor shape for unnormalization.")
        else:
            raise ValueError("Tensor ndim must be 3 or 4 for unnormalization.")

        mean = mean.view(reshape)
        std = std.view(reshape)
        return torch.clamp(tensor * std + mean, 0.0, 1.0)
