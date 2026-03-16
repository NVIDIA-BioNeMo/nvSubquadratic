"""Fully-fused DALI ImageNet DataModule — all augmentations inside DALI.

Moves ThreeAugment, ColorJitter, normalization, and the uint8→float
conversion into the DALI pipeline so that ``on_before_batch_transfer``
only does Mixup/CutMix (which needs labels) and the final layout permute.

This eliminates the ~25-33 ms of serial GPU augmentation that blocks the
training loop in ``dali_imagenet_optimized.py``.

Requires: ``pip install nvidia-dali-cuda120``
"""

import os
import shutil
import subprocess

# ---------------------------------------------------------------------------
# Shared ImageNet constants and config dataclasses
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pytorch_lightning as pl
import torch
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from omegaconf import DictConfig, OmegaConf
from timm.data import Mixup

from experiments.datamodules.utils.dali_rand_augment import dali_rand_augment


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
    rand_augment: Optional[str] = None
    random_erasing_prob: float = 0.0
    random_erasing_mode: str = "pixel"


# ---------------------------------------------------------------------------
# DALI pipelines — augmentations fused into the pipeline
# ---------------------------------------------------------------------------


def _solarize(images):
    """Solarize: invert pixels whose value >= 128 (per-element masking)."""
    mask = fn.cast(images >= 128, dtype=types.UINT8)
    inverted = fn.cast(255, dtype=types.UINT8) - images
    result = mask * inverted + (fn.cast(1, dtype=types.UINT8) - mask) * images
    return fn.cast(result, dtype=types.UINT8)


@pipeline_def(enable_conditionals=True)
def _train_pipeline_fused(
    file_root: str,
    image_size: int,
    final_image_size: int,
    norm_mean: tuple,
    norm_std: tuple,
    use_three_augment: bool = False,
    color_jitter: float = 0.0,
    rand_augment_config: str = "",
    shard_id: int = 0,
    num_shards: int = 1,
):
    """Training pipeline with decode, crop, augmentations, and normalization."""
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

    # ── ThreeAugment (grayscale / solarize / blur, each with p=1/3) ──
    # Uses DALI conditional execution (per-sample branching).
    if use_three_augment:
        coin = fn.random.uniform(range=(0.0, 1.0))
        if coin < (1.0 / 3.0):
            grey = fn.color_space_conversion(
                images,
                image_type=types.RGB,
                output_type=types.GRAY,
            )
            images = fn.cat(grey, grey, grey, axis=2)
        else:
            if coin < (2.0 / 3.0):
                images = _solarize(images)
            else:
                sigma = fn.random.uniform(range=(0.1, 2.0))
                images = fn.gaussian_blur(images, sigma=sigma, window_size=5)

    # ── ColorJitter (brightness, contrast, saturation) ───────────────
    if color_jitter > 0:
        brightness = fn.random.uniform(range=(1.0 - color_jitter, 1.0 + color_jitter))
        contrast = fn.random.uniform(range=(1.0 - color_jitter, 1.0 + color_jitter))
        saturation = fn.random.uniform(range=(1.0 - color_jitter, 1.0 + color_jitter))
        images = fn.color_twist(
            images,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
        )

    # ── RandAugment (timm-compatible, applied on uint8 before normalize) ──
    if rand_augment_config:
        images = dali_rand_augment(
            images,
            config_str=rand_augment_config,
            shape=(final_image_size, final_image_size),
        )

    # ── uint8 → float32 + normalize ─────────────────────────────────
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[m * 255.0 for m in norm_mean],
        std=[s * 255.0 for s in norm_std],
    )

    return images, labels


@pipeline_def
def _val_pipeline_fused(
    file_root: str,
    image_size: int,
    final_image_size: int,
    eval_crop_ratio: float,
    norm_mean: tuple,
    norm_std: tuple,
    shard_id: int = 0,
    num_shards: int = 1,
):
    """Validation pipeline with decode, resize, crop, and normalization."""
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

    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[m * 255.0 for m in norm_mean],
        std=[s * 255.0 for s in norm_std],
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
            images = data["images"]  # (B, C, H, W) float32 GPU (already NCHW + normalized)
            labels = data["labels"].squeeze(-1).long()
            yield images, labels

    def __len__(self):
        return len(self._iter)


# ---------------------------------------------------------------------------
# Main DataModule
# ---------------------------------------------------------------------------


class DALIImageNetFusedDataModule(pl.LightningDataModule):
    """DALI ImageNet DataModule with all augmentations inside the DALI pipeline.

    Unlike ``DALIImageNetOptimizedDataModule``, this module performs
    ThreeAugment, ColorJitter, uint8→float conversion, and normalization
    entirely within DALI. The ``on_before_batch_transfer`` hook only handles
    Mixup/CutMix (which requires label access) and optional NCHW→NHWC permute.

    This eliminates ~25-33 ms of serial GPU augmentation overhead per step.
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
        """Initialize DALI ImageNet DataModule with augmentation and mixup configs."""
        super().__init__()

        self._local_staging_dir = Path(local_staging_dir) if local_staging_dir is not None else None

        if imagefolder_dir is None:
            raise ValueError(
                "DALIImageNetFusedDataModule requires imagefolder_dir "
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

        # ── Normalization ────────────────────────────────────────────────
        mean, std = IMAGENET_MEAN_STD_BY_SIZE.get(
            self.final_image_size,
            (DEFAULT_IMAGENET_MEAN, DEFAULT_IMAGENET_STD),
        )
        if task == "generation":
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        self.normalization_mean = mean
        self.normalization_std = std
        self._norm_mean_tuple = tuple(mean)
        self._norm_std_tuple = tuple(std)

        # ── Augment config ───────────────────────────────────────────────
        if isinstance(augment_cfg, (dict, DictConfig)):
            base = OmegaConf.structured(AugmentConfig)
            augment_cfg = OmegaConf.to_object(OmegaConf.merge(base, augment_cfg))
        self.augment_cfg = augment_cfg

        self._use_three_augment = augment_cfg is not None and augment_cfg.use_three_augment
        self._color_jitter = augment_cfg.color_jitter if augment_cfg is not None else 0.0
        self._rand_augment_config = (
            augment_cfg.rand_augment if augment_cfg is not None and getattr(augment_cfg, "rand_augment", None) else ""
        )

        # ── RandomErasing (applied on GPU in on_before_batch_transfer) ───
        self._random_erasing_fn = None
        if augment_cfg is not None and getattr(augment_cfg, "random_erasing_prob", 0.0) > 0:
            from timm.data.random_erasing import RandomErasing

            self._random_erasing_fn = RandomErasing(
                probability=augment_cfg.random_erasing_prob,
                mode=getattr(augment_cfg, "random_erasing_mode", "pixel"),
                device="cuda",
            )

        # ── Mixup config ────────────────────────────────────────────────
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

        self._train_pipe = None
        self._val_pipe = None

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def _stage_to_local(self) -> None:
        """Copy ImageFolder data to fast local storage (e.g. NVMe).

        Idempotent: uses a ``.staging_complete`` sentinel so partial copies
        are retried and completed copies are skipped.  Raises on failure.
        """
        src = self.imagefolder_dir
        dst = self._local_staging_dir

        print(f"[data-staging] local_staging_dir={dst}, checking ...", flush=True)

        try:
            dst.mkdir(parents=True, exist_ok=True)
            free_bytes = shutil.disk_usage(dst).free
            min_bytes = 160 * (1024**3)
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

        print(f"[data-staging] Copying {src} -> {dst} (this may take 10-20 min) ...", flush=True)
        result = subprocess.run(
            ["cp", "-a", "--no-clobber", "-r", str(src / "train"), str(src / "val"), str(dst)],
            check=False,
            timeout=3600,
        )
        # cp --no-clobber exits 1 when it skips existing files (e.g. parallel
        # jobs staging to the same directory).  Only fail on unexpected codes.
        if result.returncode not in (0, 1):
            raise RuntimeError(f"[data-staging] cp failed with exit code {result.returncode}")
        sentinel.write_text("ok\n")
        self.imagefolder_dir = dst
        print(f"[data-staging] Done. Using local path: {dst}", flush=True)

    def prepare_data(self) -> None:
        """Stage data to local storage if configured."""
        if self._local_staging_dir is not None:
            self._stage_to_local()

    def setup(self, stage: Optional[str] = None) -> None:
        """Build DALI train/val pipelines for the given stage."""
        train_root = str(self.imagefolder_dir / "train")
        val_root = str(self.imagefolder_dir / "val")

        if self.trainer is not None:
            local_rank = self.trainer.local_rank
            world_size = self.trainer.world_size
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", self.device_id))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

        if stage in ("fit", None):
            self._train_pipe = _train_pipeline_fused(
                file_root=train_root,
                image_size=self.image_size,
                final_image_size=self.final_image_size,
                norm_mean=self._norm_mean_tuple,
                norm_std=self._norm_std_tuple,
                use_three_augment=self._use_three_augment,
                color_jitter=self._color_jitter,
                rand_augment_config=self._rand_augment_config,
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
            self._val_pipe = _val_pipeline_fused(
                file_root=val_root,
                image_size=self.image_size,
                final_image_size=self.final_image_size,
                eval_crop_ratio=self.eval_crop_ratio,
                norm_mean=self._norm_mean_tuple,
                norm_std=self._norm_std_tuple,
                shard_id=local_rank,
                num_shards=world_size,
                batch_size=self.batch_size,
                num_threads=self.num_workers,
                device_id=local_rank,
                seed=self.seed,
            )
            self._val_pipe.build()

    def train_dataloader(self):
        """Return DALI-backed training dataloader."""
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
        """Return DALI-backed validation dataloader."""
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
        """Return DALI-backed test dataloader (same as validation)."""
        return self.val_dataloader()

    # ------------------------------------------------------------------
    # Batch hooks — minimal work, only Mixup/CutMix + layout
    # ------------------------------------------------------------------

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Apply RandomErasing, Mixup/CutMix, and layout permute on GPU."""
        images, labels = batch  # (B, C, H, W) float32 GPU (already augmented + normalized)
        labels = labels.to(device=images.device)

        if self._random_erasing_fn is not None and self.trainer is not None and self.trainer.training:
            images = self._random_erasing_fn(images)

        if self.mixup_fn is not None and self.trainer is not None and self.trainer.training:
            images, labels = self.mixup_fn(images, labels)

        if not self.channels_first:
            images = images.permute(0, 2, 3, 1).contiguous()  # → NHWC

        if labels.ndim == 1:
            labels = labels.view(-1)

        return {"input": images, "label": labels, "condition": None}

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """No-op: DALI data is already on GPU."""
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
