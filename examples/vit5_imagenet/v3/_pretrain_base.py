"""Shared base config for ViT-5-Small ImageNet-1k pretraining (v3).

Provides:
  - All shared constants (dataset, model dimensions, training recipe)
  - ``get_base_config()`` — returns an ``ExperimentConfig`` with everything
    set *except* ``config.net`` (the network is mixer-specific).
  - ``make_block_cfg(sequence_mixer_cfg)`` — builds a ``ViT5ResidualBlock``
    LazyConfig with the given sequence mixer and the shared MLP / norms /
    LayerScale / DropPath settings.

Individual configs (attention, hyena, ...) import from here and only define
their mixer-specific ``sequence_mixer_cfg`` and ``config.net``.
"""

import os

import torch
from apex.optimizers import FusedLAMB as Lamb

from experiments.callbacks.model_ema import LabeledEMAWeightAveraging
from experiments.datamodules.dali_imagenet_fused import (
    AugmentConfig,
    DALIImageNetFusedDataModule,
    MixupConfig,
)
from experiments.default_cfg import (
    AutoResumeConfig,
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.utils.init import trunc_normal_init, trunc_normal_init_factory


# ─── Dataset ────────────────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 224
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
IMAGENET_FOLDER_PATH = os.environ.get("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

# ─── Model (ViT-5-Small) ────────────────────────────────────────────────────────
HIDDEN_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 16
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14

INIT_FN = trunc_normal_init(std=0.02)
INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)

# ─── Training recipe ────────────────────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS = 800
IMAGENET_TRAIN_SIZE = 1_281_167
EFFECTIVE_BATCH_SIZE = 2048
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS

LEARNING_RATE = 4e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"

NUM_WORKERS = 12


def make_block_cfg(sequence_mixer_cfg: LazyConfig) -> LazyConfig:
    """Build a ViT5ResidualBlock config with the given sequence mixer.

    Everything except the mixer is shared across all v3 configs: MLP,
    norms, LayerScale, and DropPath.

    Args:
        sequence_mixer_cfg: LazyConfig for the sequence mixer
            (e.g. ViT5Attention, ViT5HyenaAdapter).

    Returns:
        LazyConfig for a ViT5ResidualBlock.
    """
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=sequence_mixer_cfg,
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=HIDDEN_DIM,
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=LAYER_SCALE_INIT,
        drop_path_rate=DROP_PATH_RATE,
    )


def get_base_config() -> ExperimentConfig:
    """Return the shared ViT-5-Small pretrain config (everything except the network).

    Callers must set ``config.net`` to their mixer-specific network before
    returning the config.
    """
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True
    config.compile_mode = "max-autotune"

    # ─── Dataset (fused DALI + local staging) ─────────────────────────────
    config.dataset = LazyConfig(DALIImageNetFusedDataModule)(
        data_dir=IMAGENET_PATH,
        imagefolder_dir=IMAGENET_FOLDER_PATH,
        prefetch_factor=3,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=config.seed,
        image_size=IMAGE_SIZE,
        final_image_size=FINAL_IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        drop_labels=False,
        task="classification",
        mixup_cfg=LazyConfig(MixupConfig)(
            mixup=0.8,
            cutmix=1.0,
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
            smoothing=0.0,
        ),
        augment_cfg=LazyConfig(AugmentConfig)(
            use_three_augment=True,
            color_jitter=0.3,
        ),
        device_id=0,
        local_staging_dir=f"/scratch/{os.environ.get('USER', 'unknown')}/imagenet_dataset",
    )

    # ─── Lightning wrapper ──────────────────────────────────────────────────
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    # ─── Optimizer (Apex FusedLAMB) ─────────────────────────────────────────
    config.optimizer = LazyConfig(Lamb)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ─── Training ───────────────────────────────────────────────────────────
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TOTAL_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=4,
        checkpoint_every_n_steps=5000,
    )

    # ─── Scheduler ──────────────────────────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── EMA ────────────────────────────────────────────────────────────────
    config.callbacks = [LazyConfig(LabeledEMAWeightAveraging)(decay=0.99996)]
    config.trainer.checkpoint_monitor = "val/acc_ema"

    # ─── Wandb ──────────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="vit5_imagenet_pretrain",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ─── Auto-resume ──────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(
        enabled=False,
    )

    return config
