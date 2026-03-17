"""ViT-5-Small ImageNet-1k — Validation only with fused DALI dataloader.

Loads pretrained weights from W&B run 2y06y121 and validates using the
DALIImageNetFusedDataModule (all augmentations inside DALI pipeline).
No training is performed.

Requires: pip install nvidia-dali-cuda120
"""

import os

import torch
from apex.optimizers import FusedLAMB as Lamb
from experiments.datamodules.imagenet import AugmentConfig, MixupConfig

from experiments.datamodules.dali_imagenet_fused import DALIImageNetFusedDataModule
from experiments.default_cfg import (
    AutoResumeConfig,
    ExperimentConfig,
    SchedulerConfig,
    StartFromCheckpointConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from experiments.utils.checkpointing import StripCompiledPrefix
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


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
NUM_HEADS = 6
PATCH_SIZE = 16
NUM_REGISTERS = 4
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE

BATCH_SIZE = 256
PRECISION = "bf16-mixed"
NUM_WORKERS = 12


def get_config() -> ExperimentConfig:
    """Return validation-only config with fused DALI datamodule."""
    config = ExperimentConfig()
    config.debug = True
    config.seed = 42
    config.compile = False

    # ─── Dataset (fused DALI) ────────────────────────────────────────────
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
    )

    # ─── Network ────────────────────────────────────────────────────────────
    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_patches_h=NUM_PATCHES_H,
                num_patches_w=NUM_PATCHES_W,
                num_registers=NUM_REGISTERS,
                qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // NUM_HEADS, eps=1e-6),
                rope_base=10000.0,
                reg_rope_base=100.0,
                attn_dropout=0.0,
                proj_dropout=0.0,
                qkv_bias=False,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=float(MLP_RATIO),
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=LAYER_SCALE_INIT,
            drop_path_rate=DROP_PATH_RATE,
        ),
    )

    # ─── Lightning wrapper ──────────────────────────────────────────────────
    # NOTE: The ViT-5 reference uses BCE for pretraining, but we observed that
    # pretraining with BCE leads to significantly lower finetuning accuracy
    # (~76%) compared to SoftTargetCE (~82%).
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    # ─── Optimizer (required by run.py even for validation) ──────────────
    config.optimizer = LazyConfig(Lamb)(
        params=PLACEHOLDER,
        lr=4e-3,
        weight_decay=0.05,
    )

    # ─── Training (disabled — validation only) ──────────────────────────
    config.train = TrainConfig(
        do=False,
        batch_size="${dataset.batch_size}",
        iterations=0,
        precision=PRECISION,
    )

    config.trainer = TrainerConfig()

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.00625,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── Load weights from W&B run 2y06y121 ─────────────────────────────
    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        run_path="implicit-long-convs/nvsubquadratic/2y06y121",
        alias="latest",
        strict=True,
        callbacks=[LazyConfig(StripCompiledPrefix)()],
    )

    config.wandb = WandbConfig(
        job_group="vit5_imagenet_validate",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    config.autoresume = AutoResumeConfig(enabled=False)

    return config
