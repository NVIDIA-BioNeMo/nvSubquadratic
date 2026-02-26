"""ViT-5-Large ImageNet-1k — DDP benchmark.

Short benchmark run (~200 steps) to measure DDP throughput on ViT-5-L (~304M params).
Uses Apex FusedLAMB + fused DALI dataloader, batch_size=256 per GPU, 8 GPUs.
"""

import os

import torch
from apex.optimizers import FusedLAMB as Lamb

from experiments.datamodules.dali_imagenet_fused import DALIImageNetFusedDataModule
from experiments.datamodules.imagenet import AugmentConfig, MixupConfig
from experiments.default_cfg import AutoResumeConfig, ExperimentConfig, SchedulerConfig, TrainConfig, TrainerConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
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

# ─── Model (ViT-5-Large) ────────────────────────────────────────────────────────
HIDDEN_DIM = 1024
NUM_BLOCKS = 24
NUM_HEADS = 16
PATCH_SIZE = 16
NUM_REGISTERS = 4
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.1
MLP_RATIO = 4
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE

# ─── Training recipe (short benchmark) ──────────────────────────────────────────
BATCH_SIZE = 256
TOTAL_ITERATIONS = 200
LEARNING_RATE = 4e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
NUM_WORKERS = 12


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Large DDP benchmark config."""
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = False

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
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(use_bce_loss=True)

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
        precision="bf16-mixed",
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=9999,
        checkpoint_every_n_steps=9999,
    )

    # ─── Scheduler ──────────────────────────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── Wandb ──────────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="vit5L_bench_ddp",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ─── Auto-resume ──────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(enabled=False)

    return config
