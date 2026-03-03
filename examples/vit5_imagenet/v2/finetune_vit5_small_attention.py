"""ViT-5-Small attention baseline — ImageNet-1k fine-tuning.

Fine-tunes from pretrained run zyf3ky33 (val/acc 81.61) using the ViT-5
paper's fine-tuning recipe (Table 13):

- AdamW lr=1e-5, wd=0.1, cosine schedule, 5-epoch warmup
- 20 epochs, batch 256/GPU × 2 GPUs = effective 512
- RandAugment rand-m9-mstd0.5-inc1, Mixup 0.8, CutMix 1.0, no Random Erasing
- Label smoothing 0.1, no ThreeAugment, no gradient clipping
- EMA decay=0.99996 (val/acc and val/loss are EMA metrics)
- Uses ImageNetDataModule (torchvision) for RandAugment + Random Erasing support
"""

import os

import torch

from experiments.callbacks.model_ema import LabeledEMAWeightAveraging
from experiments.datamodules.imagenet import AugmentConfig, ImageNetDataModule, MixupConfig
from experiments.default_cfg import (
    AutoResumeConfig,
    ExperimentConfig,
    SchedulerConfig,
    StartFromCheckpointConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.utils.checkpointing import StripCompiledPrefix
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

# ─── Model (ViT-5-Small, attention) ────────────────────────────────────────────
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

# ─── Fine-tuning recipe (Table 13) ──────────────────────────────────────────────
BATCH_SIZE = 256
EPOCHS = 20
IMAGENET_TRAIN_SIZE = 1_281_167
NUM_GPUS = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * NUM_GPUS  # 512
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE  # 2502
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH  # 50040
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS  # 0.25

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1
PRECISION = "bf16-mixed"
NUM_WORKERS = 12

# ─── Pretrained checkpoint ──────────────────────────────────────────────────────
PRETRAINED_RUN_PATH = "implicit-long-convs/nvsubquadratic/qyjyx58f"


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small attention fine-tuning config."""
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True

    # ─── Dataset (torchvision, not DALI — for RandAugment + Random Erasing) ──
    config.dataset = LazyConfig(ImageNetDataModule)(
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
        eval_crop_ratio=1.0,
        mixup_cfg=LazyConfig(MixupConfig)(
            mixup=0.8,
            cutmix=1.0,
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
            smoothing=0.1,
        ),
        augment_cfg=LazyConfig(AugmentConfig)(
            use_three_augment=False,
            color_jitter=0.3,
            rand_augment="rand-m9-mstd0.5-inc1",
            random_erasing_prob=0.0,
            random_erasing_mode="pixel",
        ),
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

    # ─── Optimizer (AdamW for fine-tuning) ──────────────────────────────────
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ─── Training ───────────────────────────────────────────────────────────
    config.train = TrainConfig(
        do=True,
        batch_size="${dataset.batch_size}",
        iterations=TOTAL_ITERATIONS,
        grad_clip=0.0,
        precision=PRECISION,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=1,
        checkpoint_monitor="val/acc_ema",
    )

    # ─── Scheduler ──────────────────────────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── Load pretrained weights ────────────────────────────────────────────
    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        run_path=PRETRAINED_RUN_PATH,
        alias="best",
        strict=True,
        callbacks=[LazyConfig(StripCompiledPrefix)()],
    )

    # ─── Wandb ──────────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="vit5_imagenet_finetune",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ─── Auto-resume ────────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(enabled=False)

    # ─── EMA callback ──────────────────────────────────────────────────────
    config.callbacks = [
        LazyConfig(LabeledEMAWeightAveraging)(decay=0.99996),
    ]

    return config
