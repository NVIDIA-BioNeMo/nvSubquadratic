"""Shared finetune base config for v3_wessels ViT-5 experiments.

Fine-tuning recipe follows the ViT-5 paper (Table 13):
- AdamW lr=1e-5, wd=0.1, cosine schedule, 25% warmup (5/20 epochs)
- 20 epochs, effective batch 512 (256/GPU × 2 GPUs)
- RandAugment rand-m9-mstd0.5-inc1, Mixup 0.8, CutMix 1.0
- Label smoothing 0.1, no ThreeAugment, Random Erasing 0.25
- No gradient clipping, EMA decay=0.99996
"""

import torch

from examples.vit5_imagenet.v3_wessels._base_config import get_base_config
from experiments.datamodules.dali_imagenet_fused import AugmentConfig, MixupConfig
from experiments.default_cfg import (
    AutoResumeConfig,
    ExperimentConfig,
    SchedulerConfig,
    StartFromCheckpointConfig,
    TrainConfig,
    TrainerConfig,
)
from experiments.utils.checkpointing import StripCompiledPrefix
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig

# ─── Fine-tuning recipe ─────────────────────────────────────────────────────
BATCH_SIZE = 256  # per GPU; 256 × 2 GPUs = 512 effective (Table 13)
EPOCHS = 20
IMAGENET_TRAIN_SIZE = 1_281_167
NUM_GPUS = 2
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * NUM_GPUS  # 512
ITERS_PER_EPOCH = IMAGENET_TRAIN_SIZE // EFFECTIVE_BATCH_SIZE
TOTAL_ITERATIONS = EPOCHS * ITERS_PER_EPOCH
WARMUP_EPOCHS = 5
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS  # 0.25

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.1


def get_finetune_base_config(*, pretrained_run_id: str) -> ExperimentConfig:
    """Return a finetune base config, loading pretrained weights from W&B.

    Starts from the v3 pretrain base (for dataset/Snellius paths, compile settings)
    and overrides training params for the ViT-5 finetune recipe.

    Args:
        pretrained_run_id: W&B run ID for the pretrained checkpoint
            (e.g. "nxm3i7g6").
    """
    config = get_base_config()

    # ─── Dataset overrides (finetune augmentation) ───────────────────────
    config.dataset.batch_size = BATCH_SIZE
    config.dataset.mixup_cfg = LazyConfig(MixupConfig)(
        mixup=0.8,
        cutmix=1.0,
        mixup_prob=1.0,
        mixup_switch_prob=0.5,
        mixup_mode="batch",
        smoothing=0.1,
    )
    config.dataset.augment_cfg = LazyConfig(AugmentConfig)(
        use_three_augment=False,
        color_jitter=0.3,
        rand_augment="rand-m9-mstd0.5-inc1",
        random_erasing_prob=0.25,
        random_erasing_mode="pixel",
    )

    # ─── Optimizer (AdamW for fine-tuning) ───────────────────────────────
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ─── Training ────────────────────────────────────────────────────────
    config.train = TrainConfig(
        do=True,
        batch_size="${dataset.batch_size}",
        iterations=TOTAL_ITERATIONS,
        grad_clip=0.0,
        precision="bf16-mixed",
        accumulate_grad_steps=1,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=1,
        checkpoint_monitor="val/acc_ema",
    )

    # ─── Scheduler ───────────────────────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── Load pretrained weights ─────────────────────────────────────────
    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        run_path=f"implicit-long-convs/nvsubquadratic/{pretrained_run_id}",
        alias="best",
        strict=True,
        callbacks=[LazyConfig(StripCompiledPrefix)()],
    )

    # ─── Wandb ───────────────────────────────────────────────────────────
    config.wandb.job_group = "vit5_imagenet_finetune"

    # ─── Auto-resume ─────────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(enabled=False)

    return config
