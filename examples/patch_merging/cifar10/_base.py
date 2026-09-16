# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""CIFAR-10 training recipe for the patch-merging ablation.

Both the isotropic and hierarchical Hyena models are trained with identical
hyperparameters so that any accuracy difference is attributable to the
architecture (flat vs. multiscale pyramid) alone.

Key design choices:
  - CIFAR-10 images are resized to 224×224 so both architectures operate at
    their designed resolution without any structural changes.
  - AdamW replaces LAMB (no apex dependency for local runs).
  - EMA is omitted; checkpoint monitor targets val/acc directly.
  - Mixup α=0.2 is applied to reduce overfitting on the small dataset.
  - 200 epochs, batch_size=128 → ~78 k optimizer steps.
    Wall-clock on a single RTX 3090: roughly 3-4 h per run.

Environment variables:
  CIFAR10_PATH   local directory for data (default: /tmp/cifar10, auto-downloaded)
  WANDB_ENTITY   W&B entity override (default: implicit-long-convs)
"""

import os

import torch

from experiments.datamodules.cifar10_hf import (
    CIFAR10_NUM_CLASSES,
    CIFAR10_TRAIN_SIZE,
    AugmentConfig,
    CIFAR10DataModule,
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


# ── Dataset ───────────────────────────────────────────────────────────────────

CIFAR10_DATA_DIR = os.environ.get("CIFAR10_PATH", ".data/cifar10")
IMAGE_SIZE = 224  # resize 32×32 → 224×224; both architectures work unchanged
NUM_CLASSES = CIFAR10_NUM_CLASSES  # 10

# ── Training recipe ───────────────────────────────────────────────────────────

BATCH_SIZE = 128
EPOCHS = 25
ITERS_PER_EPOCH = CIFAR10_TRAIN_SIZE // BATCH_SIZE  # 50 000 // 128 = 390
WARMUP_EPOCHS = 2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"
NUM_WORKERS = 4

WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "implicit-long-convs")


def get_base_config(epochs: int = EPOCHS) -> ExperimentConfig:
    """Return a pre-filled ExperimentConfig for CIFAR-10 comparison runs.

    Caller must set ``config.net`` and optionally override ``config.compile``.
    """
    config = ExperimentConfig()
    config.debug = True  # offline wandb — no API key required for local runs
    config.seed = 42
    config.compile = False

    # ── Dataset ───────────────────────────────────────────────────────────────
    config.dataset = LazyConfig(CIFAR10DataModule)(
        data_dir=CIFAR10_DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=config.seed,
        image_size=32,
        final_image_size=IMAGE_SIZE,
        num_classes=NUM_CLASSES,
        drop_labels=False,
        task="classification",
        mixup_cfg=LazyConfig(MixupConfig)(
            mixup=0.2,
            cutmix=0.0,
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
            smoothing=0.1,
        ),
        augment_cfg=LazyConfig(AugmentConfig)(
            use_three_augment=False,
            color_jitter=0.3,
            rand_augment=None,
            random_erasing_prob=0.0,
        ),
    )

    # ── Lightning wrapper ─────────────────────────────────────────────────────
    # soft_target_ce is required when mixup is active (labels are soft distributions).
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    # ── Optimizer (AdamW — no apex required) ──────────────────────────────────
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    total_iters = epochs * ITERS_PER_EPOCH
    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=total_iters,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    # ── Trainer / checkpointing ───────────────────────────────────────────────
    config.trainer = TrainerConfig(
        check_val_every_n_epoch=5,  # validates at epochs 5, 10, 15, 20, 25
        checkpoint_every_n_steps=2000,
        checkpoint_monitor="val/acc",  # no EMA for local runs
    )

    # ── LR scheduler (cosine + linear warmup) ────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_EPOCHS / epochs,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="patch_merging_cifar10",
        entity=WANDB_ENTITY,
        project="nvsubquadratic",
    )

    config.autoresume = AutoResumeConfig(enabled=False)
    config.callbacks = []

    return config
