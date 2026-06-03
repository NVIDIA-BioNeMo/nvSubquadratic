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

"""ViT-5-Small ImageNet-1k pretraining — v5 shared training recipe.

This base owns everything **except** the network: dataset, optimizer,
scheduler, callbacks, wandb, auto-resume, and training loop settings.

Instance configs (e.g. ``attention_pretrain.py``, ``hyena_gap_pretrain.py``)
call ``get_base_config(...)`` to get a pre-filled ExperimentConfig, then
set ``config.net`` and compile flags.

Training recipe (shared defaults):
- 800 epochs, batch 256/GPU, effective batch 2048 (8 GPUs)
- Optimizer: Apex FusedLAMB, lr=4e-3, wd=0.05, grad_clip=1.0
- Scheduler: Cosine with 5-epoch warmup
- EMA decay 0.99996, SoftTargetCE loss, bf16-mixed
- 3-Augment, Mixup 0.8, CutMix 1.0
- DALI fused data pipeline with local NVMe staging
"""

import os

from apex.optimizers import FusedLAMB as Lamb

from experiments.callbacks.iteration_speed import IterationSpeedCallback
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


# ─── Dataset constants ───────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 224
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
IMAGENET_FOLDER_PATH = os.environ.get("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

# ─── Model constants (shared across mixers) ─────────────────────────────────────
HIDDEN_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 16
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE  # 14
LAYER_SCALE_INIT = 1e-4
MLP_RATIO = 4

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


def get_base_config(
    *,
    lr: float = LEARNING_RATE,
    wd: float = WEIGHT_DECAY,
    epochs: int = EPOCHS,
    grad_clip: float = GRAD_CLIP,
    ema_decay: float = 0.99996,
    # ─── Augmentation ────────────────────────────────────────────────
    mixup: float = 0.8,
    cutmix: float = 1.0,
    smoothing: float = 0.0,
    use_three_augment: bool = True,
    color_jitter: float = 0.3,
    rand_augment: str | None = None,
    random_erasing_prob: float = 0.0,
    num_repeats: int = 1,
    # ─── Extra callbacks ─────────────────────────────────────────────
    extra_callbacks: list | None = None,
) -> ExperimentConfig:
    """Return a config with everything except ``config.net`` and compile flags.

    Args:
        lr: Learning rate for LAMB.
        wd: Global weight decay.
        epochs: Number of pretraining epochs.
        grad_clip: Gradient clipping norm.
        ema_decay: EMA decay rate.
        mixup: Mixup alpha (0.0 = disabled).
        cutmix: CutMix alpha (0.0 = disabled).
        smoothing: Label smoothing.
        use_three_augment: Use three-augment instead of RandAugment.
        color_jitter: Color jitter factor.
        rand_augment: RandAugment config string (when ``use_three_augment=False``).
        random_erasing_prob: Random erasing probability.
        num_repeats: Repeated augmentation factor (1 = disabled).
        extra_callbacks: Additional callbacks appended after EMA + IterationSpeed.
    """
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42

    # ─── Dataset (fused DALI + local NVMe staging) ───────────────────
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
            mixup=mixup,
            cutmix=cutmix,
            mixup_prob=1.0,
            mixup_switch_prob=0.5,
            mixup_mode="batch",
            smoothing=smoothing,
        ),
        augment_cfg=LazyConfig(AugmentConfig)(
            use_three_augment=use_three_augment,
            color_jitter=color_jitter,
            rand_augment=rand_augment,
            random_erasing_prob=random_erasing_prob,
            random_erasing_mode="pixel",
            num_repeats=num_repeats,
        ),
        device_id=0,
        local_staging_dir=f"/scratch/{os.environ.get('USER', 'unknown')}/imagenet_dataset",
    )

    # ─── Lightning wrapper ───────────────────────────────────────────
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

    # ─── Optimizer (Apex FusedLAMB) ──────────────────────────────────
    config.optimizer = LazyConfig(Lamb)(
        params=PLACEHOLDER,
        lr=lr,
        weight_decay=wd,
    )

    # ─── Training ────────────────────────────────────────────────────
    total_iters = epochs * ITERS_PER_EPOCH
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=total_iters,
        grad_clip=grad_clip,
        precision=PRECISION,
    )

    config.trainer = TrainerConfig(
        check_val_every_n_epoch=4,
        checkpoint_every_n_steps=5000,
        checkpoint_monitor="val/acc_ema",
    )

    # ─── Scheduler (cosine, warmup) ──────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="max",
    )

    # ─── Wandb ───────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        job_group="imagenet_v5_pretrain",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ─── Auto-resume ─────────────────────────────────────────────────
    config.autoresume = AutoResumeConfig(enabled=False)

    # ─── Callbacks ───────────────────────────────────────────────────
    config.callbacks = [
        LazyConfig(LabeledEMAWeightAveraging)(decay=ema_decay),
        LazyConfig(IterationSpeedCallback)(
            log_every_n_steps=10,
            batch_size_per_gpu=BATCH_SIZE,
        ),
    ]
    if extra_callbacks:
        config.callbacks.extend(extra_callbacks)

    return config
