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

"""ViT-5-Small Hyena CLS-row apex — ImageNet-1k fine-tuning.

Fine-tunes from pretrained run 96wy1zzj (val/acc 80.30) using the ViT-5
paper's fine-tuning recipe (Table 13):

- AdamW lr=1e-5, wd=0.1, cosine schedule, 5-epoch warmup
- 20 epochs, batch 256/GPU × 2 GPUs = effective 512
- RandAugment rand-m9-mstd0.5-inc1, Mixup 0.8, CutMix 1.0, no Random Erasing
- Label smoothing 0.1, no ThreeAugment, no gradient clipping
- EMA decay=0.99996 (val/acc and val/loss are EMA metrics)
- Uses DALIImageNetFusedDataModule for GPU-accelerated data loading
"""

import os

import torch

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
    StartFromCheckpointConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from experiments.utils.checkpointing import StripCompiledPrefix
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Dataset ────────────────────────────────────────────────────────────────────
INPUT_CHANNELS = 3
NUM_CLASSES = 1000
IMAGE_SIZE = 224
FINAL_IMAGE_SIZE = 224
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/shared/data/image_datasets/imagenet")
IMAGENET_FOLDER_PATH = os.environ.get("IMAGENET_FOLDER_PATH", "/shared/data/image_datasets/imagenet_folder")

# ─── Model (ViT-5-Small + Hyena, CLS-row) ───────────────────────────────────────
HIDDEN_DIM = 384
NUM_BLOCKS = 12
PATCH_SIZE = 16
LAYER_SCALE_INIT = 1e-4
DROP_PATH_RATE = 0.05
MLP_RATIO = 4
NUM_PATCHES_H = FINAL_IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES_W = FINAL_IMAGE_SIZE // PATCH_SIZE
NUM_REGISTERS = NUM_PATCHES_W - 1  # 13 — fills the extra row: [CLS, regs, patches] → (H'+1)×W' grid

# ─── Hyena / SIREN kernel hyperparameters ────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

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
PRETRAINED_RUN_PATH = "implicit-long-convs/nvsubquadratic/rmoaziyc"


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small Hyena CLS-row apex fine-tuning config."""
    config = ExperimentConfig()
    config.debug = False
    config.seed = 42
    config.compile = True

    # ─── Dataset (DALI with fused augmentations including RandAugment) ──
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
        device_id=0,
        local_staging_dir=f"/scratch/{os.environ.get('USER', 'unknown')}/imagenet_dataset",
    )

    # ─── Network (Hyena CLS-row) ────────────────────────────────────────────────
    hyena_mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=HIDDEN_DIM,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=NUM_PATCHES_H + 1,  # 15: grid is (H'+1)×W' due to the extra CLS row.
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * HIDDEN_DIM,
                out_channels=3 * HIDDEN_DIM,
                kernel_size=3,
                groups=3 * HIDDEN_DIM,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        ),
        init_method_in=small_init,
        init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
    )

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        readout="cls",
        prepend_registers=True,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=LazyConfig(ViT5ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
                inner_mixer_cfg=hyena_mixer_cfg,
                grid_w=NUM_PATCHES_W,
            ),
            sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=float(MLP_RATIO),
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            hidden_dim=HIDDEN_DIM,
            layer_scale_init=LAYER_SCALE_INIT,
            drop_path_rate=DROP_PATH_RATE,
        ),
    )

    # ─── Lightning wrapper ──────────────────────────────────────────────────
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)(loss="soft_target_ce")

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
        alias="latest",
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
