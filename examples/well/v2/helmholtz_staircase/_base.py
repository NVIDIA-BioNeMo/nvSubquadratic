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

"""Shared config for helmholtz_staircase experiments (v2).

All dataset, optimizer, scheduler, and training parameters live here.
Model configs (cfg_*.py) import these constants and call
``get_base_config(...)`` to get a pre-filled ExperimentConfig, then only
need to set ``config.net`` and compile flags.

Hyperparameters sourced from the Well benchmark codebase:
    configs/data/helmholtz_staircase.yaml  → batch_size default (24)
    configs/config.yaml                    → n_steps_input/output, data_workers
    configs/trainer/defaults.yaml          → epochs, max_rollout_steps
    configs/lr_scheduler/cosine_with_warmup.yaml → warmup_epochs
    configs/optimizer/adam.yaml            → default optimizer

Dataset: helmholtz_staircase
    - 2D, 1024×256, 2 dynamic fields + 1 constant field (mask)
    - Dynamic fields: pressure_re, pressure_im
    - Constant fields: mask (binary obstacle geometry)
    - 416 train trajectories × 50 timesteps
    - Primary stress test for bf16 precision (high-frequency oscillatory solutions)
"""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig


# ─── Dataset constants ────────────────────────────────────────────────────────
DATA_DIM = 2
SPATIAL_RESOLUTION = (1024, 256)
WELL_BASE_PATH = os.environ.get(
    "WELL_DATA_PATH",
    "/shared/data/image_datasets/the_well/datasets",
)
WELL_DATASET_NAME = "helmholtz_staircase"

N_STEPS_INPUT = 4  # configs/config.yaml
N_STEPS_OUTPUT = 1  # configs/config.yaml
MAX_ROLLOUT_STEPS = 100  # configs/trainer/defaults.yaml

BOUNDARY_CONDITIONS = {"x": "OPEN", "y": ("OPEN", "WALL")}

N_FIELDS = 2  # pressure_re, pressure_im
N_CONSTANT_FIELDS = 1  # mask
IN_CHANNELS = N_STEPS_INPUT * N_FIELDS + N_CONSTANT_FIELDS  # 9
OUT_CHANNELS = N_FIELDS  # 2

# 16 files × 26 trajs = 416 trajs, 50 timesteps each → 46 windows
# Total: 416 × 46 = 19,136 samples
SAMPLES_PER_EPOCH = 19_136

# ─── Training constants (shared across models) ───────────────────────────────
TRAINING_ITERATIONS = 110_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05

BATCH_SIZE = 24  # configs/data/helmholtz_staircase.yaml
NUM_WORKERS = 12
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4


def get_base_config(
    *,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> ExperimentConfig:
    """Return a config with everything except ``config.net`` and compile flags.

    Args:
        batch_size: Per-GPU batch size (default from paper: 24).
        learning_rate: Peak learning rate (CNextU-net best: 5e-4).
        weight_decay: AdamW weight decay (default 1e-4 from optimizer/adam.yaml).
    """
    config = ExperimentConfig()
    config.debug = False

    # ── DataModule ────────────────────────────────────────────────────────
    config.dataset = LazyConfig(WellDataModule)(
        well_base_path=WELL_BASE_PATH,
        well_dataset_name=WELL_DATASET_NAME,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        use_normalization=True,
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        min_dt_stride=1,
        max_dt_stride=1,
        local_staging_dir=None,
    )

    # ── Lightning wrapper ─────────────────────────────────────────────────
    config.lightning_wrapper_class = LazyConfig(WELLRegressionWrapper)(
        metadata=PLACEHOLDER,
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        metric="MSE",
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ── Training ──────────────────────────────────────────────────────────
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )
    config.mp_sharing_strategy = None

    # ── Trainer: validate + checkpoint every half epoch ────────────────────
    config.trainer.samples_per_epoch = SAMPLES_PER_EPOCH
    config.trainer.check_val_every_n_iterations = "${eval:'${trainer.samples_per_epoch} // (${train.batch_size} * 2)'}"
    config.trainer.checkpoint_every_n_steps = "${trainer.check_val_every_n_iterations}"

    # ── Scheduler ─────────────────────────────────────────────────────────
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # ── WandB ─────────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        entity="implicit-long-convs",
        project="nvsubquadratic",
        job_group=WELL_DATASET_NAME,
    )

    return config
