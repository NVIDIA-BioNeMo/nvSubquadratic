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

"""Shared config for rayleigh_taylor_instability experiments (v2).

Dataset: rayleigh_taylor_instability
    - 3D, 128³, 4 dynamic fields, 0 constant fields
    - Dynamic fields: density, velocity_{x,y,z}
    - ~45 train trajectories × 119 timesteps
    - All models fail (VRMSE > 10); included as open problem
    - CNextU-net best LR: 5e-3 (paper Table 6)
"""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig


# ─── Dataset constants ────────────────────────────────────────────────────────
DATA_DIM = 3
SPATIAL_RESOLUTION = (128, 128, 128)
WELL_BASE_PATH = os.environ.get(
    "WELL_DATA_PATH",
    "/shared/data/image_datasets/the_well/datasets",
)
WELL_DATASET_NAME = "rayleigh_taylor_instability"

N_STEPS_INPUT = 4
N_STEPS_OUTPUT = 1
MAX_ROLLOUT_STEPS = 100

BOUNDARY_CONDITIONS = {"x": "PERIODIC", "y": "PERIODIC", "z": "WALL"}

N_FIELDS = 4  # density, velocity_{x,y,z}
N_CONSTANT_FIELDS = 0
IN_CHANNELS = N_STEPS_INPUT * N_FIELDS + N_CONSTANT_FIELDS  # 16
OUT_CHANNELS = N_FIELDS  # 4

# 45 train trajectories × 115 windows = 5,175 samples
SAMPLES_PER_EPOCH = 5_175

# ─── Training constants (shared across models) ───────────────────────────────
TRAINING_ITERATIONS = 110_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05

BATCH_SIZE = 2  # configs/data/rayleigh_taylor_instability.yaml
NUM_WORKERS = 12
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4


def get_base_config(
    *,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> ExperimentConfig:
    """Return a config with everything except ``config.net`` and compile flags."""
    config = ExperimentConfig()
    config.debug = False

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

    config.lightning_wrapper_class = LazyConfig(WELLRegressionWrapper)(
        metadata=PLACEHOLDER,
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        metric="MSE",
    )

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )
    config.mp_sharing_strategy = None

    config.trainer.samples_per_epoch = SAMPLES_PER_EPOCH
    config.trainer.check_val_every_n_iterations = "${eval:'${trainer.samples_per_epoch} // (${train.batch_size} * 2)'}"
    config.trainer.checkpoint_every_n_steps = "${trainer.check_val_every_n_iterations}"

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    config.wandb = WandbConfig(
        entity="implicit-long-convs",
        project="nvsubquadratic",
        job_group=WELL_DATASET_NAME,
    )

    return config
