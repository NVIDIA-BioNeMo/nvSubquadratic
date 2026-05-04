"""Shared config for euler_multi_quadrants_periodicBC experiments (v2).

Dataset: euler_multi_quadrants_periodicBC
    - 2D, 512×512, 5 dynamic fields, 0 constant fields
    - Dynamic fields: energy, density, pressure, momentum_{x,y}
    - 4000 train trajectories × 101 timesteps
    - CNextU-net best LR: 5e-3 (paper Table 6)
"""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig


# ─── Dataset constants ────────────────────────────────────────────────────────
DATA_DIM = 2
SPATIAL_RESOLUTION = (512, 512)
WELL_BASE_PATH = os.environ.get(
    "WELL_DATA_PATH",
    "/shared/data/image_datasets/the_well/datasets",
)
WELL_DATASET_NAME = "euler_multi_quadrants_periodicBC"

N_STEPS_INPUT = 4
N_STEPS_OUTPUT = 1
MAX_ROLLOUT_STEPS = 100

BOUNDARY_CONDITIONS = {"x": "PERIODIC", "y": "PERIODIC"}

N_FIELDS = 5  # energy, density, pressure, momentum_{x,y}
N_CONSTANT_FIELDS = 0
IN_CHANNELS = N_STEPS_INPUT * N_FIELDS + N_CONSTANT_FIELDS  # 20
OUT_CHANNELS = N_FIELDS  # 5

# 4000 train trajectories × 97 windows = 388,000 samples
SAMPLES_PER_EPOCH = 388_000

# ─── Training constants (shared across models) ───────────────────────────────
TRAINING_ITERATIONS = 110_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05

BATCH_SIZE = 48  # configs/data/euler_multi_quadrants_periodicBC.yaml
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
