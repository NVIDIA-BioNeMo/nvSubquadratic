"""Shared config for euler_multi_quadrants_periodicBC experiments.

All dataset, optimizer, scheduler, and training parameters live here.
Model configs (cfg_*.py) import these constants and call
``get_base_config(...)`` to get a pre-filled ExperimentConfig, then only
need to set ``config.net`` and compile flags.

Hyperparameters sourced from the Well benchmark codebase:
    configs/data/euler_multi_quadrants_periodicBC.yaml  → batch_size default
    configs/config.yaml                                  → n_steps_input/output, data_workers
    configs/trainer/defaults.yaml                        → epochs, max_rollout_steps
    configs/lr_scheduler/cosine_with_warmup.yaml        → warmup_epochs
    configs/optimizer/adam.yaml                          → default optimizer

Dataset: euler_multi_quadrants_periodicBC
    - 2D, 512×512, 5 fields, 0 constant fields
    - 3000 trajectories × 101 timesteps → 388,000 training samples
"""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import LazyConfig


PLACEHOLDER = None

# ─── Dataset constants ────────────────────────────────────────────────────────
DATA_DIM = 2
SPATIAL_RESOLUTION = (512, 512)
WELL_BASE_PATH = os.environ.get(
    "WELL_DATA_PATH",
    "/shared/data/image_datasets/the_well/datasets",
)
WELL_DATASET_NAME = "euler_multi_quadrants_periodicBC"

N_STEPS_INPUT = 4  # configs/config.yaml
N_STEPS_OUTPUT = 1  # configs/config.yaml
MAX_ROLLOUT_STEPS = 100  # configs/trainer/defaults.yaml

N_FIELDS = 5
N_CONSTANT_FIELDS = 0
IN_CHANNELS = N_STEPS_INPUT * N_FIELDS + N_CONSTANT_FIELDS  # 20
OUT_CHANNELS = N_FIELDS  # 5

# ─── Training constants (shared across models) ───────────────────────────────
EPOCHS = 200  # configs/trainer/defaults.yaml
WARMUP_EPOCHS = 5  # configs/lr_scheduler/cosine_with_warmup.yaml
WARMUP_ITERATIONS_PERCENTAGE = WARMUP_EPOCHS / EPOCHS

NUM_WORKERS = 14  # configs/config.yaml
GRAD_CLIP = 1.0
PRECISION = "bf16-mixed"


def get_base_config(
    *,
    batch_size: int,
    learning_rate: float,
    weight_decay: float = 1e-4,
) -> ExperimentConfig:
    """Return a config with everything except ``config.net`` and compile flags.

    Args:
        batch_size: Per-GPU batch size.
        learning_rate: Peak learning rate (from Table 6 of the paper).
        weight_decay: AdamW weight decay (default 1e-4 from optimizer/adam.yaml).
    """
    # iterations = samples_per_epoch / batch_size * epochs
    samples_per_epoch = 388_000
    iters_per_epoch = samples_per_epoch // batch_size
    training_iterations = iters_per_epoch * EPOCHS

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
        iterations=training_iterations,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

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
