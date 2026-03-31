"""UNet-Hyena config for MHD_64 dataset.

Magneto-hydrodynamic turbulence: 64x64x64 spatial resolution (3D), 7 fields.
Periodic boundary conditions -> circular FFT padding.

Uses the unified UNet with Hyena blocks (gated global convolution via
CKConvND with SIREN kernels + FFN).
"""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.unet import HyenaBlock, WellUNet


PLACEHOLDER = None

# Dataset parameters
DATA_TYPE = "volume"
DATA_DIM = 3
WELL_BASE_PATH = os.environ.get("WELL_DATA_PATH", "./data/the_well")
WELL_DATASET_NAME = "MHD_64"

# Data parameters
N_STEPS_INPUT = 4
N_STEPS_OUTPUT = 1
MAX_ROLLOUT_STEPS = 1

N_FIELDS = 7
N_CONSTANT_FIELDS = 0
IN_CHANNELS = N_STEPS_INPUT * N_FIELDS + N_CONSTANT_FIELDS
OUT_CHANNELS = N_FIELDS

SPATIAL_RESOLUTION = (64, 64, 64)

# UNet parameters
BATCH_SIZE = 8
INIT_FEATURES = 42
BLOCKS_PER_STAGE = 2
STAGES = 3  # 64 -> 32 -> 16 -> 8 (neck)
BLOCKS_AT_NECK = 1
MLP_RATIO = 4
GRADIENT_CHECKPOINTING = True  # 3D is memory-heavy

# Hyena / SIREN parameters
OMEGA_0 = 100.0
SIREN_LAYERS = 3
SIREN_HIDDEN_DIM = 64

# Training parameters
TRAINING_ITERATIONS = 260_000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
NUM_WORKERS = 8
GRAD_CLIP = 1.0

WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3


def get_config() -> ExperimentConfig:
    """Returns the experiment configuration."""
    config = ExperimentConfig()

    config.debug = False
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    config.dataset = LazyConfig(WellDataModule)(
        well_base_path=WELL_BASE_PATH,
        well_dataset_name=WELL_DATASET_NAME,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_normalization=True,
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        min_dt_stride=1,
        max_dt_stride=1,
        local_staging_dir=None,
    )

    config.net = LazyConfig(WellUNet)(
        dim_in=IN_CHANNELS,
        dim_out=OUT_CHANNELS,
        n_spatial_dims=DATA_DIM,
        spatial_resolution=SPATIAL_RESOLUTION,
        stages=STAGES,
        blocks_per_stage=BLOCKS_PER_STAGE,
        blocks_at_neck=BLOCKS_AT_NECK,
        init_features=INIT_FEATURES,
        block_cfg=LazyConfig(HyenaBlock)(
            n_spatial_dims=DATA_DIM,
            mlp_ratio=MLP_RATIO,
            omega_0=OMEGA_0,
            siren_layers=SIREN_LAYERS,
            siren_hidden_dim=SIREN_HIDDEN_DIM,
        ),
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
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
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision="bf16-mixed",
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    config.wandb = WandbConfig(
        entity="implicit-long-convs",
        project="nvsubquadratic",
        job_group="MHD_64_unet_hyena",
    )

    return config
