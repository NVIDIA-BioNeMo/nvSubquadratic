# TODO: Add license header here

"""Config for ARC-AGI grid-to-grid prediction with a lightweight residual MLP."""

import os

import torch

from experiments.datamodules.arc_agi import ArcAGIDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.arc_agi_wrapper import ArcAGIWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


PLACEHOLDER = None

# Dataset
BATCH_SIZE = 64
PRECISION = "bf16-mixed"
MAX_WORKERS = min(16, os.cpu_count() or 1)

# Model
HIDDEN_DIM = 128
NUM_BLOCKS = 4
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1

# Training
TRAINING_ITERATIONS = 200_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
GRAD_CLIP = 1.0

WEIGHT_DECAY = 0.01
LEARNING_RATE = 2e-3


def get_config() -> ExperimentConfig:
    """Return the ExperimentConfig for ARC-AGI grid prediction."""
    config = ExperimentConfig()

    # Dataset
    config.dataset = LazyConfig(ArcAGIDataModule)(
        data_dir=".data/arc_agi",
        batch_size=BATCH_SIZE,
        num_workers=MAX_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
        include_test_pairs=False,
        normalize_inputs=True,
        one_hot_inputs=False,
        max_grid_size=None,  # infer from data
        input_pad_value=0,
        label_pad_value=-100,
    )

    # Network
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=PLACEHOLDER,
        out_channels=PLACEHOLDER,
        num_blocks=NUM_BLOCKS,
        hidden_dim=HIDDEN_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=HIDDEN_DIM),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=HIDDEN_DIM, out_features=PLACEHOLDER),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=HIDDEN_DIM),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            sequence_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="gelu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
            ),
            mlp_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=HIDDEN_DIM),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
        condition_in_proj_cfg=None,
    )

    # Lightning wrapper
    config.lightning_wrapper_class = LazyConfig(ArcAGIWrapper)(ignore_index=-100)

    # Optimizer
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Training
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    # Scheduler
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
    )

    # Wandb
    config.wandb = WandbConfig(job_group="arc_agi_grid")

    return config

