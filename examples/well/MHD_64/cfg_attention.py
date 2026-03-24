"""Config file for WELL benchmark: MHD_64 dataset with Attention.

Magneto-hydrodynamic turbulence: 64x64x64 spatial resolution (3D), 4 fields.
Same architecture as cfg_hyena.py but with multi-head attention replacing Hyena.
"""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init


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

# Model parameters
BATCH_SIZE = 2
NUM_HIDDEN_CHANNELS = 256
NUM_BLOCKS = 8
NUM_HEADS = 8
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0
PATCH_SIZE = 4
SPATIAL_RESOLUTION = 64

# Training parameters
TRAINING_ITERATIONS = 130_000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
NUM_WORKERS = 8
GRAD_CLIP = 1.0

WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3


def get_config() -> ExperimentConfig:
    """Returns the experiment configuration."""
    config = ExperimentConfig()

    config.debug = False
    config.compile = False

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

    norm_cfg = LazyConfig(torch.nn.RMSNorm)(normalized_shape=NUM_HIDDEN_CHANNELS)

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=PLACEHOLDER,
        out_channels=PLACEHOLDER,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features=PLACEHOLDER,
            out_features=PLACEHOLDER,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features=PLACEHOLDER,
            out_features=PLACEHOLDER,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        norm_cfg=norm_cfg,
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim=NUM_HIDDEN_CHANNELS,
                mixer_cfg=LazyConfig(Attention)(
                    hidden_dim=NUM_HIDDEN_CHANNELS,
                    num_heads=NUM_HEADS,
                    apply_qk_norm=True,
                    use_rope=False,
                    is_causal=False,
                    attn_dropout=0.0,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg=norm_cfg,
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim=NUM_HIDDEN_CHANNELS,
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=norm_cfg,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
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
        job_group="MHD_64_attention",
    )

    return config
