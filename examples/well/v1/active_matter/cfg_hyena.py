"""Config file for WELL benchmark: active_matter dataset with Hyena.

This config uses Hyena as the sequence mixer for the active matter dataset.
Hyena wraps CKConv with QKV projections, short depthwise convolution,
gating, and optional RoPE / QK-normalization.
"""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


PLACEHOLDER = None

# Dataset parameters
DATA_TYPE = "image"
DATA_DIM = 2
WELL_BASE_PATH = os.environ.get("WELL_DATA_PATH", "/gpfs/scratch1/shared/dwessels2/data/the_well/datasets")
WELL_DATASET_NAME = "active_matter"

# Data parameters (following WELL benchmark defaults)
N_STEPS_INPUT = 4  # Number of input timesteps
N_STEPS_OUTPUT = 1  # Number of output timesteps for training
MAX_ROLLOUT_STEPS = 1  # Maximum rollout for validation

N_FIELDS = 11
N_CONSTANT_FIELDS = 0
IN_CHANNELS = N_STEPS_INPUT * N_FIELDS + N_CONSTANT_FIELDS
OUT_CHANNELS = N_FIELDS

# Model parameters (overridable via environment variables for sweeps)
BATCH_SIZE = 16
NUM_HIDDEN_CHANNELS = int(os.environ.get("HYENA_HIDDEN_DIM", 512))
NUM_BLOCKS = int(os.environ.get("HYENA_DEPTH", 12))
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0
GRID_TYPE = "single"
FFT_PADDING = "circular"  # Appropriate for active matter (periodic boundaries)
OMEGA_0 = 100.0
PATCH_SIZE = int(os.environ.get("HYENA_PATCH_SIZE", 4))

# TRAINING parameters
TRAINING_ITERATIONS = 130_000  # ~200 epochs with batch_size=16 (10395 samples / 16 ≈ 650 iters/epoch)
WARMUP_ITERATIONS_PERCENTAGE = 0.1
NUM_WORKERS = 8
GRAD_CLIP = 1.0

WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3


def get_config() -> ExperimentConfig:
    """Get the configuration for the WELL active_matter experiment with Hyena.

    Returns:
        ExperimentConfig: The configuration for the experiment.
    """
    config = ExperimentConfig()

    config.debug = False
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.compile_compatible_fftconv = True

    # Add dataset config
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

    # Create norm config once to reuse
    norm_cfg = LazyConfig(torch.nn.RMSNorm)(normalized_shape=NUM_HIDDEN_CHANNELS)

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features=IN_CHANNELS,
            out_features=NUM_HIDDEN_CHANNELS,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features=NUM_HIDDEN_CHANNELS,
            out_features=OUT_CHANNELS,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        norm_cfg=norm_cfg,
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim=NUM_HIDDEN_CHANNELS,
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=DATA_DIM,
                        hidden_dim=NUM_HIDDEN_CHANNELS,
                        fft_padding=FFT_PADDING,
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim=DATA_DIM,
                            out_dim=NUM_HIDDEN_CHANNELS,
                            mlp_hidden_dim=96,
                            num_layers=3,
                            embedding_dim=32,
                            omega_0=OMEGA_0,
                            L_cache=256 // PATCH_SIZE,  # Matches patched spatial dim
                            use_bias=True,
                            hidden_omega_0=1.0,
                        ),
                        mask_cfg=LazyConfig(GaussianModulationND)(
                            data_dim=DATA_DIM,
                            num_channels=NUM_HIDDEN_CHANNELS,
                            min_attenuation_at_step=0.1,
                            max_attenuation_at_limit=0.95,
                            init_extent=1.0,
                            parametrization="direct",
                        ),
                        grid_type=GRID_TYPE,
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels=3 * NUM_HIDDEN_CHANNELS,
                        out_channels=3 * NUM_HIDDEN_CHANNELS,
                        kernel_size=3,
                        groups=3 * NUM_HIDDEN_CHANNELS,
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=NUM_HIDDEN_CHANNELS),
                    qk_norm_cfg=LazyConfig(L2Norm)(),
                    use_rope=False,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg=norm_cfg,
            # Condition mixer
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            # MLP
            mlp_cfg=LazyConfig(MLP)(
                dim=NUM_HIDDEN_CHANNELS,
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=norm_cfg,
            # Dropout
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
    )

    # Add lightning wrapper config
    config.lightning_wrapper_class = LazyConfig(WELLRegressionWrapper)(
        metadata=PLACEHOLDER,  # Will be filled from datamodule at instantiation time
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        metric="MSE",
    )

    # Add optimizer config
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Modify the train config
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision="bf16-mixed",
    )

    # Modify the scheduler config
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # Add wandb config
    config.wandb = WandbConfig(
        entity="implicit-long-convs",
        project="nvsubquadratic",
        job_group="active_matter_hyena",
    )

    return config
