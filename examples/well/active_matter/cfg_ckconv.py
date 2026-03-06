"""Config file for WELL benchmark: active_matter dataset with standalone CKConv.

This config uses CKConv (Continuous Kernel Convolution) as a standalone mixer
for the active matter dataset. CKConv uses learned convolutional kernels parameterized
by implicit neural representations (SIREN networks).
"""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


PLACEHOLDER = None

# Dataset parameters
DATA_TYPE = "image"
DATA_DIM = 2
WELL_BASE_PATH = os.environ.get("WELL_DATA_PATH", "./data/the_well")
WELL_DATASET_NAME = "active_matter"

# Data parameters (following WELL benchmark defaults)
N_STEPS_INPUT = 4  # Number of input timesteps
N_STEPS_OUTPUT = 1  # Number of output timesteps for training
MAX_ROLLOUT_STEPS = 1  # Maximum rollout for validation

# Model parameters
BATCH_SIZE = 8  # Start smaller for 256x256 images
NUM_HIDDEN_CHANNELS = 512  # Reduced from 1024 for memory efficiency
NUM_BLOCKS = 12  # Reduced from 20 for initial testing
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0
GRID_TYPE = "single"
FFT_PADDING = "circular"  # Appropriate for active matter (periodic boundaries)
OMEGA_0 = 100.0
PATCH_SIZE = 4  # 64x64 -> 16x16 = 256 tokens

# TRAINING parameters
TRAINING_ITERATIONS = 50_000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
NUM_WORKERS = 8
GRAD_CLIP = 1.0

WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3


class SimpleCKConvMixer(torch.nn.Module):
    """Simple wrapper to use CKConv as a sequence mixer.

    This adapter allows CKConv to be used directly in the residual block
    without the QKV projection pattern used by Hyena and Attention.
    """

    def __init__(self, ckconv_cfg: LazyConfig):
        """Initialize the SimpleCKConvMixer.

        Args:
            ckconv_cfg: LazyConfig for CKConvND.
        """
        super().__init__()
        from nvsubquadratic.lazy_config import instantiate
        self.ckconv = instantiate(ckconv_cfg)

    def forward(self, x: torch.Tensor, cp_group: torch.distributed.ProcessGroup = None) -> torch.Tensor:
        """Forward pass through CKConv.

        Args:
            x: Input tensor of shape [B, *spatial_dims, C]
            cp_group: Context parallel process group (optional)

        Returns:
            Output tensor of shape [B, *spatial_dims, C]
        """
        return self.ckconv(x, is_bhl_input=False, cp_group=cp_group)


def get_config() -> ExperimentConfig:
    """Get the configuration for the WELL active_matter experiment with CKConv.

    Returns:
        ExperimentConfig: The configuration for the experiment.
    """
    # Start with default config
    config = ExperimentConfig()

    config.debug = False
    config.compile = False

    # Add dataset config
    config.dataset = LazyConfig(WellDataModule)(
        well_base_path=WELL_BASE_PATH,
        well_dataset_name=WELL_DATASET_NAME,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        use_normalization=True,
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        min_dt_stride=1,
        max_dt_stride=1,
        seed=config.seed,
        use_deterministic_worker_init=True,
        prefetch_factor=2,
        spatial_downsample_factor=4,  # Full 256x256 resolution
    )

    # Create norm config once to reuse
    norm_cfg = LazyConfig(torch.nn.RMSNorm)(normalized_shape=NUM_HIDDEN_CHANNELS)

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=PLACEHOLDER,  # Will be set from datamodule
        out_channels=PLACEHOLDER,  # Will be set from datamodule
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
            sequence_mixer_cfg=LazyConfig(SimpleCKConvMixer)(
                ckconv_cfg=LazyConfig(CKConvND)(
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
                        L_cache=16,  # Matches patched spatial dim (64 / PATCH_SIZE)
                        use_bias=True,
                        hidden_omega_0=1.0,
                    ),
                    mask_cfg=LazyConfig(GaussianModulationND)(
                        data_dim=DATA_DIM,
                        num_channels=NUM_HIDDEN_CHANNELS,
                        min_std=0.025,
                        max_std=1.25,
                        init_std_low=0.05,
                        init_std_high=1.0,
                        parametrization="direct",
                    ),
                    grid_type=GRID_TYPE,
                ),
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
    # Note: metadata will be passed from the datamodule at runtime
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
        project="nvsubquadratic-well",
        entity="maxxxzdn",
        job_group="active_matter_ckconv",
    )

    return config
