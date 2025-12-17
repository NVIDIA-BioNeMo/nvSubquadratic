# TODO: Add license header here


"""Config file for EMNIST spatial recall regression (2D) with Mamba2 backbone."""

import os

import torch

# Import Mamba2 from mamba-ssm library
from mamba_ssm import Mamba2

from experiments.callbacks.image_grid_val_visualization import ValidationImageGridCallback
from experiments.datamodules.emnist import EMNISTDataModule
from experiments.datamodules.spatial_recall_dataset import SpatialRecallDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.regression_wrapper import RegressionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.mamba_nd import Mamba as MambaNDMixer
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


DATA_TYPE = "image"
DATA_DIM = 2

# Spatial recall task parameters
TARGET_SIZE = 16
CANVAS_SIZE = 64

# Model parameters
NUM_HIDDEN_CHANNELS = 160
NUM_BLOCKS = 4
MAMBA_HEADDIM = 64  # Mamba2 head dimension
MAMBA_EXPAND = 2  # Expansion factor for inner dimension
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0

# Training parameters
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
GRAD_CLIP = 10.0
PRECISION = "bf16-mixed"
BATCH_SIZE = 64

WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-4


def get_config() -> ExperimentConfig:
    """Get the configuration for the EMNIST spatial recall regression experiment.

    Returns:
        ExperimentConfig: The configuration for the experiment.
    """
    config = ExperimentConfig()

    # Base EMNIST datamodule config
    base_datamodule_cfg = LazyConfig(EMNISTDataModule)(
        data_dir=".data/emnist",
        batch_size=BATCH_SIZE,
        data_type=DATA_TYPE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        permuted=False,
        seed=config.seed,
        normalize_input=True,
        split="byclass",
    )

    # Spatial recall datamodule wrapping the base EMNIST datamodule
    # Colored frames mode: 3-channel RGB input with colored bounding boxes
    # num_items=4 means 1 target + 3 distractors
    config.dataset = LazyConfig(SpatialRecallDataModule)(
        base_datamodule_cfg=base_datamodule_cfg,
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        data_type=DATA_TYPE,
        placement="random",  # Items placed randomly for colored frames
        with_mask=False,
        use_colored_frames=True,  # 3-channel RGB with colored bounding boxes
        num_items=4,  # 1 target + 3 distractors
    )

    # Network config - ResidualNetwork for regression with Mamba2 backbone
    # Input: [B, canvas_size, canvas_size, input_channels]
    # Output: [B, target_size, target_size, 1] (the recalled image)
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=PLACEHOLDER,  # Will be filled from dataset.input_channels
        out_channels=PLACEHOLDER,  # Will be filled from dataset.output_channels
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=PLACEHOLDER),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=PLACEHOLDER),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            # Mamba as the sequence mixer (not wrapped in QKVSequenceMixer)
            # The Mamba wrapper handles flattening spatial dims and bidirectionality
            sequence_mixer_cfg=LazyConfig(MambaNDMixer)(
                mamba_layer_cfg=LazyConfig(Mamba2)(
                    d_model="${net.hidden_dim}",
                    headdim=MAMBA_HEADDIM,
                    expand=MAMBA_EXPAND,
                ),
                bidirectional=True,  # Use bidirectional Mamba for better spatial understanding
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            # Condition mixer (not used for spatial recall)
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),  # No condition mixer.
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),  # No condition mixer.
            # MLP
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            # Dropout
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
        target_size=TARGET_SIZE,  # For readout region extraction
    )

    # Lightning wrapper for regression
    config.lightning_wrapper_class = LazyConfig(RegressionWrapper)(metric="MSE")

    # Optimizer config
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler config
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # Training config
    config.train = TrainConfig(
        batch_size="${dataset.base_datamodule_cfg.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
    )

    # Wandb config
    config.wandb = WandbConfig(
        job_group="spatial_recall_2d_emnist_regression_colored",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    config.callbacks = [
        ValidationImageGridCallback(
            num_samples=8,
            every_n_epochs=None,
            every_n_train_steps=2000,
        ),
    ]

    return config
