# TODO: Add license header here

"""EMNIST Spatial Recall 1D - Mamba XS (Extra-Small) - Autoregressive Pretraining.

This config uses autoregressive (next-token prediction) training on the 1D canvas.
The model learns to predict the next element in the sequence given all previous elements.

Key differences from standard regression:
- Uses AutoregressiveWrapper with continuous mode
- Input is shifted: model sees x[:, :-1] and predicts x[:, 1:]
- Loss: MSE on next-element prediction

This can serve as a pretraining objective before fine-tuning on the spatial recall task.

Mamba is designed for 1D sequences:
- Native 1D sequence processing
- Linear complexity O(n) vs O(n²) for attention
- Unidirectional (causal) mode for autoregressive

Model Size: XS (Extra-Small)
- Hidden dim: 128
- Headdim: 32
- Expand: 2
- Params: ~738K (unidirectional)
"""

import torch

import examples.spatial_recall_1d.mixer_defaults as spatial_recall_1d_mixer_defaults
from examples.spatial_recall_1d.base_config import (
    base_emnist_spatial_recall_1d_dataset_config,
)
from experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.autoregressive_wrapper import AutoregressiveWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


# Dataset-specific parameters
BATCH_SIZE = 64
TARGET_SIZE = 16  # 16×16 → 256 element segment
CANVAS_SIZE = 64  # 64×64 → 4096 element canvas
READOUT_VALUE = 0.0

# Network parameters - XS size
# For unidirectional Mamba (bidirectional=False), we need ~738K params to match Attention's ~719K
# hidden_dim must be multiple of 16 for Mamba2 compatibility
INPUT_CHANNELS = 1  # Grayscale
OUTPUT_CHANNELS = 1  # Predict next element (same as input channels)
HIDDEN_DIM = 128
HEADDIM = 32
EXPAND = 2
NUM_BLOCKS = 4

# Training parameters
TRAINING_ITERATIONS = 20_000


def get_config() -> ExperimentConfig:
    """Get the configuration for autoregressive pretraining on 1D canvas with Mamba."""
    config = ExperimentConfig()

    # =========================================================================
    # Network Configuration (1D: data_dim=1)
    # =========================================================================

    # For autoregressive, output_channels = input_channels (predicting next element)
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=HIDDEN_DIM,
        data_dim=1,  # 1D sequence!
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=PLACEHOLDER,  # Will be set below
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            # Condition mixer (not used)
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
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
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        # For autoregressive, target_size = canvas_length (full sequence)
        target_size=None,
    )

    # Mixer: Mamba2 (unidirectional for autoregressive)
    config.net.block_cfg.sequence_mixer_cfg = spatial_recall_1d_mixer_defaults.get_mamba_mixer_cfg(
        headdim=HEADDIM,
        expand=EXPAND,
        bidirectional=False,  # Causal for autoregressive!
    )

    # =========================================================================
    # Training Configuration - Autoregressive
    # =========================================================================

    # Lightning wrapper for autoregressive (continuous mode)
    config.lightning_wrapper_class = LazyConfig(AutoregressiveWrapper)(
        mode="continuous",
        loss_type="mse",
    )

    # Optimizer config
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=1e-4,
        weight_decay=1e-3,
    )

    # Scheduler config
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # Training config
    config.train = TrainConfig(
        batch_size="${dataset.base_datamodule_cfg.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=10.0,
    )

    # Wandb config
    config.wandb = WandbConfig(
        job_group="spatial_recall_1d_emnist_simple_copy_pretrain_xs",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # =========================================================================
    # Dataset Configuration
    # =========================================================================

    config.dataset = base_emnist_spatial_recall_1d_dataset_config(
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        batch_size=BATCH_SIZE,
        num_items=1,
        placement="fixed",
        with_mask=False,
        normalize_input=True,
        readout_value=READOUT_VALUE,
    )

    # =========================================================================
    # Callbacks
    # =========================================================================

    # Note: The standard 1D visualization callback expects regression format
    # For autoregressive, we may need a different visualization
    config.callbacks = []

    return config
