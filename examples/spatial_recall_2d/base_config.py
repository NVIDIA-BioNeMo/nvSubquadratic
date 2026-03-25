# TODO: Add license header here

"""Base configuration for spatial recall 2D experiments.

This module provides the base configuration that is common across all spatial recall
experiments. Individual experiment configs only need to specify:
1. The dataset configuration
2. The mixer configuration (from mixer_defaults.py)

Usage:
    from examples.spatial_recall_2d.base_config import get_base_config
    from examples.spatial_recall_2d.mixer_defaults import get_hyena_mixer_cfg

    def get_config():
        config = get_base_config(
            in_channels=1,
            out_channels=1,
            mixer_cfg=get_hyena_mixer_cfg(),
        )
        config.dataset = ...  # Your dataset config
        return config
"""

import os
from typing import Literal

import torch

from experiments.callbacks.image_grid_val_visualization import ValidationImageGridCallback
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.regression_wrapper import RegressionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init


def get_num_workers() -> int:
    """Get the number of workers for data loading."""
    if torch.cuda.is_available():
        return os.cpu_count() // torch.cuda.device_count()
    return os.cpu_count()


# =============================================================================
# Base Experiment Configuration
# =============================================================================
def base_experiment_config(
    # Required: These define your experiment
    in_channels: int,
    out_channels: int,
    # Model architecture
    num_blocks: int = 4,
    hidden_dim: int = 160,
    data_dim: int = 2,
    dropout_in_rate: float = 0.0,
    dropout_rate: float = 0.0,
    # Training
    training_iterations: int = 100_000,
    warmup_iterations_percentage: float = 0.05,
    grad_clip: float = 10.0,
    # Optimizer
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    # Wandb
    wandb_job_group: str = "spatial_recall_2d",
    wandb_entity: str = "implicit-long-convs",
    wandb_project: str = "nvsubquadratic",
    # Callbacks
    image_grid_every_n_steps: int = 2000,
    image_grid_num_samples: int = 8,
) -> ExperimentConfig:
    """Get the complete experiment configuration.

    Only requires:
    - in_channels: Number of input channels (dataset-dependent)
    - out_channels: Number of output channels (dataset-dependent)

    IMPORTANT!
    After calling this, you must set:
    - config.dataset: Use base_emnist_spatial_recall_2d_dataset_config()
    - config.net.block_cfg.sequence_mixer_cfg: Use mixer from mixer_defaults.py

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        num_blocks: Number of residual blocks.
        hidden_dim: Hidden dimension size.
        data_dim: Spatial dimensionality (1 or 2).
        dropout_in_rate: Input dropout rate.
        dropout_rate: Block dropout rate.
        training_iterations: Total training iterations.
        warmup_iterations_percentage: Warmup as fraction of total iterations.
        grad_clip: Gradient clipping value.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        wandb_job_group: Wandb job group name.
        wandb_entity: Wandb entity.
        wandb_project: Wandb project name.
        image_grid_every_n_steps: Log image grid every N steps.
        image_grid_num_samples: Number of samples in image grid.

    Returns:
        ExperimentConfig with everything set except config.dataset.
    """
    config = ExperimentConfig()

    # =========================================================================
    # Network Configuration
    # =========================================================================

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        data_dim=data_dim,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=PLACEHOLDER,  # Must be set after calling this function
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            # Condition mixer (not used for spatial recall)
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
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=dropout_rate),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=dropout_in_rate),
        target_size="${dataset.target_size}",
    )

    # =========================================================================
    # Training Configuration
    # =========================================================================

    # Lightning wrapper for regression
    config.lightning_wrapper_class = LazyConfig(RegressionWrapper)(metric="MSE")

    # Optimizer config
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Scheduler config
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=warmup_iterations_percentage,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # Training config
    config.train = TrainConfig(
        batch_size="${dataset.base_datamodule_cfg.batch_size}",
        iterations=training_iterations,
        grad_clip=grad_clip,
    )

    # Wandb config
    config.wandb = WandbConfig(
        job_group=wandb_job_group,
        entity=wandb_entity,
        project=wandb_project,
    )

    # Callbacks - use LazyConfig so interpolation works for show_mask_separately
    config.callbacks = [
        LazyConfig(ValidationImageGridCallback)(
            num_samples=image_grid_num_samples,
            every_n_epochs=None,
            every_n_train_steps=image_grid_every_n_steps,
            show_mask_separately="${dataset.with_mask}",
        ),
    ]

    return config


# =============================================================================
# Dataset Configuration Helpers
# =============================================================================


def base_emnist_spatial_recall_2d_dataset_config(
    # Required: spatial recall task parameters
    target_size: int,
    canvas_size: int,
    batch_size: int,
    # Recall Specific
    use_colored_frames: bool,
    num_items: int,
    placement: Literal["fixed", "random"],
    with_mask: bool,
    normalize_input: bool,
    # Optional overrides
    readout_value: float = 0.0,
    colored_label: bool = False,
    data_dir: str = ".data/emnist",
    split: str = "byclass",
    pin_memory: bool = True,
) -> LazyConfig:
    """Get base EMNIST spatial recall 2D dataset configuration.

    Args:
        target_size: Size of the target image to recall.
        canvas_size: Size of the full canvas.
        batch_size: Batch size for training.
        use_colored_frames: If True, use RGB with colored bounding boxes (3 channels).
            If False, use grayscale (1 channel).
        num_items: Number of items on canvas (1 = just target, >1 = target + distractors).
        placement: "fixed" for deterministic placement, "random" for random placement.
        with_mask: If True, add a binary mask channel indicating target location.
        normalize_input: Whether to normalize input to [-1, 1].
        readout_value: Value to fill the readout region with (default 0.0). Use e.g. -1.0 to
            explicitly mark the readout region so the model knows where to output.
            Note: When use_colored_frames=True, the colored border is preserved.
        colored_label: If True and use_colored_frames=True, the label will be RGB (3 channels)
            with the digit colored using the same color as its frame. This creates a
            "color conditioning" task where the model must output the digit in the correct color.
        data_dir: Directory for EMNIST data.
        split: EMNIST split to use.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        LazyConfig for SpatialRecallDataModule.
    """
    from experiments.datamodules.emnist import EMNISTDataModule
    from experiments.datamodules.spatial_recall_dataset import SpatialRecallDataModule

    base_datamodule_cfg = LazyConfig(EMNISTDataModule)(
        data_dir=data_dir,
        batch_size=batch_size,
        data_type="image",
        num_workers=get_num_workers(),
        pin_memory=pin_memory,
        permuted=False,
        seed="${seed}",
        normalize_input=normalize_input,
        split=split,
    )

    return LazyConfig(SpatialRecallDataModule)(
        base_datamodule_cfg=base_datamodule_cfg,
        target_size=target_size,
        canvas_size=canvas_size,
        data_type="image",
        placement=placement,
        with_mask=with_mask,
        use_colored_frames=use_colored_frames,
        num_items=num_items,
        readout_value=readout_value,
        colored_label=colored_label,
    )
