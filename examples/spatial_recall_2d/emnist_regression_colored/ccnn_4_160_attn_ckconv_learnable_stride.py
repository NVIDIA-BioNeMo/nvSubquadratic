# TODO: Add license header here


"""Config file for EMNIST spatial recall regression (2D) with CKConv learnable stride patchification.

This config uses:
- CKConvPatchify: SIREN-based continuous kernel with learnable stride
- CKConvUnpatchify: Matching unpatchify with shared kernel
- Attention as the sequence mixer

Key features:
- Learnable stride via differentiable blending between adjacent integer strides
- Shared SIREN kernel between patchify and unpatchify (24x more parameter efficient)
- Can achieve zero reconstruction loss at any integer stride
- Stride logged to wandb for tracking

This is the recommended patchification approach for learnable stride + perfect reconstruction.
"""

import os

import torch

from experiments.callbacks.image_grid_val_visualization import ValidationImageGridCallback
from experiments.datamodules.emnist import EMNISTDataModule
from experiments.datamodules.spatial_recall_dataset import SpatialRecallDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.regression_wrapper import RegressionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.ckconv_patchify import CKConvPatchify, CKConvUnpatchify
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


# Dataset parameters
INPUT_CHANNELS = 3  # RGB with colored frames
OUTPUT_CHANNELS = 1  # Grayscale target
DATA_TYPE = "image"
DATA_DIM = 2

# Spatial recall task parameters
TARGET_SIZE = 16
CANVAS_SIZE = 64

# CKConv patchify hyperparameters (matching baseline patchify config)
KERNEL_SIZE = 16  # Same as PATCH_SIZE in baseline (overlapping patches if > stride)
INIT_STRIDE = 4  # Same as STRIDE in baseline
MAX_STRIDE = 16
FREEZE_STRIDE = False  # Set to True to keep stride fixed during training
KERNEL_HIDDEN_DIM = 64
KERNEL_NUM_LAYERS = 3

# Model parameters
NUM_HIDDEN_CHANNELS = 256
NUM_BLOCKS = 4
NUM_HEADS = 8
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0

# Training parameters
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
GRAD_CLIP = 10.0
PRECISION = "bf16-mixed"
BATCH_SIZE = 64

# Optimizer parameters
WEIGHT_DECAY = 1e-3
LEARNING_RATE = 1e-4


def get_config():
    """Build and return the full experiment configuration."""
    config = ExperimentConfig()

    # ============================================================
    # Dataset Configuration
    # ============================================================
    # Use EMNIST as the base dataset for spatial recall task
    config.dataset = LazyConfig(SpatialRecallDataModule)(
        # Base datamodule for the dataset
        base_datamodule_cfg=LazyConfig(EMNISTDataModule)(
            data_dir=".data/emnist",
            batch_size=BATCH_SIZE,
            data_type=DATA_TYPE,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available() and config.device == "cuda",
            permuted=False,
            seed=config.seed,
            normalize_input=True,
            split="byclass",
        ),
        canvas_size=CANVAS_SIZE,
        target_size=TARGET_SIZE,
        data_type=DATA_TYPE,
        placement="random",  # Items placed randomly for colored frames
        with_mask=False,
        use_colored_frames=True,
        num_items=4,  # 1 target + 3 distractors
    )

    # ============================================================
    # Lightning Wrapper Configuration
    # ============================================================
    # Lightning wrapper for regression
    config.lightning_wrapper_class = LazyConfig(RegressionWrapper)(metric="MSE")

    # ============================================================
    # Network Configuration
    # ============================================================
    # Input: [B, canvas_size, canvas_size, input_channels]
    # After CKConvPatchify: [B, ~canvas_size/stride, ~canvas_size/stride, hidden_dim]
    # After CKConvUnpatchify: [B, canvas_size, canvas_size, output_channels]
    # After Readout: [B, target_size, target_size, output_channels]
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        # CKConvPatchify as input projection (SIREN-based continuous kernel with learnable stride)
        in_proj_cfg=LazyConfig(CKConvPatchify)(
            in_features="${net.in_channels}",
            out_features="${net.hidden_dim}",
            data_dim="${net.data_dim}",
            init_stride=INIT_STRIDE,
            max_stride=MAX_STRIDE,
            kernel_size=KERNEL_SIZE,  # Same as PATCH_SIZE in baseline (overlapping patches)
            freeze_stride=FREEZE_STRIDE,
            kernel_hidden_dim=KERNEL_HIDDEN_DIM,
            kernel_num_layers=KERNEL_NUM_LAYERS,
        ),
        # CKConvUnpatchify as output projection
        out_proj_cfg=LazyConfig(CKConvUnpatchify)(
            in_features="${net.hidden_dim}",
            out_features="${net.out_channels}",
            data_dim="${net.data_dim}",
            init_stride="${net.in_proj_cfg.init_stride}",
            max_stride="${net.in_proj_cfg.max_stride}",
            kernel_size="${net.in_proj_cfg.kernel_size}",  # Same as PATCH_SIZE in baseline (overlapping patches)
            freeze_stride="${net.in_proj_cfg.freeze_stride}",
            kernel_hidden_dim="${net.in_proj_cfg.kernel_hidden_dim}",
            kernel_num_layers="${net.in_proj_cfg.kernel_num_layers}",
            target_size="${dataset.canvas_size}",
        ),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Attention)(
                    hidden_dim="${net.hidden_dim}",
                    num_heads=NUM_HEADS,
                    apply_qk_norm=True,
                    use_rope=True,
                    is_causal=False,
                    rope_base=10000.0,
                    attn_dropout=DROPOUT_RATE,
                ),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
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
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
        target_size="${dataset.target_size}",  # For readout region extraction
    )

    # ============================================================
    # Optimizer Configuration
    # ============================================================
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ============================================================
    # Scheduler Configuration
    # ============================================================
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # ============================================================
    # Training Configuration
    # ============================================================
    config.train = TrainConfig(
        batch_size="${dataset.base_datamodule_cfg.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
    )

    # ============================================================
    # Wandb Configuration
    # ============================================================
    config.wandb = WandbConfig(
        job_group="spatial_recall_2d_emnist_regression_colored_patchify_ablations",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    # ============================================================
    # Callbacks Configuration
    # ============================================================
    config.callbacks = [
        ValidationImageGridCallback(
            num_samples=8,
            every_n_epochs=None,
            every_n_train_steps=2000,
        ),
    ]

    return config
