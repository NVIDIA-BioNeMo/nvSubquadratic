# TODO: Add license header here


"""Config file for EMNIST spatial recall regression (2D) with Transformer (Attention) backbone and dual-path patchification.

This config uses:
- DualPathPatchify (spectral + spatial paths) as the input projection, which combines:
  - Spectral path: Anti-aliased low-pass filtering with learnable stride
  - Spatial path: Standard strided convolution preserving high-frequency (aliased) content
- DualPathUnpatchify (spectral + spatial upsampling) as the output projection, which combines:
  - Spectral path: Bilinear interpolation upsampling
  - Spatial path: PixelShuffle-based learned upsampling
- Attention as the sequence mixer (instead of Hyena).

The dual-path approach enables perfect reconstruction by combining clean low-frequencies
from the spectral path with aliased high-frequencies from the spatial path.
"""

import os

import torch

from experiments.callbacks.image_grid_val_visualization import ValidationImageGridCallback
from experiments.datamodules.emnist import EMNISTDataModule
from experiments.datamodules.spatial_recall_dataset import SpatialRecallDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.regression_wrapper import RegressionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.masks_nd import SpectralGaussianMaskND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import DualPathPatchify, DualPathUnpatchify, SpectralPatchify, SpectralUnpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


DATA_TYPE = "image"
DATA_DIM = 2

# Spatial recall task parameters
TARGET_SIZE = 16
CANVAS_SIZE = 64

# Dual-path patchify hyperparameters
INIT_STRIDE = 4
MAX_STRIDE = 16

# Model parameters
NUM_HIDDEN_CHANNELS = 160
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
    # After DualPathPatchify: [B, ~canvas_size/stride, ~canvas_size/stride, hidden_dim]
    # After DualPathUnpatchify: [B, canvas_size, canvas_size, output_channels]
    # After Readout: [B, target_size, target_size, output_channels]
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=PLACEHOLDER,  # Will be filled from dataset.input_channels
        out_channels=PLACEHOLDER,  # Will be filled from dataset.output_channels
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        # DualPathPatchify as input projection (spectral + spatial paths)
        # Note: in_features and out_features are passed by ResidualNetwork during instantiation
        in_proj_cfg=LazyConfig(DualPathPatchify)(
            data_dim=DATA_DIM,
            max_stride=MAX_STRIDE,
            spectral_patchify_cfg=LazyConfig(SpectralPatchify)(
                in_features=3,  # RGB input
                out_features=NUM_HIDDEN_CHANNELS,
                data_dim=DATA_DIM,
                spectral_mask_cfg=LazyConfig(SpectralGaussianMaskND)(
                    data_dim=DATA_DIM,
                    clip_value=0.5,
                    init_stride_value=float(INIT_STRIDE),
                    min_stride_value=1.0,
                    max_stride_value=float(MAX_STRIDE),
                    parametrization="direct",
                ),
                conv_cfg=LazyConfig(torch.nn.Conv2d)(
                    in_channels=3,  # RGB input
                    out_channels=NUM_HIDDEN_CHANNELS,
                    kernel_size=MAX_STRIDE,
                    padding="same",
                ),
            ),
            freeze_spectral_mask=False,  # Allow stride to be learned
        ),
        # DualPathUnpatchify as output projection (spectral + spatial upsampling)
        # Note: in_features and out_features are passed by ResidualNetwork during instantiation
        out_proj_cfg=LazyConfig(DualPathUnpatchify)(
            data_dim=DATA_DIM,
            spectral_unpatchify_cfg=LazyConfig(SpectralUnpatchify)(
                in_features=NUM_HIDDEN_CHANNELS,
                out_features=1,  # Single channel output for this task
                data_dim=DATA_DIM,
                output_proj_cfg=LazyConfig(torch.nn.Conv2d)(
                    in_channels=NUM_HIDDEN_CHANNELS,
                    out_channels=1,
                    kernel_size=MAX_STRIDE,
                    padding="same",
                ),
                interpolation_mode="bilinear",
            ),
            max_stride=MAX_STRIDE,
            interpolation_mode="bilinear",
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
        job_group="spatial_recall_2d_emnist_regression_colored",
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
