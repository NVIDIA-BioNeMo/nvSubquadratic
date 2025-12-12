# TODO: Add license header here


"""Config file for EMNIST spatial recall regression (2D) with Hyena backbone and learnable spectral patchification.

This config uses:
- SpectralPatchify (CKConvND + SpectralGaussianMaskND) as the input projection, which learns the optimal
  downsampling factor during training using DiffStride-style spectral masking.
- SpectralUnpatchify (bilinear interpolation + CKConv) as the output projection, which reconstructs the
  full resolution output from the downsampled latent representation.
- Hyena as the sequence mixer (instead of Attention).
"""

import os

import torch

from experiments.callbacks.image_grid_val_visualization import ValidationImageGridCallback
from experiments.datamodules.emnist import EMNISTDataModule
from experiments.datamodules.spatial_recall_dataset import SpatialRecallDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.regression_wrapper import RegressionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import SpectralLinearMaskND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import SpectralUnpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


DATA_TYPE = "image"
DATA_DIM = 2

# Spatial recall task parameters
TARGET_SIZE = 16
CANVAS_SIZE = 64

# Spectral patchify parameters
INIT_STRIDE = 8.0  # Initial stride (learnable): 64 / 8 = 8x8 tokens
MIN_STRIDE = 1.0  # Minimum allowed stride
MAX_STRIDE = 16.0  # Maximum allowed stride
CLIP_VALUE = 0.5  # Gaussian mask clip value

# Model parameters
NUM_HIDDEN_CHANNELS = 160
NUM_BLOCKS = 4
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0
GRID_TYPE = "double"
FFT_PADDING = "zero"

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
    """Get the configuration for the EMNIST spatial recall regression experiment with spectral patchification.

    This configuration uses learnable spectral patchification with Hyena:
    - SpectralPatchify (CKConvND + SpectralGaussianMaskND):
      - Learns optimal downsampling factor during training via DiffStride-style spectral masking
      - Initial stride of 8.0 reduces 64x64 canvas to ~8x8 tokens (64 vs 4096 with Linear)
    - SpectralUnpatchify (bilinear interpolation + CKConv):
      - Bilinear upsampling to target resolution
      - CKConv refinement for high-quality reconstruction
    - Hyena as the sequence mixer:
      - Global CKConv with SIREN kernel for long-range dependencies
      - Short 3x3 depthwise convolution for local context
      - QK normalization

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

    # Network config - ResidualNetwork for regression with Hyena backbone
    # Input: [B, canvas_size, canvas_size, input_channels]
    # After SpectralPatchify: [B, ~canvas_size/stride, ~canvas_size/stride, hidden_dim]
    # After SpectralUnpatchify: [B, canvas_size, canvas_size, output_channels]
    # After Readout: [B, target_size, target_size, output_channels]
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=PLACEHOLDER,  # Will be filled from dataset.input_channels
        out_channels=PLACEHOLDER,  # Will be filled from dataset.output_channels
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        # SpectralPatchify as input projection (CKConvND + SpectralGaussianMaskND for learnable stride)
        in_proj_cfg=LazyConfig(CKConvND)(
            data_dim=DATA_DIM,
            hidden_dim="3 * ${net.hidden_dim}",
            kernel_cfg=LazyConfig(SIRENKernelND)(
                data_dim=DATA_DIM,
                out_dim="3 * ${net.hidden_dim}",
                mlp_hidden_dim=32,
                num_layers=3,
                embedding_dim=32,
                omega_0=10.0,
                L_cache="${dataset.canvas_size}",
                use_bias=True,
                hidden_omega_0=1.0,
            ),
            mask_cfg=LazyConfig(torch.nn.Identity)(),  # No spatial mask
            spectral_mask_cfg=LazyConfig(SpectralLinearMaskND)(
                data_dim=DATA_DIM,
                transition_fraction=0.1,
                init_stride_value=1.5,
                min_stride_value=0.95,
                max_stride_value=16.0,
                parametrization="direct",
            ),
            grid_type="single",
            fft_padding="zero",
            use_shortcut=False,
            is_depthwise=False,
        ),
        # SpectralUnpatchify as output projection (bilinear interpolation + CKConv refinement)
        out_proj_cfg=LazyConfig(SpectralUnpatchify)(
            in_features=PLACEHOLDER,
            out_features=PLACEHOLDER,
            data_dim=DATA_DIM,
            output_proj_cfg=LazyConfig(CKConvND)(
                data_dim=DATA_DIM,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    out_dim="${net.hidden_dim}",
                    data_dim=DATA_DIM,
                    mlp_hidden_dim=32,
                    num_layers=3,
                    embedding_dim=32,
                    omega_0=10.0,
                    L_cache="${dataset.canvas_size}",
                    use_bias=True,
                    hidden_omega_0=1.0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="single",
                fft_padding="zero",
                use_shortcut=False,
                is_depthwise=False,
            ),
            interpolation_mode="nearest",
        ),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=DATA_DIM,
                        hidden_dim="${net.hidden_dim}",
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim="${net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.data_dim}",
                            out_dim="${net.hidden_dim}",
                            mlp_hidden_dim=32,
                            num_layers=3,
                            embedding_dim=32,
                            omega_0=10.0,
                            L_cache="${dataset.canvas_size}",
                            use_bias=True,
                            hidden_omega_0=1.0,
                        ),
                        mask_cfg=LazyConfig(torch.nn.Identity)(),  # No mask required.
                        grid_type=GRID_TYPE,
                        fft_padding=FFT_PADDING,
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels="3 * ${net.hidden_dim}",
                        out_channels="3 * ${net.hidden_dim}",
                        kernel_size=3,
                        groups="3 * ${net.hidden_dim}",
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),  # No gate required.
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
                    apply_qk_norm=True,
                    use_rope=False,
                    rope_base=10000.0,
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
