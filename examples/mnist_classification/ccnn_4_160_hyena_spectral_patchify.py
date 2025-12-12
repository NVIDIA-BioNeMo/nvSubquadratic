# TODO: Add license header here


"""Config file for MNIST classification with learnable spectral patchification.

This config uses SpectralPatchify as the input projection, which learns the
optimal downsampling factor during training using DiffStride-style spectral masking.
"""

import os

import torch

from experiments.datamodules.mnist import MNISTDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import RandomFourierKernelND, SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND, SpectralGaussianMaskND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet


PLACEHOLDER = None

DATA_TYPE = "image"
DATA_DIM = 2

# Dataset
BATCH_SIZE = 128
MAX_WORKERS = 16
PRECISION = "bf16-mixed"
NUM_WORKERS = min(MAX_WORKERS, os.cpu_count() - 1 or MAX_WORKERS)

# Model parameters
NUM_HIDDEN_CHANNELS = 160
NUM_BLOCKS = 4
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "double"
FFT_PADDING = "zero"

# Spectral patchify parameters
INIT_STRIDE = 4.0  # Initial stride (learnable)
MIN_STRIDE = 1.0  # Minimum allowed stride
MAX_STRIDE = None  # No upper bound
CLIP_VALUE = 0.5  # Gaussian mask clip value

# TRAINING parameters
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
GRAD_CLIP = 10.0

WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.001


def get_config() -> ExperimentConfig:
    """Get the configuration for the MNIST classification experiment with spectral patchification.

    This configuration uses SpectralPatchify as the input projection:
    - Learns optimal downsampling factor during training
    - Uses CKConvND with SpectralGaussianMaskND for differentiable striding
    - Initial stride of 4.0 reduces 28x28 MNIST to ~7x7 tokens

    Returns:
        ExperimentConfig: The configuration for the MNIST classification experiment.
    """
    config = ExperimentConfig()

    # Dataset config
    config.dataset = LazyConfig(MNISTDataModule)(
        data_dir=".data/mnist",
        data_type=DATA_TYPE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        use_deterministic_worker_init=True,
        seed=config.seed,
        task="classification",
    )

    # Network config with SpectralPatchify as input projection
    config.net = LazyConfig(ClassificationResNet)(
        in_channels=PLACEHOLDER,
        out_channels=PLACEHOLDER,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        # SpectralPatchify as input projection (learnable stride)
        in_proj_cfg=LazyConfig(CKConvND)(
            data_dim=DATA_DIM,
            hidden_dim="${net.hidden_dim}",
            kernel_cfg=LazyConfig(RandomFourierKernelND)(
                out_dim="${net.hidden_dim}",
                data_dim=DATA_DIM,
                mlp_hidden_dim=64,
                num_layers=3,
                embedding_dim=32,
                omega_0=1.0,
                L_cache=32,
                use_bias=True,
                nonlinear_cfg=LazyConfig(torch.nn.GELU)(),
            ),
            mask_cfg=LazyConfig(torch.nn.Identity)(),  # No spatial mask
            spectral_mask_cfg=LazyConfig(SpectralGaussianMaskND)(
                data_dim=DATA_DIM,
                clip_value=CLIP_VALUE,
                init_stride_value=INIT_STRIDE,
                min_stride_value=MIN_STRIDE,
                max_stride_value=MAX_STRIDE,
                parametrization="direct",
            ),
            grid_type="single",
            fft_padding="zero",
            use_shortcut=False,
            is_depthwise=False,
        ),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=PLACEHOLDER),
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
                            omega_0=100.0,
                            L_cache=32,
                            use_bias=True,
                            hidden_omega_0=1.0,
                        ),
                        mask_cfg=LazyConfig(GaussianModulationND)(
                            data_dim="${net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.data_dim}",
                            num_channels="${net.hidden_dim}",
                            min_std=0.025,
                            max_std=1.25,
                            init_std_low=0.05,
                            init_std_high=1.0,
                            parametrization="direct",
                        ),
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
                    gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.GroupNorm)(num_groups=1, num_channels="${net.hidden_dim}"),
                    apply_qk_norm=True,
                    use_rope=False,
                    rope_base=10000.0,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            # Condition mixer
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            # MLP
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            # Dropout
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
    )

    # Lightning wrapper config
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()

    # Optimizer config
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Train config
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
    )

    # Scheduler config
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
    )

    # Wandb config
    config.wandb = WandbConfig(
        job_group="mnist_classification",
        entity="implicit-long-convs",
        project="nvsubquadratic",
    )

    return config
