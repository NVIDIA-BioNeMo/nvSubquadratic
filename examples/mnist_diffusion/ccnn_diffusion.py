# TODO: Add license header here

"""Config file for MNIST diffusion using the shared ResNet backbone."""

import os

import torch

from experiments.datamodules.mnist import MNISTDataModule
from experiments.default_cfg import (
    DiffusionConfig,
    DiffusionEMAConfig,
    DiffusionExperimentConfig,
    DiffusionSamplingConfig,
    DiffusionScheduleConfig,
    SchedulerConfig,
    TrainConfig,
    WandbConfig,
)
from experiments.lightning_wrappers import DiffusionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.condition_mixer import QKVConditionMixer
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import RandomFourierKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


PLACEHOLDER = None

DATA_TYPE = "image"
DATA_DIM = 2

# Model parameters
BATCH_SIZE = 128
HIDDEN_DIM = 160
NUM_BLOCKS = 4
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "double"

# Training parameters
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
GRAD_CLIP = 10.0
WEIGHT_DECAY = 0.01
LEARNING_RATE = 1e-3

# Diffusion parameters
TIME_EMBED_DIM = HIDDEN_DIM
MAX_PERIOD = 10_000.0
NUM_INFERENCE_STEPS = 50
NUM_SAMPLES = 4
LOG_SAMPLES = True
EMA_ENABLED = True
EMA_DECAY = 0.999
EMA_WARMUP_STEPS = 1_000
EMA_UPDATE_EVERY = 1


def get_config() -> DiffusionExperimentConfig:
    """Return the MNIST diffusion configuration."""

    config = DiffusionExperimentConfig()

    cpu_count = os.cpu_count() or 4
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        num_workers = max(1, cpu_count // torch.cuda.device_count())
    else:
        num_workers = max(1, cpu_count // 2)

    config.dataset = LazyConfig(MNISTDataModule)(
        data_dir=".data/mnist",
        data_type=DATA_TYPE,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
    )

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=PLACEHOLDER,
        out_channels=PLACEHOLDER,
        num_blocks=NUM_BLOCKS,
        hidden_dim=HIDDEN_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=PLACEHOLDER),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=PLACEHOLDER),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=DATA_DIM,
                        hidden_dim="${net.hidden_dim}",
                        kernel_cfg=LazyConfig(RandomFourierKernelND)(
                            data_dim=DATA_DIM,
                            out_dim="${net.hidden_dim}",
                            mlp_hidden_dim=64,
                            num_layers=2,
                            embedding_dim=32,
                            omega_0=50.0,
                            L_cache=16,
                            use_bias=True,
                            nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                        ),
                        mask_cfg=LazyConfig(GaussianModulationND)(
                            data_dim=DATA_DIM,
                            num_channels="${net.hidden_dim}",
                            min_std=0.02,
                            max_std=1.0,
                            init_std_low=0.05,
                            init_std_high=0.8,
                            parametrization="direct",
                        ),
                        grid_type=GRID_TYPE,
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
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.GroupNorm)(
                        num_groups=1,
                        num_channels="${net.hidden_dim}",
                    ),
                    apply_qk_norm=True,
                    use_rope=True,
                    rope_base=10000.0,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            condition_mixer_cfg=LazyConfig(QKVConditionMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(torch.nn.MultiheadAttention)(
                    embed_dim="${net.hidden_dim}",
                    num_heads=1,
                    batch_first=True,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            condition_mixer_norm_cfg="${net.norm_cfg}",
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=2.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
        condition_in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=PLACEHOLDER),
    )

    config.lightning_wrapper_class = LazyConfig(DiffusionWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
    )

    # Compose diffusion config with explicit schedule, sampling, and EMA parameters.
    config.diffusion = DiffusionConfig(
        time_embed_dim=TIME_EMBED_DIM,
        max_period=MAX_PERIOD,
        num_inference_steps=NUM_INFERENCE_STEPS,
        num_samples=NUM_SAMPLES,
        log_samples=LOG_SAMPLES,
        ema_enabled=EMA_ENABLED,
        ema_decay=EMA_DECAY,
        ema_update_every=EMA_UPDATE_EVERY,
        ema_warmup_steps=EMA_WARMUP_STEPS,
    )

    config.wandb = WandbConfig(job_group="mnist-diffusion")

    return config
