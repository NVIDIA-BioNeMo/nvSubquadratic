# David W. Romero, 2025-09-09

"""Config file for MNIST classification."""

import os
from dataclasses import dataclass

import torch

from nvsubquadratic.examples.mnist_classification.classification_resnet import ClassificationResNet
from nvsubquadratic.examples.mnist_classification.mnist_datamodule import MNISTDataModule
from nvsubquadratic.src.ckconv_nd import CKConvND
from nvsubquadratic.src.hyena_nd import Hyena
from nvsubquadratic.src.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.src.kernels_nd import RandomFourierKernelND
from nvsubquadratic.src.masks_nd import ExponentialModulationND
from nvsubquadratic.src.mlp import MLP
from nvsubquadratic.src.residual_block import ResidualBlock
from nvsubquadratic.src.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.src.utils.lazy_config import LazyConfig


PLACEHOLDER = None

DATA_TYPE = "image"
DATA_DIM = 2

# Model parameters
BATCH_SIZE = 128
NUM_HIDDEN_CHANNELS = 128
NUM_BLOCKS = 4
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "double"

# TRAINING parameters
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS = int(
    TRAINING_ITERATIONS * 0.05
)  # 5% of the training iterations -- initially 5 epochs from 200 epochs
NUM_WORKERS = os.cpu_count() // 4
GRAD_CLIP = 10.0

WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.001


@dataclass
class TrainConfig:
    """Configuration for training parameters."""

    do: bool = True
    precision: str = "32-true"
    iterations: int = -1
    batch_size: int = -1
    grad_clip: float = 0.0
    track_grad_norm: int = -1  # -1 for no tracking
    accumulate_grad_steps: int = 1  # Accumulate gradient over different batches
    distributed: bool = False
    num_nodes: int = -1
    avail_gpus: int = -1


@dataclass
class ExperimentConfig:
    """Configuration template for MNIST classification."""

    device: str = "cuda"
    seed: int = 0
    dataset: LazyConfig = PLACEHOLDER
    net: LazyConfig = PLACEHOLDER
    optimizer: LazyConfig = PLACEHOLDER
    train: TrainConfig = PLACEHOLDER


def get_config():
    """Get the configuration for the MNIST classification experiment.

    Returns:
        ExperimentConfig: The configuration for the MNIST classification experiment.
    """
    # Sratr with default config
    config = ExperimentConfig()

    # Add dataset config
    # Update dataset with LazyConfig directly referencing the MNISTDataModule class
    # and providing all parameters directly (no nested params dataclass)
    config.dataset = LazyConfig(MNISTDataModule)(
        data_dir=".data/mnist",
        data_type=DATA_TYPE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        use_deterministic_worker_init=True,  # Flag to use deterministic worker initialization
        seed=config.seed,  # Pass the seed value instead of a Generator object
    )

    # Add net config
    config.net = LazyConfig(ClassificationResNet)(
        in_channels=PLACEHOLDER,
        out_channels=PLACEHOLDER,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
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
                            data_dim="${net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.data_dim}",
                            out_dim="${net.hidden_dim}",
                            mlp_hidden_dim=32,
                            num_layers=3,
                            embedding_dim=32,
                            omega_0=30.0,
                            L_cache=28,
                            use_bias=True,
                            nonlinear_cfg=LazyConfig(torch.nn.GELU)(),
                            init_method=small_init,
                        ),
                        mask_cfg=LazyConfig(ExponentialModulationND)(
                            data_dim="${net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.data_dim}",
                            num_channels="${net.hidden_dim}",
                            fast_decay_pct=13.81,
                            slow_decay_pct=2.3,
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
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="relu",
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
            ),
            norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
    )

    # Add optimizer config
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Modify the train config - only set what is different from the default
    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
    )

    return config
