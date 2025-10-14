# TODO: Add license header here

"""Config file for MNIST classification with Context Parallelism.

This config demonstrates how to enable Context Parallelism for distributed training.

Usage:
    # Non-distributed (standard training):
    python examples/run.py --config examples/mnist_classification/experiments/mnist_classification_ccnn_cp_test.py

    # Distributed with CP (2 GPUs, CP size=2):
    torchrun --nproc_per_node=2 examples/run.py \
        --config examples/mnist_classification/experiments/mnist_classification_ccnn_cp_test.py \
        distributed.enabled=True \
        distributed.context_parallel_size=2 \
        dataset.enable_cp=True

    # Distributed with CP (4 GPUs, 2 DP replicas x 2 CP devices):
    torchrun --nproc_per_node=4 examples/run.py \
        --config examples/mnist_classification/experiments/mnist_classification_ccnn_cp_test.py \
        distributed.enabled=True \
        distributed.context_parallel_size=2 \
        dataset.enable_cp=True
"""

import os

import torch

from examples.default_cfg import DistributedConfig, ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from examples.lightning_wrappers import ClassificationWrapper
from examples.mnist_classification.mnist_datamodule import MNISTDataModule
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.distributed_depthwise_conv_nd import DistributedDepthwiseConv1d
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet


PLACEHOLDER = None

# Data configuration
DATA_TYPE = "sequence"  # Use sequence for CP (splits along sequence dimension)
DATA_DIM = 1  # 1D sequence

# Model parameters (smaller for testing)
BATCH_SIZE = 64
NUM_HIDDEN_CHANNELS = 128
NUM_BLOCKS = 2
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "single"

# Training parameters (shorter for testing)
TRAINING_ITERATIONS = 1000  # Short test run
WARMUP_ITERATIONS = 50
NUM_WORKERS = max(1, os.cpu_count() // max(1, torch.cuda.device_count()))
GRAD_CLIP = 10.0

# Optimizer parameters
WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.001


def get_config() -> ExperimentConfig:
    """Get the configuration for MNIST classification with Context Parallelism.

    Returns:
        ExperimentConfig: The configuration for the experiment.
    """
    # Start with default config
    config = ExperimentConfig()

    # Dataset config with CP support
    config.dataset = LazyConfig(MNISTDataModule)(
        data_dir=".data/mnist",
        data_type=DATA_TYPE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        use_deterministic_worker_init=True,
        seed=config.seed,
        enable_cp=False,  # Will be overridden by command line when using distributed
    )

    # Network config with Hyena using DistributedDepthwiseConv (CP-compatible)
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
                    ),
                    # IMPORTANT: Use DistributedDepthwiseConv1d for CP support
                    short_conv_cfg=LazyConfig(DistributedDepthwiseConv1d)(
                        hidden_dim="3 * ${net.hidden_dim}",
                        kernel_size=3,
                        num_groups="3 * ${net.hidden_dim}",
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.GroupNorm)(num_groups=1, num_channels="${net.hidden_dim}"),
                    apply_qk_norm=True,
                    use_rope=True,
                    rope_base=10000.0,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
    )

    # Lightning wrapper
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()

    # Optimizer config
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Training config
    config.train = TrainConfig(
        batch_size=BATCH_SIZE,
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
    )

    # Scheduler config
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations=WARMUP_ITERATIONS,
        total_iterations="${train.iterations}",
    )

    # Wandb config
    config.wandb = WandbConfig(job_group="mnist_classification_cp_test")

    # Distributed config - can be overridden via command line
    config.distributed = DistributedConfig(
        enabled=False,  # Set to True via command line or change here
        backend="megatron",
        context_parallel_size=1,  # Override via command line (e.g., 2, 4)
        use_distributed_checkpoint=False,
        checkpoint_dir="./checkpoints_cp",
    )

    # Debug mode for quick testing
    config.debug = True

    return config


# Create the config instance
config = get_config()
