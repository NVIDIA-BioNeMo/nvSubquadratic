# TODO: Add license header here

r"""Minimal MNIST config for gradient equivalence testing.

This config creates a tiny dataset (only 4 training samples = 1 batch) to ensure
CP=1 and CP=2 process the exact same data for gradient comparison testing.

Usage:
    python tests/test_e2e_gradient_with_dataloader.py \\
        --config examples/mnist_classification/experiments/mnist_gradient_test_minimal_dataset.py
"""

import torch

from examples.default_cfg import DistributedConfig, ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from examples.lightning_wrappers import ClassificationWrapper
from examples.mnist_classification.mnist_datamodule import MNISTDataModule
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.distributed_depthwise_conv_nd import DistributedDepthwiseConv2d
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet


# Minimal dataset for gradient testing
class MinimalMNISTDataModule(MNISTDataModule):
    """MNIST DataModule limited to exactly 4 samples for gradient equivalence testing.

    This ensures both CP=1 and CP=2 process the exact same 4 samples.
    """

    def setup(self, stage=None):
        """Setup with only 4 training samples (= 1 batch)."""
        super().setup(stage)

        if stage == "fit" or stage is None:
            from torch.utils.data import Subset

            # Limit to exactly 4 samples (= 1 batch)
            indices = list(range(4))
            self.train_dataset = Subset(self.train_dataset, indices)
            self.val_dataset = Subset(self.val_dataset, indices)


PLACEHOLDER = None

# Data configuration (matches production config structure)
DATA_TYPE = "image"
DATA_DIM = 2

# Model parameters (minimal for fast testing)
BATCH_SIZE = 4  # Small batch for gradient testing
NUM_HIDDEN_CHANNELS = 64  # Smaller than production (160)
NUM_BLOCKS = 1  # Minimal (vs 4 in production)
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0  # No dropout for deterministic gradients
GRID_TYPE = "double"  # Same as production

# Training parameters (minimal for testing)
TRAINING_ITERATIONS = 1  # Single step for gradient comparison
WARMUP_ITERATIONS = 0
NUM_WORKERS = 0  # No workers for determinism
GRAD_CLIP = 10.0

# Optimizer parameters
WEIGHT_DECAY = 0.0  # No weight decay for cleaner gradients
LEARNING_RATE = 0.001


def get_config() -> ExperimentConfig:
    """Get minimal configuration for gradient equivalence testing.

    Returns:
        ExperimentConfig: Minimal configuration matching production structure.
    """
    # Start with default config
    config = ExperimentConfig()

    # Dataset config - minimal subset
    config.dataset = LazyConfig(MinimalMNISTDataModule)(
        data_dir=".data/mnist",
        data_type=DATA_TYPE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,  # Disable for testing
        use_deterministic_worker_init=True,
        seed=config.seed,
        enable_cp=False,  # Override via command line
    )

    # Network config (matches production structure, minimal size)
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
                    # IMPORTANT: Use DistributedDepthwiseConv2d for CP support (not regular Conv2d)
                    short_conv_cfg=LazyConfig(DistributedDepthwiseConv2d)(
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

    # Lightning wrapper config
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

    # Scheduler config (no scheduler for single step testing)
    config.scheduler = SchedulerConfig(
        name=None,  # No scheduler for gradient testing
        warmup_iterations=WARMUP_ITERATIONS,
        total_iterations="${train.iterations}",
    )

    # Wandb config
    config.wandb = WandbConfig(job_group="gradient_test_minimal")

    # Distributed config (will be overridden by command line)
    config.distributed = DistributedConfig(
        enabled=False,
        backend="megatron",
        context_parallel_size=1,
    )

    config.debug = True

    return config


# Create the config instance
config = get_config()
