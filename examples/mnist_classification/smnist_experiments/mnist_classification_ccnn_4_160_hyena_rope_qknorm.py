# TODO: Add license header here


"""Config file for MNIST classification."""

import os

import torch

from examples.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from examples.lightning_wrappers import ClassificationWrapper
from examples.mnist_classification.mnist_datamodule import MNISTDataModule
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet


PLACEHOLDER = None

DATA_TYPE = "sequence"
DATA_DIM = 1

# Model parameters
BATCH_SIZE = 128
NUM_HIDDEN_CHANNELS = 160
NUM_BLOCKS = 4
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "double"

# TRAINING parameters
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS = int(
    TRAINING_ITERATIONS * 0.05
)  # 5% of the training iterations -- initially 5 epochs from 200 epochs
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count()
GRAD_CLIP = 10.0

WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.001


def get_config() -> ExperimentConfig:
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
                    short_conv_cfg=LazyConfig(torch.nn.Conv1d)(
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

    # Add lightning wrapper config
    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()

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

    # Modify the scheduler config - only set what's different from default
    config.scheduler = SchedulerConfig(
        name="cosine", warmup_iterations=WARMUP_ITERATIONS, total_iterations="${train.iterations}"
    )

    # Add wandb group
    config.wandb = WandbConfig(job_group="smnist_classification")

    return config
