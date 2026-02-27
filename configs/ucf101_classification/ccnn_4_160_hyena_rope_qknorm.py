# TODO: Add license header here


"""Config file for UCF101 classification."""

import os

import torch

from experiments.datamodules.ucf101 import UCF101DataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubq_paper.lazy_config import PLACEHOLDER, LazyConfig
from nvsubq_paper.modules.ckconv_nd import CKConvND
from nvsubq_paper.modules.hyena_nd import Hyena
from nvsubq_paper.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubq_paper.modules.kernels_nd import SIRENKernelND
from nvsubq_paper.modules.masks_nd import GaussianModulationND
from nvsubq_paper.modules.mlp import MLP
from nvsubq_paper.modules.residual_block import ResidualBlock
from nvsubq_paper.modules.sequence_mixer import QKVSequenceMixer
from nvsubq_paper.networks.classification_resnet import ClassificationResNet


# Dataset parameters
INPUT_CHANNELS = 3  # RGB video frames
OUTPUT_CHANNELS = 101  # UCF101 action classes
DATA_TYPE = "video"
DATA_DIM = 3

# Training parameters
BATCH_SIZE = 1
PRECISION = "bf16-mixed"  # Tested options: "32-true", "bf16-mixed"

# Model parameters
NUM_HIDDEN_CHANNELS = 156  # Must be divisible by 6
NUM_BLOCKS = 4
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1
GRID_TYPE = "single"
FFT_PADDING = "zero"

# TRAINING parameters
TRAINING_ITERATIONS = 100_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count()
GRAD_CLIP = 10.0

WEIGHT_DECAY = 0.01
LEARNING_RATE = 0.001

# Dataset parameters
FRAME_SIZE = (128, 128)
FRAMES_PER_CLIP = 16
STEP_BETWEEN_CLIPS = 1
VAL_SPLIT_FRACTION = 0.1


def get_config() -> ExperimentConfig:
    """Get the configuration for the UCF101 classification experiment.

    Returns:
        ExperimentConfig: The configuration for the UCF101 classification experiment.
    """
    # Sratr with default config
    config = ExperimentConfig()

    # Add dataset config
    # Update dataset with LazyConfig directly referencing the UCF101DataModule class
    # and providing all parameters directly (no nested params dataclass)
    config.dataset = LazyConfig(UCF101DataModule)(
        data_dir=".data/ucf101",
        split_fold=1,
        data_type=DATA_TYPE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        use_deterministic_worker_init=True,  # Flag to use deterministic worker initialization
        seed=config.seed,  # Pass the seed value instead of a Generator object
        frames_per_clip=FRAMES_PER_CLIP,
        step_between_clips=STEP_BETWEEN_CLIPS,
        frame_size=FRAME_SIZE,
        val_split_fraction=VAL_SPLIT_FRACTION,
    )

    # Add net config
    config.net = LazyConfig(ClassificationResNet)(
        in_channels=INPUT_CHANNELS,
        out_channels=OUTPUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim="${net.data_dim}",
                        hidden_dim="${net.hidden_dim}",
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim="${net.data_dim}",
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
                            data_dim="${net.data_dim}",
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
                    short_conv_cfg=LazyConfig(torch.nn.Conv3d)(
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
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
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
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            # Dropout
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
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
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
    )

    # Modify the scheduler config - only set what's different from default
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
    )

    # Add wandb group
    config.wandb = WandbConfig(
        job_group="ucf101_classification",
        entity="implicit-long-convs",
        project="nvsubq_paper",
    )

    return config
