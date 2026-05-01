# TODO: Add license header here

"""Shared config for 3D color conditioning -- spatial recall v2.

Task: 4 EMNIST digits with coloured bounding boxes placed on depth slices of a
3D volume [D, H, W].  The model must output the target digit coloured to match
its frame at the back-bottom-right readout region.

v2 modernisations (relative to v1):
- RMSNorm instead of LayerNorm
- IterationSpeedCallback for throughput logging
- torch.compile support (set per model config)
"""

import torch

from experiments.callbacks.image_grid_val_visualization import ValidationVolumeGridCallback
from experiments.callbacks.iteration_speed import IterationSpeedCallback
from experiments.default_cfg import (
    ExperimentConfig,
    SchedulerConfig,
    TrainConfig,
    TrainerConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.regression_wrapper import RegressionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init


# --- Dataset constants --------------------------------------------------------
TARGET_SIZE = 16
CANVAS_SIZE = 64
CANVAS_DEPTH = 8
INPUT_CHANNELS = 3  # RGB with coloured frames
OUTPUT_CHANNELS = 3  # RGB digit in frame colour
NUM_ITEMS = 4

# --- Training constants -------------------------------------------------------
TRAINING_ITERATIONS = 50_000
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 10.0
PRECISION = "bf16-mixed"
NUM_WORKERS = 4


def base_experiment_config(
    in_channels: int = INPUT_CHANNELS,
    out_channels: int = OUTPUT_CHANNELS,
    hidden_dim: int = 160,
    num_blocks: int = 4,
    training_iterations: int = TRAINING_ITERATIONS,
) -> ExperimentConfig:
    """Return config with everything except the mixer and compile flags.

    After calling this, set:
        ``config.net.block_cfg.sequence_mixer_cfg`` (from mixer_defaults)
    """
    from experiments.datamodules.emnist import EMNISTDataModule
    from experiments.datamodules.spatial_recall_dataset import SpatialRecall3DDataModule

    config = ExperimentConfig()
    config.debug = False

    # -- Compile (on by default; Mamba configs override to False) ------
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    # -- Network (3D: data_dim=3, mixer = PLACEHOLDER) -----------------
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        data_dim=3,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=PLACEHOLDER,
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p="${net.block_cfg.dropout_cfg.p}"),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
        # For 3D, the readout is (1, target_size, target_size) -- last depth slice
        target_size=(1, "${dataset.target_size}", "${dataset.target_size}"),
    )

    # -- Dataset -------------------------------------------------------
    config.dataset = LazyConfig(SpatialRecall3DDataModule)(
        base_datamodule_cfg=LazyConfig(EMNISTDataModule)(
            data_dir=".data/emnist",
            batch_size=BATCH_SIZE,
            data_type="image",
            num_workers=NUM_WORKERS,
            pin_memory=True,
            permuted=False,
            seed="${seed}",
            normalize_input=True,
            split="byclass",
        ),
        target_size=TARGET_SIZE,
        canvas_size=CANVAS_SIZE,
        canvas_depth=CANVAS_DEPTH,
        data_type="volume",
        placement="random",
        with_mask=False,
        use_colored_frames=True,
        num_items=NUM_ITEMS,
        readout_value=0.0,
        colored_label=True,
    )

    # -- Wrapper / Optimiser / Schedule --------------------------------
    config.lightning_wrapper_class = LazyConfig(RegressionWrapper)(metric="MSE")

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.base_datamodule_cfg.batch_size}",
        iterations=training_iterations,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    config.trainer = TrainerConfig(
        checkpoint_every_n_steps=2000,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # -- Wandb ---------------------------------------------------------
    config.wandb = WandbConfig(
        entity="implicit-long-convs",
        project="nvsubquadratic",
        job_group="spatial_recall_v2_3d_color_conditioning",
    )

    # -- Callbacks -----------------------------------------------------
    config.callbacks = [
        LazyConfig(ValidationVolumeGridCallback)(
            num_samples=8,
            every_n_epochs=None,
            every_n_train_steps=2000,
            show_mask_separately=False,
            target_size="${dataset.target_size}",
        ),
        LazyConfig(IterationSpeedCallback)(
            log_every_n_steps=10,
            batch_size_per_gpu="${train.batch_size}",
        ),
    ]

    return config
