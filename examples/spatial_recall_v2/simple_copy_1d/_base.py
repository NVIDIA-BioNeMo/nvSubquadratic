# TODO: Add license header here

"""Shared config for 1D simple copy — spatial recall v2.

Task: flattened EMNIST digit (16x16 → 256 elements) placed in a 1D canvas
(64x64 → 4096 elements).  The model must recall the segment at the end of
the sequence (causal / autoregressive formulation).

v2 modernisations (relative to v1):
- bf16-mixed precision
- RMSNorm instead of LayerNorm
- IterationSpeedCallback for throughput logging
- torch.compile support (set per model config)
"""

import torch

from experiments.callbacks.iteration_speed import IterationSpeedCallback
from experiments.callbacks.sequence_visualization_1d import Sequence1DVisualizationCallback
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.regression_wrapper import RegressionWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init


# ─── Dataset constants ────────────────────────────────────────────────────────
TARGET_SIZE = 16
CANVAS_SIZE = 64  # canvas_length = 64² = 4096

# ─── Training constants ──────────────────────────────────────────────────────
TRAINING_ITERATIONS = 20_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 10.0
PRECISION = "bf16-mixed"
NUM_WORKERS = 4


def base_experiment_config(
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_dim: int = 160,
    num_blocks: int = 4,
    training_iterations: int = TRAINING_ITERATIONS,
) -> ExperimentConfig:
    """Return config with everything except the mixer and compile flags.

    After calling this, set:
        ``config.net.block_cfg.sequence_mixer_cfg`` (from mixer_defaults)
    """
    from experiments.datamodules.emnist import EMNISTDataModule
    from experiments.datamodules.spatial_recall_dataset import SpatialRecall1DDataModule

    config = ExperimentConfig()
    config.debug = False

    # ── Compile (on by default; Mamba configs override to False) ──────
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    # ── Network (1D: data_dim=1, mixer = PLACEHOLDER) ─────────────────
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=in_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        data_dim=1,
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
        # target_size² for 1D (flattened image segment)
        target_size="${dataset.target_size} * ${dataset.target_size}",
    )

    # ── Dataset ───────────────────────────────────────────────────────
    config.dataset = LazyConfig(SpatialRecall1DDataModule)(
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
        placement="fixed",
        with_mask=False,
        num_items=1,
        readout_value=0.0,
    )

    # ── Wrapper / Optimiser / Schedule ────────────────────────────────
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

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # ── Wandb ─────────────────────────────────────────────────────────
    config.wandb = WandbConfig(
        entity="implicit-long-convs",
        project="nvsubquadratic",
        job_group="spatial_recall_v2_1d_simple_copy",
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    config.callbacks = [
        LazyConfig(Sequence1DVisualizationCallback)(
            num_samples=8,
            target_size="${dataset.target_size}",
            every_n_train_steps=2000,
            readout_value="${dataset.readout_value}",
        ),
        LazyConfig(IterationSpeedCallback)(
            log_every_n_steps=10,
            batch_size_per_gpu="${train.batch_size}",
        ),
    ]

    return config
