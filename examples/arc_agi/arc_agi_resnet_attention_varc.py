# TODO: Add license header here

"""Attention-based config using the VARC-style training wrapper (no conditioning)."""

import os

import torch

from experiments.datamodules.arc_agi import ArcAGIDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.arc_agi_varc_wrapper import ArcAGIVARCWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


PLACEHOLDER = None
WANDB_ENTITY = "dafidofff"

# Dataset
BATCH_SIZE = 32
PRECISION = "bf16-mixed"
MAX_WORKERS = min(16, os.cpu_count() - 1 or 1)

# Model
HIDDEN_DIM = 128
NUM_BLOCKS = 4
NUM_ATTENTION_HEADS = 8
DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.1

# Training
TRAINING_ITERATIONS = 200_000
WARMUP_ITERATIONS_PERCENTAGE = 0.05
GRAD_CLIP = 1.0

WEIGHT_DECAY = 0.01
LEARNING_RATE = 2e-3


def _attention_cfg():
    """Return an attention mixer that mirrors the MNIST attention recipe."""
    return LazyConfig(Attention)(
        hidden_dim="${net.hidden_dim}",
        num_heads=NUM_ATTENTION_HEADS,
        apply_qk_norm=True,
        use_rope=True,
        rope_base=10_000.0,
        is_causal=False,
        attn_dropout=DROPOUT_RATE,
    )


def get_config() -> ExperimentConfig:
    """Return the Attention config trained with the VARC augmentations."""
    config = ExperimentConfig()
    config.debug = False

    config.dataset = LazyConfig(ArcAGIDataModule)(
        data_dir=".data/arc_agi",
        batch_size=BATCH_SIZE,
        num_workers=MAX_WORKERS,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
        include_test_pairs=False,
        normalize_inputs=True,
        one_hot_inputs=False,
        max_grid_size=None,
        input_pad_value=0,
        label_pad_value=-100,
        condition_on_label_mask=False,
    )

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=PLACEHOLDER,
        out_channels=PLACEHOLDER,
        num_blocks=NUM_BLOCKS,
        hidden_dim=HIDDEN_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=HIDDEN_DIM),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=HIDDEN_DIM, out_features=PLACEHOLDER),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=HIDDEN_DIM),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=_attention_cfg(),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=HIDDEN_DIM),
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim=HIDDEN_DIM,
                activation="glu",
                expansion_factor=2.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape=HIDDEN_DIM),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
    )

    config.lightning_wrapper_class = LazyConfig(ArcAGIVARCWrapper)(
        ignore_index=-100,
        num_colors=10,
        canvas_size=64,
        min_scale=1,
        max_scale=4,
        train_views=2,
        val_views=8,
        test_views=16,
    )

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision=PRECISION,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        monitor="val/pixel_acc",
    )

    config.wandb = WandbConfig(
        job_group="arc_agi_attn_varc",
        entity=WANDB_ENTITY,
    )

    return config

