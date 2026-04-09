"""Attention (ViT-style) config for euler_multi_quadrants_periodicBC.

Uses a ResidualNetwork with multi-head self-attention + 2D RoPE as the
sequence mixer.  Same patchification and backbone as the Hyena config.

With patch_size=16 the effective sequence resolution is 32×32 = 1024 tokens.
"""

import torch

from examples.well.v1.euler_multi_quadrants_periodicBC._base import (
    DATA_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init


# ─── Model hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE = 24
NUM_HIDDEN_CHANNELS = 384
NUM_BLOCKS = 12
NUM_HEADS = 12  # head_dim = 384 / 12 = 32
PATCH_SIZE = 16

DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


def get_config() -> ExperimentConfig:
    """Build Attention experiment config for euler_multi_quadrants_periodicBC."""
    config = get_base_config(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    norm_cfg = LazyConfig(RMSNorm)(dim=NUM_HIDDEN_CHANNELS)

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features=IN_CHANNELS,
            out_features=NUM_HIDDEN_CHANNELS,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features=NUM_HIDDEN_CHANNELS,
            out_features=OUT_CHANNELS,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        norm_cfg=norm_cfg,
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim=NUM_HIDDEN_CHANNELS,
                mixer_cfg=LazyConfig(Attention)(
                    hidden_dim=NUM_HIDDEN_CHANNELS,
                    num_heads=NUM_HEADS,
                    apply_qk_norm=True,
                    use_rope=True,
                    is_causal=False,
                    attn_dropout=0.0,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg=norm_cfg,
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim=NUM_HIDDEN_CHANNELS,
                activation="glu",
                expansion_factor=1.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg=norm_cfg,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_IN_RATE),
    )

    return config
