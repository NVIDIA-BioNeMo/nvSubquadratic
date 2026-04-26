"""Attention config for MHD_64 (v2).

Uses a ResidualNetwork with multi-head self-attention (QKV + RoPE) as the
sequence mixer.  With patch_size=8 the effective sequence resolution is 8×8×8.

Patch-size CLI override
-----------------------
Only ``net.in_proj_cfg.patch_size=P`` is needed; stride and out_proj patch_size
are derived via OmegaConf interpolators.
"""

import torch

from examples.well.v2.MHD_64._base import (
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
# NUM_HIDDEN_CHANNELS / NUM_BLOCKS / NUM_HEADS match the v2 baseline shared by
# `acoustic_scattering_maze` and `active_matter`.  PATCH_SIZE is kept at 8 (vs 16
# in the 2D configs) because MHD_64 is 3D 64³: patch_size=16 would give only
# 4×4×4 = 64 tokens, too coarse for a 3D PDE.
NUM_HIDDEN_CHANNELS = 384
NUM_BLOCKS = 12
NUM_HEADS = 6
PATCH_SIZE = 8

DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0


def get_config() -> ExperimentConfig:
    """Build Attention experiment config for MHD_64."""
    # NOTE: Override _base.py defaults (LR=5e-3, WD=1e-4 — tuned for CNextU-net,
    # paper Table 6) with v1 attention/Hyena defaults (LR=1e-3, WD=1e-5).
    # The 5e-3 LR has only been validated for CNextU-net on MHD_64.
    config = get_base_config(learning_rate=1e-3, weight_decay=1e-5)

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
            stride="${net.in_proj_cfg.patch_size}",
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features=NUM_HIDDEN_CHANNELS,
            out_features=OUT_CHANNELS,
            data_dim=DATA_DIM,
            patch_size="${net.in_proj_cfg.patch_size}",
            stride="${net.in_proj_cfg.patch_size}",
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
                    rope_base=10000.0,
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
