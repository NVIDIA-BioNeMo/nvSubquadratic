"""Hyena config for gray_scott_reaction_diffusion (v2).

Uses a ResidualNetwork with Hyena (QKV + CKConv global conv) as the
sequence mixer.  Circular FFT padding matches the dataset's periodic
boundary conditions.  With patch_size=16 the effective sequence
resolution is 32×32.

Patch-size CLI override
-----------------------
Only ``net.in_proj_cfg.patch_size=P`` is needed; stride, out_proj patch_size,
and kernel L_cache are derived via OmegaConf interpolators.
"""

import torch

from examples.well.v2.euler_multi_quadrants_periodicBC._base import (
    DATA_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


# ─── Model hyperparameters ────────────────────────────────────────────────────
NUM_HIDDEN_CHANNELS = 384
NUM_BLOCKS = 12
PATCH_SIZE = 16

DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0
GRID_TYPE = "single"
FFT_PADDING = "circular"  # periodic boundary conditions
OMEGA_0 = 30.0

GRADIENT_CHECKPOINTING = True


def get_config() -> ExperimentConfig:
    """Build Hyena experiment config for euler_multi_quadrants_periodicBC."""
    config = get_base_config()

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
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=DATA_DIM,
                        hidden_dim=NUM_HIDDEN_CHANNELS,
                        fft_padding=FFT_PADDING,
                        use_fp16_fft=False,
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim=DATA_DIM,
                            out_dim=NUM_HIDDEN_CHANNELS,
                            mlp_hidden_dim=64,
                            num_layers=3,
                            embedding_dim=64,
                            omega_0=OMEGA_0,
                            L_cache="${eval:'512 // ${net.in_proj_cfg.patch_size}'}",
                            use_bias=True,
                            hidden_omega_0=1.0,
                        ),
                        mask_cfg=LazyConfig(torch.nn.Identity)(),
                        grid_type=GRID_TYPE,
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels=3 * NUM_HIDDEN_CHANNELS,
                        out_channels=3 * NUM_HIDDEN_CHANNELS,
                        kernel_size=3,
                        groups=3 * NUM_HIDDEN_CHANNELS,
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                    gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
                    pixelhyena_norm_cfg=LazyConfig(RMSNorm)(
                        dim=NUM_HIDDEN_CHANNELS,
                    ),
                    output_norm_cfg=LazyConfig(RMSNorm)(
                        dim=NUM_HIDDEN_CHANNELS,
                    ),
                    qk_norm_cfg=LazyConfig(L2Norm)(),
                    use_rope=False,
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
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
    )

    return config
