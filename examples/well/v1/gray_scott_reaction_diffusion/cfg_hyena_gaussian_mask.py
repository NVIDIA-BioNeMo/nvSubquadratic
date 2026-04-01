"""Hyena + Gaussian mask config for gray_scott_reaction_diffusion.

With patch_size=2 the effective sequence resolution is 64×64 (4096 tokens).
This matches the token count of the Euler patch_size=8 setup.

For raw-pixel experiments (patch_size=1 → 128×128 = 16384 tokens),
override via CLI:
    net.in_proj_cfg.patch_size=1  net.in_proj_cfg.stride=1
    net.out_proj_cfg.patch_size=1  net.out_proj_cfg.stride=1
    net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.L_cache=128
    dataset.batch_size=6
"""

import torch

from examples.well.v1.gray_scott_reaction_diffusion._base import (
    DATA_DIM,
    IN_CHANNELS,
    OUT_CHANNELS,
    get_base_config,
)
from experiments.callbacks.iteration_speed import IterationSpeedCallback
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork
from nvsubquadratic.utils.init import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.utils.qk_norm import L2Norm


PLACEHOLDER = None

# ─── Model hyperparameters ────────────────────────────────────────────────────
BATCH_SIZE = 24
NUM_HIDDEN_CHANNELS = 384
NUM_BLOCKS = 12
PATCH_SIZE = 2

DROPOUT_IN_RATE = 0.0
DROPOUT_RATE = 0.0
GRID_TYPE = "single"
FFT_PADDING = "circular"  # Periodic boundary conditions
OMEGA_0 = 30.0

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

PATCHED_RESOLUTION = 128 // PATCH_SIZE  # 64


def get_config() -> ExperimentConfig:
    """Build Hyena + Gaussian mask config for gray_scott_reaction_diffusion."""
    config = get_base_config(
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.compile_compatible_fftconv = False

    norm_cfg = LazyConfig(RMSNorm)(dim=NUM_HIDDEN_CHANNELS)

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features=PLACEHOLDER,
            out_features=PLACEHOLDER,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features=PLACEHOLDER,
            out_features=PLACEHOLDER,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
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
                            L_cache=PATCHED_RESOLUTION,
                            use_bias=True,
                            hidden_omega_0=1.0,
                        ),
                        mask_cfg=LazyConfig(GaussianModulationND)(
                            data_dim=DATA_DIM,
                            num_channels=NUM_HIDDEN_CHANNELS,
                            min_attenuation_at_step=0.1,
                            max_attenuation_at_limit=0.95,
                            init_extent=0.75,
                            parametrization="direct",
                        ),
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
    )

    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))
    config.callbacks.append(LazyConfig(IterationSpeedCallback)(log_every_n_steps=10))

    return config
