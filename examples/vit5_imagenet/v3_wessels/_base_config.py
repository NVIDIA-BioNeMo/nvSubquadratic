"""Shared constants and builder functions for v3_wessels ViT-5 experiments.

Builds on the v3 pretrain base config, adding Snellius-specific paths,
compile_compatible_fftconv, and builder helpers for FiLM, multihead,
register-head, and CLS-row network variants.
"""

import os

import torch

from examples.vit5_imagenet.v3._pretrain_base import (
    FINAL_IMAGE_SIZE,
    HIDDEN_DIM,
    INIT_FN_FACTORY,
    INPUT_CHANNELS,
    NUM_BLOCKS,
    NUM_CLASSES,
    NUM_PATCHES_H,
    NUM_PATCHES_W,
    PATCH_SIZE,
    make_block_cfg,
)
from examples.vit5_imagenet.v3._pretrain_base import (
    get_base_config as _get_v3_base_config,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_multihead_nd import CKConvMultiheadND
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.film import KernelFiLMGenerator, RegisterPooling
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.rms_norm import PerHeadRMSNorm, RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.qk_norm import L2Norm


# Re-export constants used by individual configs
__all__ = [
    "HIDDEN_DIM",
    "NUM_CLASSES",
    "NUM_REGISTERS_CLS",
    "NUM_REGISTERS_NO_CLS",
    "build_cls_row_network",
    "build_depthwise_hyena_mixer",
    "build_film_cfg",
    "build_hyena_mixer",
    "build_multihead_hyena_mixer",
    "get_base_config",
]

# ─── Snellius data paths ─────────────────────────────────────────────────────
IMAGENET_PATH = os.environ.get("IMAGENET_PATH", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageNet/")
IMAGENET_FOLDER_PATH = os.environ.get(
    "IMAGENET_FOLDER_PATH", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder"
)
LOCAL_STAGING_DIR = os.environ.get("LOCAL_STAGING_DIR", "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder")

# ─── Multi-head dimensions ────────────────────────────────────────────────────
NUM_HEADS = 6
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 64

# CLS-row layout: [CLS, reg×13, patch×196] = 210 tokens = 15×14
NUM_REGISTERS_CLS = NUM_PATCHES_W - 1  # 13
# Register-only layout: [reg×14, patch×196] = 210 tokens = 15×14
NUM_REGISTERS_NO_CLS = NUM_PATCHES_W  # 14

# ─── SIREN kernel hyperparameters ────────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

# ─── FiLM conditioning ──────────────────────────────────────────────────────
FILM_HIDDEN_DIM = 64


# ─── Builder functions ──────────────────────────────────────────────────────


def get_base_config() -> ExperimentConfig:
    """Return the v3 base config with Snellius-specific overrides."""
    config = _get_v3_base_config()

    # Snellius data paths
    config.dataset.data_dir = IMAGENET_PATH
    config.dataset.imagefolder_dir = IMAGENET_FOLDER_PATH
    config.dataset.local_staging_dir = LOCAL_STAGING_DIR

    # Compile-compatible FFT convolution for torch.compile
    config.compile_compatible_fftconv = True

    return config


def build_film_cfg() -> LazyConfig:
    """Build FiLM generator config for SIREN kernel conditioning."""
    return LazyConfig(KernelFiLMGenerator)(
        cond_dim=HIDDEN_DIM,
        kernel_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_film_layers=KERNEL_NUM_LAYERS - 1,
        film_hidden_dim=FILM_HIDDEN_DIM,
    )


def build_depthwise_hyena_mixer(
    *,
    film_cfg: LazyConfig | None = None,
    use_rope: bool = False,
) -> LazyConfig:
    """Build depthwise CKConvND-based Hyena mixer (QKVSequenceMixer)."""
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=HIDDEN_DIM,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=NUM_PATCHES_H + 1,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                    film_cfg=film_cfg,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * HIDDEN_DIM,
                out_channels=3 * HIDDEN_DIM,
                kernel_size=3,
                groups=3 * HIDDEN_DIM,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            use_rope=use_rope,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )


# Backward-compat alias used by existing configs
build_hyena_mixer = build_depthwise_hyena_mixer


def build_multihead_hyena_mixer(
    *,
    film_cfg: LazyConfig | None = None,
    use_rope: bool = False,
) -> LazyConfig:
    """Build multi-head CKConvMultiheadND-based Hyena mixer (QKVSequenceMixer).

    6 heads, head_dim=64, dense within-head channel mixing.
    """
    kernel_out_dim = NUM_HEADS * HEAD_DIM * HEAD_DIM  # dense kernel per head
    return LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvMultiheadND)(
                data_dim=2,
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=2,
                    out_dim=kernel_out_dim,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=NUM_PATCHES_H + 1,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                    film_cfg=film_cfg,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="double",
                fft_padding="zero",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels=3 * HIDDEN_DIM,
                out_channels=3 * HIDDEN_DIM,
                kernel_size=3,
                groups=3 * HIDDEN_DIM,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(PerHeadRMSNorm)(num_heads=NUM_HEADS, head_dim=HEAD_DIM, eps=1e-6),
            use_rope=use_rope,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )


def build_cls_row_network(
    mixer_cfg: LazyConfig,
    *,
    num_registers: int = NUM_REGISTERS_CLS,
    use_cls_token: bool = True,
    register_start_idx: int = 1,
    register_head_cfg: LazyConfig | None = None,
    grn_cfg: LazyConfig | None = None,
    find_unused_parameters: bool = False,
) -> tuple[LazyConfig, dict]:
    """Build ViT5ClassificationNet config and extra trainer overrides.

    Uses make_block_cfg from v3 base for correct init/bias/MLP settings.

    Returns:
        (net_cfg, trainer_overrides) tuple. Caller should apply trainer_overrides
        to config.trainer after calling this.
    """
    register_pooling_cfg = LazyConfig(RegisterPooling)(num_registers=num_registers)

    block_kwargs = dict(
        register_pooling_cfg=register_pooling_cfg,
        num_registers=num_registers,
        register_start_idx=register_start_idx,
    )
    if grn_cfg is not None:
        block_kwargs["grn_cfg"] = grn_cfg

    block_cfg = make_block_cfg(
        sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=mixer_cfg,
            grid_w=NUM_PATCHES_W,
        ),
        **block_kwargs,
    )

    net_cfg = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=num_registers,
        dropout_rate=0.0,
        use_cls_token=use_cls_token,
        prepend_registers=True,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        register_head_cfg=register_head_cfg,
        block_cfg=block_cfg,
    )

    trainer_overrides = {}
    if find_unused_parameters:
        trainer_overrides["find_unused_parameters"] = True

    return net_cfg, trainer_overrides
