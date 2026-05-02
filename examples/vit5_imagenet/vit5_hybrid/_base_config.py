"""Shared base config for hybrid ViT-5 experiments: interleaved Hyena + Attention.

Reuses the training recipe, dataset, and model constants from the v5 base.
Adds ``build_hybrid_net`` which produces a ``ViT5ClassificationNet`` with a
``layer_pattern`` + ``layer_types`` dict for interleaved block stacking.

All layers use trunc_normal(std=0.02) initialization uniformly.
Hyena blocks use a learnable GaussianModulationND mask on the SIREN kernel output.

Patch-size-dependent quantities (``num_patches_h/w``, ``grid_w``, ``L_cache``)
use ``"${eval:'${net.image_size} // ${net.patch_size}'}"`` interpolations that
reference the top-level network config keys.  This means ``patch_size`` is the
sole source of truth — changing it on the network config automatically updates
every derived value at ``OmegaConf.resolve()`` time.

Typical usage::

    config.net = build_hybrid_net(layer_pattern="HA" * 6, patch_size=16)
"""

import torch

from examples.vit5_imagenet.v5._base import (
    FINAL_IMAGE_SIZE,
    HIDDEN_DIM,
    INPUT_CHANNELS,
    LAYER_SCALE_INIT,
    MLP_RATIO,
    NUM_BLOCKS,
    NUM_CLASSES,
    PATCH_SIZE,
    get_base_config,
)
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.grn import GlobalResponseNorm
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from mamba_ssm import Mamba as MambaCoreLayer
from nvsubquadratic.modules.mamba_nd import Mamba
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapter
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet
from nvsubquadratic.utils.init import trunc_normal_init, trunc_normal_init_factory
from nvsubquadratic.utils.qk_norm import L2Norm


# Re-export for convenience in leaf configs
__all__ = ["build_hybrid_net", "get_base_config"]


# ─── Initialization (uniform trunc_normal for all layers) ────────────────────
INIT_FN = trunc_normal_init(std=0.02)
INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)

# ─── Attention constants ─────────────────────────────────────────────────────
NUM_HEADS = 6
HEAD_DIM = HIDDEN_DIM // NUM_HEADS  # 64
NUM_REGISTERS = 4
DROP_PATH_RATE = 0.05

# ─── SIREN kernel hyperparameters ────────────────────────────────────────────
KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0


# ─── Interpolation expressions ──────────────────────────────────────────────
# These use the ${eval:'...'} OmegaConf resolver registered in cli.py.
# Inner ${net.xxx} references are resolved first, then the arithmetic is evaluated.
_NUM_PATCHES_W = "${eval:'${net.image_size} // ${net.patch_size}'}"
_NUM_NON_PAD = "${eval:'(${net.image_size} // ${net.patch_size}) ** 2 + 1 + ${net.num_registers}'}"
_PAD_SIZE = "${eval:'(-(${net.image_size} // ${net.patch_size}) ** 2 - 1 - ${net.num_registers}) % (${net.image_size} // ${net.patch_size})'}"
_GRID_H = "${eval:'((${net.image_size} // ${net.patch_size}) ** 2 + 1 + ${net.num_registers} + (-(${net.image_size} // ${net.patch_size}) ** 2 - 1 - ${net.num_registers}) % (${net.image_size} // ${net.patch_size})) // (${net.image_size} // ${net.patch_size})'}"


# ─── Block builders ──────────────────────────────────────────────────────────


def _make_attention_block_cfg() -> LazyConfig:
    """Build a ViT5ResidualBlock config with standard attention.

    All dimension and grid values use ``${eval:'...'}"`` interpolation so they
    stay in sync with the top-level network config.
    """
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(ViT5Attention)(
            hidden_dim="${net.hidden_dim}",
            num_heads=NUM_HEADS,
            num_patches_h=_NUM_PATCHES_W,
            num_patches_w=_NUM_PATCHES_W,
            num_registers="${net.num_registers}",
            qk_norm=LazyConfig(RMSNorm)(
                dim="${eval:'${net.hidden_dim} // ${net.layer_types.A.sequence_mixer_cfg.num_heads}'}", eps=1e-6
            ),
            rope_base=10000.0,
            reg_rope_base=100.0,
            attn_dropout=0.0,
            proj_dropout=0.0,
            qkv_bias=False,
            out_proj_bias=False,
            init_fn_qkv_proj=INIT_FN,
            init_fn_out_proj=INIT_FN,
        ),
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim="${net.hidden_dim}",
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
        hidden_dim="${net.hidden_dim}",
        layer_scale_init=LAYER_SCALE_INIT,
    )


def _make_hyena_block_cfg() -> LazyConfig:
    """Build a ViT5ResidualBlock config with Hyena + GRN (no FiLM).

    All dimension and grid values use ``${eval:'...'}"`` interpolation so they
    stay in sync with the top-level network config.
    """
    mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim="${net.hidden_dim}",
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=2,
                hidden_dim="${net.hidden_dim}",
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim="${net.layer_types.H.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.data_dim}",
                    out_dim="${net.hidden_dim}",
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=_GRID_H,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                ),
                mask_cfg=LazyConfig(GaussianModulationND)(
                    data_dim="${net.layer_types.H.sequence_mixer_cfg.inner_mixer_cfg.mixer_cfg.global_conv_cfg.data_dim}",
                    num_channels="${net.hidden_dim}",
                    min_attenuation_at_step=0.1,
                    max_attenuation_at_limit=0.95,
                    init_extent=1.0,
                    parametrization="direct",
                ),
                grid_type="double",
                fft_padding="zero",
                fft_backend="subq_ops",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                in_channels="${eval:'3 * ${net.hidden_dim}'}",
                out_channels="${eval:'3 * ${net.hidden_dim}'}",
                kernel_size=3,
                groups="${eval:'3 * ${net.hidden_dim}'}",
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            output_norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )

    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(ViT5HyenaAdapter)(
            inner_mixer_cfg=mixer_cfg,
            grid_w=_NUM_PATCHES_W,
        ),
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim="${net.hidden_dim}",
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
        hidden_dim="${net.hidden_dim}",
        layer_scale_init=LAYER_SCALE_INIT,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim="${net.hidden_dim}"),
    )


def _make_mamba_block_cfg(
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    bidirectional: bool = True,
) -> LazyConfig:
    """Build a ViT5ResidualBlock config with a Mamba SSM sequence mixer.

    The flat token sequence (CLS + patches + registers) is passed directly to
    ``Mamba`` which flattens any leading spatial dims and delegates to the
    mamba_ssm core.  No QKVSequenceMixer or ViT5HyenaAdapter wrapper needed.

    Args:
        d_state: SSM state dimension (default 16).
        d_conv: Local convolution width (default 4).
        expand: Inner-dimension expansion factor (default 2).
        bidirectional: If True, a second reversed Mamba core is added and
            summed with the forward pass — important for non-causal 2D inputs.
    """
    return LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=LazyConfig(Mamba)(
            mamba_layer_cfg=LazyConfig(MambaCoreLayer)(
                d_model="${net.hidden_dim}",
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ),
            bidirectional=bidirectional,
        ),
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim="${net.hidden_dim}",
            activation="gelu",
            expansion_factor=float(MLP_RATIO),
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
        hidden_dim="${net.hidden_dim}",
        layer_scale_init=LAYER_SCALE_INIT,
    )


# ─── Network builder ─────────────────────────────────────────────────────────


def build_hybrid_net(
    layer_pattern: str = "HA" * (NUM_BLOCKS // 2),
    patch_size: int = PATCH_SIZE,
    max_drop_path_rate: float = DROP_PATH_RATE,
    drop_path_schedule: str = "constant",
) -> LazyConfig:
    """Build ViT5ClassificationNet with interleaved Hyena/Attention blocks.

    Args:
        layer_pattern: Pattern string where ``"H"`` = Hyena block,
            ``"A"`` = Attention block.  Default ``"HA" * 6`` (12 blocks).
        patch_size: Patch size for the embedding layer (default 16).
            All grid / padding math is derived via
            ``"${eval:'${net.image_size} // ${net.patch_size}'}"``
            interpolation in the nested block configs.
        max_drop_path_rate: Maximum stochastic depth rate.
        drop_path_schedule: ``"constant"`` or ``"linear"`` ramp.

    Returns:
        LazyConfig for ViT5ClassificationNet.
    """
    num_blocks = len(layer_pattern)

    layer_types = {}
    if "A" in layer_pattern:
        layer_types["A"] = _make_attention_block_cfg()
    if "H" in layer_pattern:
        layer_types["H"] = _make_hyena_block_cfg()
    if "M" in layer_pattern:
        layer_types["M"] = _make_mamba_block_cfg()

    return LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=num_blocks,
        patch_size=patch_size,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        readout="cls",
        norm_cfg=LazyConfig(RMSNorm)(dim="${net.hidden_dim}", eps=1e-6),
        layer_pattern=layer_pattern,
        layer_types=layer_types,
        max_drop_path_rate=max_drop_path_rate,
        drop_path_schedule=drop_path_schedule,
    )
