"""ViT-5-Small ImageNet-1k pretrain — attention baseline with EMA.

Matches the reference ViT-5-Small (82.2% top-1).  Only the
attention-specific network definition lives here; everything else
(dataset, optimizer, scheduler, EMA, etc.) comes from ``_base``.
"""

from examples.vit5_imagenet.v3._pretrain_base import (
    FINAL_IMAGE_SIZE,
    HIDDEN_DIM,
    INIT_FN,
    INPUT_CHANNELS,
    NUM_BLOCKS,
    NUM_CLASSES,
    NUM_PATCHES_H,
    NUM_PATCHES_W,
    PATCH_SIZE,
    get_base_config,
    make_block_cfg,
)
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.vit5_attention import ViT5Attention
from nvsubquadratic.networks.vit5_classification import ViT5ClassificationNet


NUM_HEADS = 6
NUM_REGISTERS = 4


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small attention pretrain config with EMA."""
    config = get_base_config()

    config.net = LazyConfig(ViT5ClassificationNet)(
        in_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        image_size=FINAL_IMAGE_SIZE,
        num_registers=NUM_REGISTERS,
        dropout_rate=0.0,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        block_cfg=make_block_cfg(
            sequence_mixer_cfg=LazyConfig(ViT5Attention)(
                hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS,
                num_patches_h=NUM_PATCHES_H,
                num_patches_w=NUM_PATCHES_W,
                num_registers=NUM_REGISTERS,
                qk_norm=LazyConfig(RMSNorm)(dim=HIDDEN_DIM // NUM_HEADS, eps=1e-6),
                rope_base=10000.0,
                reg_rope_base=100.0,
                attn_dropout=0.0,
                proj_dropout=0.0,
                qkv_bias=False,
                out_proj_bias=False,
                init_fn_qkv_proj=INIT_FN,
                init_fn_out_proj=INIT_FN,
            ),
        ),
    )

    return config
