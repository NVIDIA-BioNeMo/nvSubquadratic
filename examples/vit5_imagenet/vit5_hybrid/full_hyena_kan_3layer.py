"""Full Hyena with KAN kernel – 3 KAN layers, shape 2 → 32 → 32 → out_dim.

Identical to full_hyena_kan.py except KAN_NUM_LAYERS = 3, giving KAN layer
dims data_dim (2) → mlp_hidden_dim (32) → mlp_hidden_dim (32) → out_dim.
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    KERNEL_MLP_HIDDEN_DIM,
    _GRID_H,
    build_hybrid_net,
    get_base_config,
)

KAN_NUM_LAYERS = 3
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.kan_kernels_nd import KANKernelND

LAYER_PATTERN = "H" * NUM_BLOCKS


def get_config() -> ExperimentConfig:
    """Return the full-Hyena config with 3-layer KANLinear kernel generator."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    kernel_cfg = LazyConfig(KANKernelND)(
        data_dim=2,
        out_dim="${net.hidden_dim}",
        mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_layers=KAN_NUM_LAYERS,
        L_cache=_GRID_H,
    )

    config.net = build_hybrid_net(
        layer_pattern=LAYER_PATTERN,
        patch_size=PATCH_SIZE,
        hyena_kernel_cfg=kernel_cfg,
    )
    config.wandb.job_group = "vit5_hybrid"
    return config
