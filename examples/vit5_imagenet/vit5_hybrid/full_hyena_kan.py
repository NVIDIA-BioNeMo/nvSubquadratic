"""Full Hyena with KAN kernel – 12 Hyena blocks, KANLinear kernel generator.

Layout: H H H H H H H H H H H H

Replaces the SIREN MLP in the kernel generator with KANLinear (B-spline)
layers from warpKAN.  No positional embedding is used: raw 2D coordinates
in [-1, 1] are fed directly into the KAN layers.

B-spline defaults: order=3, grid_range=[-5, 5], dx=1/L_cache,
has_mlp=True, enable_standalone_scale_spline=True.
All KAN computation stays in float32.
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    KERNEL_MLP_HIDDEN_DIM,
    _GRID_H,
    build_hybrid_net,
    get_base_config,
)

KAN_NUM_LAYERS = 2
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.kan_kernels_nd import KANKernelND

LAYER_PATTERN = "H" * NUM_BLOCKS


def get_config() -> ExperimentConfig:
    """Return the full-Hyena config with KANLinear kernel generator."""
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
