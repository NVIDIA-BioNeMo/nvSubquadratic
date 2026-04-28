"""Full Hyena with KAN kernel – patch_size=8, grid_range=[-2, 2].

Identical to full_hyena_kan.py except:
  - patch_size = 8       (was 16 default in v5._base.PATCH_SIZE)
  - grid_range = [-2.0, 2.0]  (was [-5.0, 5.0] default in KANKernelND)
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    KERNEL_MLP_HIDDEN_DIM,
    _GRID_H,
    build_hybrid_net,
    get_base_config,
)

KAN_NUM_LAYERS = 3
PATCH_SIZE = 8
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.kan_kernels_nd import KANKernelND

LAYER_PATTERN = "H" * NUM_BLOCKS


def get_config() -> ExperimentConfig:
    """Return the full-Hyena KAN config with patch_size=8 and grid_range=[-2, 2]."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "default"

    kernel_cfg = LazyConfig(KANKernelND)(
        data_dim=2,
        out_dim="${net.hidden_dim}",
        mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_layers=KAN_NUM_LAYERS,
        L_cache=_GRID_H,
        grid_range=[-2.0, 2.0],
    )

    config.net = build_hybrid_net(
        layer_pattern=LAYER_PATTERN,
        patch_size=PATCH_SIZE,
        hyena_kernel_cfg=kernel_cfg,
    )
    config.wandb.job_group = "vit5_hybrid"
    return config
