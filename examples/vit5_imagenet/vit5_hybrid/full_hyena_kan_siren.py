"""Full Hyena with KAN kernel – SIREN-style activations and tight grid_range.

Identical to full_hyena_kan.py except:
  - grid_range = [-1.0, 1.0]  (was [-5.0, 5.0])
  - KANLinear base_activation = Sine (torch.sin)  (was SiLU)
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
from nvsubquadratic.modules.kernels_nd import Sine

LAYER_PATTERN = "H" * NUM_BLOCKS


def get_config() -> ExperimentConfig:
    """Return the full-Hyena config with KAN kernel using sin activations and grid_range=[-1, 1]."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

    kernel_cfg = LazyConfig(KANKernelND)(
        data_dim=2,
        out_dim="${net.hidden_dim}",
        mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
        num_layers=KAN_NUM_LAYERS,
        L_cache=_GRID_H,
        grid_range=[-1.0, 1.0],
        base_activation=Sine,
    )

    config.net = build_hybrid_net(
        layer_pattern=LAYER_PATTERN,
        patch_size=PATCH_SIZE,
        hyena_kernel_cfg=kernel_cfg,
    )
    config.wandb.job_group = "vit5_hybrid"
    return config
