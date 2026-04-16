"""Hybrid Hyena/Attention – alternating HA pattern (12 blocks).

Layout: H A H A H A H A H A H A
         └─────── 6 pairs ─────┘

Override ``net.patch_size`` to change resolution (default 16).
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    build_hybrid_net,
    get_base_config,
)
from experiments.default_cfg import ExperimentConfig


LAYER_PATTERN = "HA" * (NUM_BLOCKS // 2)


def get_config() -> ExperimentConfig:
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = build_hybrid_net(layer_pattern=LAYER_PATTERN, patch_size=PATCH_SIZE)
    config.wandb.job_group = "vit5_hybrid"
    return config
