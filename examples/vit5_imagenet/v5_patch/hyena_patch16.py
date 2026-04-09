"""v5_patch ablation — Hyena (CLS-row + FiLM + GRN), patch_size=16.

Grid: 14x14 patches + 1 CLS + 13 registers = 15x14 = 210 tokens.
Batch: 256/gpu x 1 accum x 8 gpus = 2048 effective.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_hyena_net, get_base_config
from experiments.default_cfg import ExperimentConfig


PATCH_SIZE = 16


def get_config() -> ExperimentConfig:
    """Return Hyena patch-16 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_hyena_net(PATCH_SIZE)
    return config
