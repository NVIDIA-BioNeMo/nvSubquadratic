"""v5_patch ablation — Hyena (CLS-row + FiLM + GRN), patch_size=8.

Grid: 28x28 patches + 1 CLS + 27 registers = 29x28 = 812 tokens.
Batch: 64/gpu x 4 accum x 8 gpus = 2048 effective.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_hyena_net, get_base_config
from experiments.default_cfg import ExperimentConfig


PATCH_SIZE = 8


def get_config() -> ExperimentConfig:
    """Return Hyena patch-8 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_hyena_net(PATCH_SIZE)
    return config
