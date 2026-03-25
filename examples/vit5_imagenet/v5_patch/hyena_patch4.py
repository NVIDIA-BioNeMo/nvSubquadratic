"""v5_patch ablation — Hyena (CLS-row + FiLM + GRN), patch_size=4.

Grid: 56x56 patches + 1 CLS + 55 registers = 57x56 = 3192 tokens.
Batch: 16/gpu x 16 accum x 8 gpus = 2048 effective.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_hyena_net, get_base_config
from experiments.default_cfg import ExperimentConfig


PATCH_SIZE = 4


def get_config() -> ExperimentConfig:
    """Return Hyena patch-4 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_hyena_net(PATCH_SIZE)
    return config
