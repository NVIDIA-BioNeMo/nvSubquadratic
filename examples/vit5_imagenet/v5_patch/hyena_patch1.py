"""v5_patch ablation — Hyena (CLS-row + FiLM + GRN), patch_size=1.

Grid: 224x224 patches + 1 CLS + 223 registers = 225x224 = 50400 tokens.
Batch: 1/gpu x 256 accum x 8 gpus = 2048 effective.

NOTE: Extremely long sequence — this is the most memory-intensive config.
May require activation checkpointing or reduced model size to fit.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_hyena_net, get_base_config
from experiments.default_cfg import ExperimentConfig

PATCH_SIZE = 1


def get_config() -> ExperimentConfig:
    """Return Hyena patch-1 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_hyena_net(PATCH_SIZE)
    return config
