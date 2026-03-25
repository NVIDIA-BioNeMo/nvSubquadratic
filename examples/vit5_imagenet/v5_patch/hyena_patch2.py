"""v5_patch ablation — Hyena (CLS-row + FiLM + GRN), patch_size=2.

Grid: 112x112 patches + 1 CLS + 111 registers = 113x112 = 12656 tokens.
Batch: 4/gpu x 64 accum x 8 gpus = 2048 effective.

NOTE: Long sequence — may require significant memory. Monitor GPU OOM.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_hyena_net, get_base_config
from experiments.default_cfg import ExperimentConfig

PATCH_SIZE = 2


def get_config() -> ExperimentConfig:
    """Return Hyena patch-2 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_hyena_net(PATCH_SIZE)
    return config
