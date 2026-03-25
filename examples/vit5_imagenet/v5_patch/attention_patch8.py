"""v5_patch ablation — Attention baseline, patch_size=8.

Grid: 28x28 = 784 patches + 1 CLS + 27 registers = 812 tokens.
Batch: 64/gpu x 4 accum x 8 gpus = 2048 effective.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_attention_net, get_base_config
from experiments.default_cfg import ExperimentConfig

PATCH_SIZE = 8


def get_config() -> ExperimentConfig:
    """Return Attention patch-8 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_attention_net(PATCH_SIZE)
    config.compile_compatible_fftconv = False
    return config
