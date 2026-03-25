"""v5_patch ablation — Attention baseline, patch_size=16.

Grid: 14x14 = 196 patches + 1 CLS + 13 registers = 210 tokens.
Batch: 256/gpu x 1 accum x 8 gpus = 2048 effective.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_attention_net, get_base_config
from experiments.default_cfg import ExperimentConfig


PATCH_SIZE = 16


def get_config() -> ExperimentConfig:
    """Return Attention patch-16 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_attention_net(PATCH_SIZE)
    # Attention does not need compile-compatible FFT path
    config.compile_compatible_fftconv = False
    return config
