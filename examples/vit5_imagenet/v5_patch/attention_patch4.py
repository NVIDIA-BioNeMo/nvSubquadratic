"""v5_patch ablation — Attention baseline, patch_size=4.

Grid: 56x56 = 3136 patches + 1 CLS + 4 registers = 3141 tokens.
Batch: 16/gpu x 16 accum x 8 gpus = 2048 effective.

NOTE: O(n^2) attention on 3141 tokens — may be slow but should fit on H100.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_attention_net, get_base_config
from experiments.default_cfg import ExperimentConfig


PATCH_SIZE = 4


def get_config() -> ExperimentConfig:
    """Return Attention patch-4 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_attention_net(PATCH_SIZE)
    config.compile_compatible_fftconv = False
    return config
