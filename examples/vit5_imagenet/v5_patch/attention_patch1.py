"""v5_patch ablation — Attention baseline, patch_size=1.

Grid: 224x224 = 50176 patches + 1 CLS + 4 registers = 50181 tokens.
Batch: 1/gpu x 256 accum x 8 gpus = 2048 effective.

WARNING: O(n^2) attention on ~50K tokens is almost certainly infeasible
on H100 80GB. This config exists for completeness — expect OOM.
Consider activation checkpointing or skip this config for attention.
"""

from examples.vit5_imagenet.v5_patch._base_config import build_attention_net, get_base_config
from experiments.default_cfg import ExperimentConfig


PATCH_SIZE = 1


def get_config() -> ExperimentConfig:
    """Return Attention patch-1 config."""
    config = get_base_config(PATCH_SIZE)
    config.net = build_attention_net(PATCH_SIZE)
    config.compile_compatible_fftconv = False
    return config
