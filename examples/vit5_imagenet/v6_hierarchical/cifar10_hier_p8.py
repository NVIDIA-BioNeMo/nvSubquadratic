"""CIFAR-10 patch-size ablation: hierarchical, patch_size=8.

2 stages: 4×4 → 2×2, dims=[192,384], depths=[4,6].
"""

from examples.vit5_imagenet.v6_hierarchical._cifar10_patch_ablation_base import (
    build_hier_config,
)


def get_config():
    """Return the hierarchical patch-8 CIFAR-10 experiment config."""
    return build_hier_config(patch_size=8)
