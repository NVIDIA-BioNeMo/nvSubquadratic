"""CIFAR-10 patch-size ablation (64×64): hierarchical, patch_size=16.

2 stages: 4×4 → 2×2, dims=[384,768], depths=[6,2]  ← Swin stages 3-4.
"""

from examples.vit5_imagenet.v6_hierarchical._cifar10_patch_ablation_base import (
    build_hier_config,
)


def get_config():
    """Return the hierarchical patch-16 CIFAR-10 experiment config."""
    return build_hier_config(patch_size=16)
