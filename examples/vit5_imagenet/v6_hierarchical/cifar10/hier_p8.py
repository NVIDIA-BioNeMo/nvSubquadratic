"""CIFAR-10 patch-size ablation: hierarchical, patch_size=8.

3 stages: 8×8 → 4×4 → 2×2, dims=[96,192,384], depths=[2,2,6].
"""

from examples.vit5_imagenet.v6_hierarchical.cifar10._base import build_hier_config


def get_config():
    """Return the hierarchical patch-8 CIFAR-10 experiment config."""
    return build_hier_config(patch_size=8)
