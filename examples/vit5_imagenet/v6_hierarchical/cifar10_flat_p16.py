"""CIFAR-10 patch-size ablation: flat (no merging), patch_size=16, grid=2×2, dim=384."""

from examples.vit5_imagenet.v6_hierarchical._cifar10_patch_ablation_base import (
    build_flat_config,
)


def get_config():
    """Return the flat patch-16 CIFAR-10 experiment config."""
    return build_flat_config(patch_size=16)
