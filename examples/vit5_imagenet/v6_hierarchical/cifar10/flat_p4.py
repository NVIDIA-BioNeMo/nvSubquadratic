"""CIFAR-10 patch-size ablation: flat (no merging), patch_size=4, grid=16×16, dim=384."""

from examples.vit5_imagenet.v6_hierarchical.cifar10._base import build_flat_config


def get_config():
    """Return the flat patch-4 CIFAR-10 experiment config."""
    return build_flat_config(patch_size=4)
