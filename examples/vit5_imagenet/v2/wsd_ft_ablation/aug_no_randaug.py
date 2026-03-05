"""WSD finetuning ablation — no augmentation policy (bare baseline)."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no augmentation policy."""
    return _base(lr=3e-5, wd=0.05, use_three_augment=False, rand_augment=None)
