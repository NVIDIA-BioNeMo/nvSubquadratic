"""WSD finetuning ablation — ThreeAugment (pretraining recipe), no label smoothing."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with ThreeAugment instead of RandAugment."""
    return _base(lr=3e-5, wd=0.05, use_three_augment=True, rand_augment=None, smoothing=0.0)
