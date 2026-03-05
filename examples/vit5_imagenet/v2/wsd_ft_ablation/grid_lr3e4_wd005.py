"""WSD finetuning ablation — LR=3e-4, WD=0.05."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config for LR=3e-4, WD=0.05."""
    return _base(lr=3e-4, wd=0.05)
