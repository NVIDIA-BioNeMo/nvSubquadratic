"""WSD finetuning ablation — LR=1e-5, WD=0.1."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config for LR=1e-5, WD=0.1."""
    return _base(lr=1e-5, wd=0.1)
