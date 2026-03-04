"""WSD finetuning ablation — no Mixup, LR=5e-5, WD=0.1 (higher weight decay)."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, LR=5e-5, WD=0.1."""
    return _base(lr=5e-5, wd=0.1, mixup=0.0, cutmix=0.0)
