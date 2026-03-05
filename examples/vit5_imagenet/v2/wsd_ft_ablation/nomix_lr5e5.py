"""WSD finetuning ablation — no-mixup LR sweep: lr=5e-5."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config for no-mixup LR=5e-5, WD=0.05."""
    return _base(lr=5e-5, wd=0.05, mixup=0.0, cutmix=0.0)
