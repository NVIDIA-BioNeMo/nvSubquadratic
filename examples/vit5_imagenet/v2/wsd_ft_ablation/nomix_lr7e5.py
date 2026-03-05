"""WSD finetuning ablation — no Mixup, LR=7e-5 (between 5e-5 and 1e-4)."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, LR=7e-5."""
    return _base(lr=7e-5, wd=0.05, mixup=0.0, cutmix=0.0)
