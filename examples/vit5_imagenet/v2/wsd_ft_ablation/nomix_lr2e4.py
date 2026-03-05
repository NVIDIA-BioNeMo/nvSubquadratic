"""WSD finetuning ablation — no Mixup, LR=2e-4 (between 1e-4 and 3e-4)."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, LR=2e-4."""
    return _base(lr=2e-4, wd=0.05, mixup=0.0, cutmix=0.0)
