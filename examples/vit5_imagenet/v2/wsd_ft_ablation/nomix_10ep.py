"""WSD finetuning ablation — no-mixup shorter training: 10 epochs."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config for no-mixup 10 epochs, LR=3e-5, WD=0.05."""
    return _base(epochs=10, lr=3e-5, wd=0.05, mixup=0.0, cutmix=0.0)
