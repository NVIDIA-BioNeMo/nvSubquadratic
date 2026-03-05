"""WSD finetuning ablation — no EMA."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no EMA."""
    config = _base(lr=3e-5, wd=0.05)
    config.callbacks = []
    config.trainer.checkpoint_monitor = "val/acc"
    return config
