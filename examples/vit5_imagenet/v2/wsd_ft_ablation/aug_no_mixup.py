"""WSD finetuning ablation — no Mixup/CutMix."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with Mixup and CutMix disabled."""
    return _base(lr=3e-5, wd=0.05, mixup=0.0, cutmix=0.0)
