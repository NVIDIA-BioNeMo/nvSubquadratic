"""WSD finetuning ablation — drop path rate 0.1 (doubled from default 0.05)."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with drop path rate 0.1."""
    return _base(lr=3e-5, wd=0.05, drop_path_rate=0.1)
