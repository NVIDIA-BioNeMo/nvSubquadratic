"""WSD finetuning ablation — no Mixup, drop path 0.1, LR=7e-5.

Drop path 0.1 is the strongest single modification (82.05% at ep10).
Testing with LR=7e-5 (between 5e-5 and 1e-4).
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.1, LR=7e-5."""
    return _base(lr=7e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.1)
