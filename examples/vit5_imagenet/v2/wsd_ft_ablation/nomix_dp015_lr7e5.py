"""WSD finetuning ablation — no Mixup, drop path 0.15, LR=7e-5.

Drop path 0.15 set the record at 82.07%. Testing with slightly higher LR.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.15, LR=7e-5."""
    return _base(lr=7e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.15)
