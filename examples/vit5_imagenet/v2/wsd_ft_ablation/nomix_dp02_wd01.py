"""WSD finetuning ablation — no Mixup, drop path 0.2, WD=0.1.

Higher drop path + higher WD. dp015 hit 82.07% and dp01_wd01 hit 82.06%.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.2, WD=0.1."""
    return _base(lr=3e-5, wd=0.1, mixup=0.0, cutmix=0.0, drop_path_rate=0.2)
