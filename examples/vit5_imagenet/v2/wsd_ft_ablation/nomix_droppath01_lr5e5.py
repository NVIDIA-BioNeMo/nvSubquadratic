"""WSD finetuning ablation — no Mixup, drop path 0.1, LR=5e-5.

Higher drop path (0.1) with slightly elevated LR (5e-5) on default WSD schedule.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.1, LR=5e-5."""
    return _base(lr=5e-5, wd=0.05, mixup=0.0, cutmix=0.0, drop_path_rate=0.1)
