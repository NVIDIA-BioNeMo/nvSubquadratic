"""WSD finetuning ablation — no Mixup, drop path 0.15, WD=0.1.

dp015 at 82.07%, and dp01_wd01 holding perfectly at 82.04%.
Higher WD may stabilize the dp 0.15 run even further.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.15, WD=0.1."""
    return _base(lr=3e-5, wd=0.1, mixup=0.0, cutmix=0.0, drop_path_rate=0.15)
