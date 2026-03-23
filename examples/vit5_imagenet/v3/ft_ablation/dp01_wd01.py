"""FiLM finetuning — dp=0.10 + wd=0.1 combo.

This combination hit 82.06% test in v2 (tied #2). Higher WD
compensates for lower drop path rate.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(drop_path_rate=0.10, wd=0.1)
