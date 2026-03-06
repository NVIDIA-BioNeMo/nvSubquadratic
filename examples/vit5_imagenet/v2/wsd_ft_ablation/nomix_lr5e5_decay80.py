"""WSD finetuning ablation — no Mixup, LR=5e-5, WSD 10/10/80.

Slightly higher than the 3e-5 default, with the best decay schedule so far.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, LR=5e-5, 80% decay."""
    return _base(
        lr=5e-5,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        warmup_pct=0.10,
        stable_pct=0.10,
    )
