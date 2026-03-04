"""WSD finetuning ablation — no Mixup, drop path 0.1, WSD 10/10/80.

Combines higher drop path with the most stable decay schedule.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.1, 80% decay."""
    return _base(
        lr=3e-5,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        drop_path_rate=0.1,
        warmup_pct=0.10,
        stable_pct=0.10,
    )
