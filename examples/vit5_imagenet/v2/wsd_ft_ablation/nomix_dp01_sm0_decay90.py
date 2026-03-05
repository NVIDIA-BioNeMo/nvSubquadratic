"""WSD finetuning ablation — no Mixup, drop path 0.1, no smoothing, WSD 10/0/90.

Triple combo of the three best individual modifications:
drop path 0.1 (82.05%), no smoothing (81.98%), heavy decay (81.99%).
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, dp 0.1, no smoothing, 90% decay."""
    return _base(
        lr=3e-5,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        drop_path_rate=0.1,
        smoothing=0.0,
        warmup_pct=0.10,
        stable_pct=0.0,
    )
