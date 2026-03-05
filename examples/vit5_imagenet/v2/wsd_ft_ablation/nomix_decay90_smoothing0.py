"""WSD finetuning ablation — no Mixup, heavy decay, no label smoothing.

Combines the two best-performing modifications: WSD 10/0/90 schedule
and no label smoothing (to be confirmed by nomix_smoothing0).
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, 90% decay, no label smoothing."""
    return _base(
        lr=3e-5,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        smoothing=0.0,
        warmup_pct=0.10,
        stable_pct=0.0,
    )
