"""WSD finetuning ablation — no Mixup, drop path 0.1, LR=5e-5, WSD 10/0/90.

dp01_lr5e5 reached 82.04% and dp01_decay90 reached 82.03%.
Combining the two best dp01 variants.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, dp 0.1, LR=5e-5, 90% decay."""
    return _base(
        lr=5e-5,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        drop_path_rate=0.1,
        warmup_pct=0.10,
        stable_pct=0.0,
    )
