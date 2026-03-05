"""WSD finetuning ablation — no Mixup, drop path 0.15, WSD 10/0/90.

Drop path 0.15 reached 81.99% (ep6). Combining with heavy decay.
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with no Mixup, drop path 0.15, 90% decay."""
    return _base(
        lr=3e-5,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        drop_path_rate=0.15,
        warmup_pct=0.10,
        stable_pct=0.0,
    )
