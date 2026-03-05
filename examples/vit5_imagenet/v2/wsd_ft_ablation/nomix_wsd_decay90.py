"""WSD finetuning ablation — no-mixup WSD schedule: 90% decay."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config for no-mixup WSD 10% warmup, 0% stable, 90% decay."""
    return _base(
        lr=3e-5,
        wd=0.05,
        warmup_pct=0.10,
        stable_pct=0.0,
        mixup=0.0,
        cutmix=0.0,
    )
