"""WSD finetuning ablation — no-mixup cosine schedule."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config for no-mixup cosine schedule, 25% warmup."""
    return _base(
        lr=3e-5,
        wd=0.05,
        scheduler_name="cosine",
        stable_pct=0.0,
        warmup_pct=0.25,
        mixup=0.0,
        cutmix=0.0,
    )
