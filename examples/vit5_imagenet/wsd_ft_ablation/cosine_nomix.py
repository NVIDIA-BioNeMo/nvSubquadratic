"""WSD finetuning ablation — cosine schedule, no Mixup/CutMix, LR=1e-5."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with cosine schedule and no Mixup/CutMix."""
    return _base(
        lr=1e-5,
        wd=0.1,
        mixup=0.0,
        cutmix=0.0,
        scheduler_name="cosine",
        warmup_pct=0.25,
        stable_pct=0.0,
    )
