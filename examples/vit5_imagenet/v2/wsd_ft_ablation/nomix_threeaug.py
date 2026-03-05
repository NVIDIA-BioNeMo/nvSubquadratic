"""WSD finetuning ablation — no-mixup three-augment (pretrain recipe, no mixup)."""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config for no-mixup with three-augment, no label smoothing."""
    return _base(
        lr=3e-5,
        wd=0.05,
        mixup=0.0,
        cutmix=0.0,
        use_three_augment=True,
        rand_augment=None,
        smoothing=0.0,
    )
