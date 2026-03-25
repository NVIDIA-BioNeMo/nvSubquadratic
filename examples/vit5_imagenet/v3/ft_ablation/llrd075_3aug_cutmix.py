"""FiLM finetuning — LLRD 0.75 + three-augment + CutMix, 10 epochs.

Combines the best data strategies: three-augment (from pretrain pipeline),
CutMix (spatial regularizer), and LLRD (feature protection). No Mixup to
avoid over-blending.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        epochs=10,
        layer_decay=0.75,
        use_three_augment=True,
        cutmix=1.0,
    )
