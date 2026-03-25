"""FiLM finetuning — LLRD 0.75 + strong RA + random erasing, 10 epochs.

Double augmentation: strong RandAugment (m14) + Random Erasing (p=0.25).
Both had good val_loss individually; together they may push generalization.
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
        rand_augment="rand-m14-mstd0.5-inc1",
        random_erasing_prob=0.25,
    )
