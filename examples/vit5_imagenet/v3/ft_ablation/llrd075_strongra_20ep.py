"""FiLM finetuning — LLRD 0.75 + strong RA (m14) + 20 epochs.

Combines the best data augmentation (m14) with extended training under
LLRD protection. Tests whether stronger augmentation prevents the
overfitting that plagued 20-epoch runs in Wave 2.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        epochs=20,
        layer_decay=0.75,
        rand_augment="rand-m14-mstd0.5-inc1",
    )
