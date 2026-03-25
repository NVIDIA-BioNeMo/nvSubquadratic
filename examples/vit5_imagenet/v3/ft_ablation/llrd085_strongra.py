"""FiLM finetuning — mild LLRD 0.85 + strong RA (m14), 10 epochs.

Combines the best LLRD (0.85 from Wave 3) with the best augmentation
(strong RA from Wave 4).
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        epochs=10,
        layer_decay=0.85,
        rand_augment="rand-m14-mstd0.5-inc1",
    )
