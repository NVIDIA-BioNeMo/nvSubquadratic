"""FiLM finetuning — LLRD 0.75 + three-augment, 10 epochs.

Combines LLRD (structural) with three-augment (data-side regularization),
the two best strategies from Waves 1-2.
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
    )
