"""FiLM finetuning — LLRD 0.75 + Random Erasing 0.25, 10 epochs.

Random erasing acts as a localized dropout on the input, encouraging the
model to use broader context. Not used in pretraining.
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
        random_erasing_prob=0.25,
    )
