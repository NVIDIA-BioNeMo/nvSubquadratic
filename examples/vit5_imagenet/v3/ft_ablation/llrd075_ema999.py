"""FiLM finetuning — LLRD 0.75 + aggressive EMA decay 0.999, 10 epochs.

Very fast EMA (~1000 step lookback). Tests whether the standard 0.99996
is too slow for a 10-epoch finetuning window, causing the EMA to average
too much of the early (pre-adapted) weights.
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
        ema_decay=0.999,
    )
