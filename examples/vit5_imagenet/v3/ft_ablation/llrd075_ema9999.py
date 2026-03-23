"""FiLM finetuning — LLRD 0.75 + faster EMA decay 0.9999, 10 epochs.

Default EMA decay 0.99996 means the EMA model looks back ~25000 steps.
With 0.9999 it looks back ~10000 steps — might track the improving model
more closely during the short finetuning window.
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
        ema_decay=0.9999,
    )
