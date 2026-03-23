"""FiLM finetuning — winning recipe + aggressive LLRD 0.65, 10 epochs.

More aggressive decay than 0.75: embedding layer gets lr * 0.65^13 ≈ 0.009x.
Tests whether stronger feature protection helps.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.3, drop_path_rate=0.2, film_wd=True, epochs=10, layer_decay=0.65)
