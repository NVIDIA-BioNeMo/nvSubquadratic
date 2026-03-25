"""FiLM finetuning — mild LLRD 0.85, 10 epochs.

Lighter decay than 0.75: embedding gets lr * 0.85^13 ≈ 0.12x.
Bracket between no-LLRD (all layers at 1x) and 0.75 (embedding at 0.024x).
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.3, drop_path_rate=0.2, film_wd=True, epochs=10, layer_decay=0.85)
