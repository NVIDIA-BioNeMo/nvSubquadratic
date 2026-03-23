"""FiLM finetuning — winning recipe + layer-wise LR decay 0.75, 10 epochs.

LLRD scales learning rate per layer: head gets full LR, embedding layer gets
lr * 0.75^13 ≈ 0.024x. This protects lower-level features while allowing
the head and upper layers to adapt more freely.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.3, drop_path_rate=0.2, film_wd=True, epochs=10, layer_decay=0.75)
