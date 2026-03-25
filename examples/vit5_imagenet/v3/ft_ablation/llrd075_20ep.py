"""FiLM finetuning — LLRD 0.75, 20 epochs (longer with decay protection).

Since LLRD sustains 0.816 at epoch 8 (vs 0.813 without LLRD), extending to
20 epochs tests whether the model can keep improving or simply plateaus.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.3, drop_path_rate=0.2, film_wd=True, epochs=20, layer_decay=0.75)
