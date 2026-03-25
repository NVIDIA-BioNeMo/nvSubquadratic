"""FiLM finetuning — lr=1e-4, dp=0.3, wd=0.3, free FiLM.

Variant of the winning lr1e4_wd03 recipe with higher drop path (0.3
vs 0.2) for extra structural regularization. Tests whether stronger
dp can further delay overfitting while wd=0.3 + free FiLM does the heavy
lifting.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.3, drop_path_rate=0.3, film_wd=True)
