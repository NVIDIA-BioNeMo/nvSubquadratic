"""FiLM finetuning — lr=1e-4, dp=0.2, wd=0.3, film_wd=0.

Very strong weight decay on backbone as the primary regularizer alongside
higher LR, but FiLM generators are exempt to allow free input conditioning.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.3, drop_path_rate=0.2, film_wd=True)
