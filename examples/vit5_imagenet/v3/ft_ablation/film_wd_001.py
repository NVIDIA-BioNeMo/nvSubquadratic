"""FiLM finetuning — relaxed FiLM weight decay (0.01).

Reduces WD on FiLM generator params to allow stronger input-dependent
conditioning while keeping global WD at 0.05.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(film_wd=0.01)
