"""FiLM finetuning — dp=0.10 + relaxed FiLM WD (0.001).

Combines lower structural regularization with very relaxed FiLM weight decay,
giving the FiLM generators maximum freedom to develop input-dependent kernels.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(drop_path_rate=0.10, film_wd=0.001)
