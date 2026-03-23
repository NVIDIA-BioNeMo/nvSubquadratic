"""FiLM finetuning — lr=1e-4, dp=0.2, wd=0.1.

Higher learning rate with compensating regularization (more drop path
and weight decay) to allow larger updates without overfitting.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.1, drop_path_rate=0.2)
