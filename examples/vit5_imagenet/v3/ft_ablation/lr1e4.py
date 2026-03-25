"""FiLM finetuning — lr=1e-4 (3x baseline).

Higher learning rate with default regularization to push the model
further from the pretrained weights.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4)
