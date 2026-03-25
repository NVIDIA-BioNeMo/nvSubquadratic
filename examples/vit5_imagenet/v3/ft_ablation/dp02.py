"""FiLM finetuning — higher drop path (0.20).

dp=0.20 reached 82.02% in v2. Tests whether FiLM benefits from
stronger structural regularization than the attention model.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(drop_path_rate=0.20)
