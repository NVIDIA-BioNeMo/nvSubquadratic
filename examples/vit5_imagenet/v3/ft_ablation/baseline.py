"""FiLM finetuning — baseline (best v2 recipe adapted to cosine).

dp=0.15, lr=3e-5, wd=0.05, cosine 25% warmup, no mixup, smoothing=0.1.
FiLM WD = global (0.05).
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base()
