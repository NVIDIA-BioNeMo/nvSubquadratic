"""FiLM finetuning — LLRD 0.75 + light Mixup/CutMix, 10 epochs.

Lighter mix than pretrain (mixup=0.3 + cutmix=0.5 vs pretrain's 0.8/1.0).
Should regularize without the convergence penalty of full pretrain-level aug.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(
        lr=1e-4,
        wd=0.3,
        drop_path_rate=0.2,
        film_wd=True,
        epochs=10,
        layer_decay=0.75,
        mixup=0.3,
        cutmix=0.5,
    )
