"""FiLM finetuning — LLRD 0.75 + CutMix only (no Mixup), 10 epochs.

CutMix preserves more spatial structure than Mixup — better suited for
this spatial model (Hyena-based). Uses cutmix=1.0 alpha.
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
        cutmix=1.0,
    )
