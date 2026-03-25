"""14 regs, film3_after, LAMB lr=1e-4 — lower LR + Mixup/CutMix.

Slower learning (1e-4 vs 3e-4) with three-augment and Mixup to reduce overshoot.
"""

from examples.vit5_imagenet.v3.gap_film_regs._base import get_config as _base


def get_config():  # noqa: D103
    return _base(
        num_registers=14,
        num_film_layers=3,
        film_after_pos_embed=True,
        lr=1e-4,
        wd=0.05,
        drop_path_rate=0.15,
        smoothing=0.1,
        mixup=0.8,
        cutmix=1.0,
        use_three_augment=True,
        rand_augment="",
        optimizer_type="lamb",
    )
