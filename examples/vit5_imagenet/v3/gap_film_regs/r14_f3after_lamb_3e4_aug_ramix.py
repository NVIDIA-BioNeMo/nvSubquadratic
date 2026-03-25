"""14 regs, film3_after, LAMB lr=3e-4 — RandAugment + Mixup/CutMix.

RandAugment m9 + Mixup(0.8) + CutMix(1.0), smoothing=0.1, dp=0.15.
"""

from examples.vit5_imagenet.v3.gap_film_regs._base import get_config as _base


def get_config():  # noqa: D103
    return _base(
        num_registers=14,
        num_film_layers=3,
        film_after_pos_embed=True,
        lr=3e-4,
        wd=0.05,
        drop_path_rate=0.15,
        smoothing=0.1,
        mixup=0.8,
        cutmix=1.0,
        use_three_augment=False,
        rand_augment="rand-m9-mstd0.5-inc1",
        optimizer_type="lamb",
    )
