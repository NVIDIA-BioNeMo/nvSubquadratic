"""14 registers, film2, Recipe B (conservative)."""

from examples.vit5_imagenet.v3.gap_film_regs._base import get_config as _base


def get_config():  # noqa: D103
    return _base(
        num_registers=14,
        num_film_layers=2,
        film_after_pos_embed=False,
        lr=1e-5,
        wd=0.1,
        drop_path_rate=0.05,
    )
