"""14 registers, film3_after, Recipe A — FiLM small_random init."""

from examples.vit5_imagenet.v3.gap_film_regs._base import get_config as _base


def get_config():  # noqa: D103
    return _base(
        num_registers=14,
        num_film_layers=3,
        film_after_pos_embed=True,
        lr=3e-5,
        wd=0.05,
        drop_path_rate=0.15,
        film_init_type="small_random",
    )
