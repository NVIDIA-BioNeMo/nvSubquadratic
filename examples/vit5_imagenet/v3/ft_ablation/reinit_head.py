"""FiLM finetuning — re-initialized classification head only (no LLRD), 10 epochs.

Control experiment: re-init head without LLRD. If the pretrained head is
already near-optimal, this will hurt. If the head is a bottleneck, this
should help independently of LLRD.
"""

from examples.vit5_imagenet.v3.ft_ablation._base import get_config as _base


def get_config():
    return _base(lr=1e-4, wd=0.3, drop_path_rate=0.2, film_wd=True, epochs=10, reinit_head=True)
