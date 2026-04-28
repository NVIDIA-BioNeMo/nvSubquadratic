# TODO: Add license header here

"""1D Color Conditioning — Attention XS (v2, causal).

Hidden dim: 256, 4 blocks, num_heads=8, head_dim=32.
4 items on 1D canvas with coloured boundary markers, output digit in matching colour.
Causal by default; pass is_causal=False for bidirectional variant.
Compile: max-autotune.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.color_conditioning_1d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER


HIDDEN_DIM = 256
NUM_HEADS = 8  # head_dim = 32


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.compile_mode = "max-autotune"

    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_attention_mixer_cfg(
        num_heads=NUM_HEADS,
        apply_qk_norm=True,
        use_rope=True,
        is_causal=True,
        rope_spatial_dims=("${eval:'${dataset.canvas_size} * ${dataset.canvas_size}'}",),
    )

    return config
