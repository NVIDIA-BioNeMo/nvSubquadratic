# TODO: Add license header here

"""1D Simple Copy — Hyena XS with Patchify (v2, causal).

Hidden dim: 256, 4 blocks, patch_size=64 (4096 → 64 tokens).

Patchification compresses the 4096-element 1D canvas to 64 tokens via Conv1d
with kernel_size=stride=64.  L_cache=64 so the SIREN kernel covers the full
patchified sequence.

Patch-size CLI override
-----------------------
Only ``net.in_proj_cfg.patch_size=P`` is needed; stride and out_proj
patch/stride are derived via interpolators.
"""

import examples.spatial_recall_v2.mixer_defaults as mixer_defaults
from examples.spatial_recall_v2.simple_copy_1d._base import base_experiment_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.patchify import Patchify, Unpatchify


HIDDEN_DIM = 256
PATCH_SIZE = 64  # 4096 / 64 = 64 tokens


def get_config() -> ExperimentConfig:
    """Build and return the experiment configuration."""
    config = base_experiment_config(hidden_dim=HIDDEN_DIM)

    config.compile_compatible_fftconv = True
    config.optimizer.lr = 5e-4
    config.train.grad_clip = 1.0

    # ── Patchify / Unpatchify projections ─────────────────────────────
    config.net.in_proj_cfg = LazyConfig(Patchify)(
        in_features="${net.in_channels}",
        out_features="${net.hidden_dim}",
        data_dim="${net.data_dim}",
        patch_size=PATCH_SIZE,
        stride="${net.in_proj_cfg.patch_size}",
    )
    config.net.out_proj_cfg = LazyConfig(Unpatchify)(
        in_features="${net.hidden_dim}",
        out_features="${net.out_channels}",
        data_dim="${net.data_dim}",
        patch_size="${net.in_proj_cfg.patch_size}",
        stride="${net.in_proj_cfg.patch_size}",
    )

    # ── Mixer ─────────────────────────────────────────────────────────
    assert config.net.block_cfg.sequence_mixer_cfg == PLACEHOLDER
    config.net.block_cfg.sequence_mixer_cfg = mixer_defaults.get_hyena_mixer_cfg(
        short_conv_cfg=mixer_defaults.short_conv_cfg(data_dim=1),
        is_causal=True,
        fft_backend="torch_fft",
        L_cache="${eval:'${dataset.canvas_size} * ${dataset.canvas_size} // ${net.in_proj_cfg.patch_size}'}",
    )

    return config
