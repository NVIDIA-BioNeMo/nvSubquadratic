"""Hybrid Hyena/Attention – alternating HA pattern, block-diagonal learnable-ω₀ + FiLM.

Layout: H A H A H A H A H A H A
         └─────── 6 pairs ─────┘

Combines three ideas on top of ``hybrid_ha``:

1. **Block-diagonal MLP init + per-block ω₀ schedule** with a learnable
   per-row ω₀ scale clamped to ``[1e-2, 2]``  — see
   :func:`apply_learnable_omega_blockdiag_overrides`.
2. **Block-aligned Gaussian mask** matching the kernel's block layout.
3. **Register-pooled FiLM conditioning** on every Hyena block — see
   :func:`apply_film_overrides`.  Default 8 registers (fixed across patch
   sizes so FiLM ``cond_dim`` stays at ``HIDDEN_DIM`` regardless of
   resolution), 3 FiLM layers (modulates pos-embed sine + 2 hidden
   linears), ``identity`` init, ``film_wd=5e-3``.

``apply_lr_scale=True`` attaches ``_lr_scale = 1/(2π·ω₀_max)`` to the
SIREN first-layer weight so the per-step update size matches the standard
SIREN init.

Override ``net.patch_size`` to change resolution (default 16).
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    build_hybrid_net,
    get_base_config,
)
from examples.vit5_imagenet.vit5_hybrid._film import apply_film_overrides
from examples.vit5_imagenet.vit5_hybrid._learnable_omega import (
    apply_learnable_omega_blockdiag_overrides,
)
from experiments.callbacks.film_monitor import FiLMMonitorCallback
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.callbacks.omega_scale_monitor import OmegaScaleMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


LAYER_PATTERN = "HA" * (NUM_BLOCKS // 2)


def get_config() -> ExperimentConfig:
    """Build the alternating HA hybrid config with block-diagonal learnable-ω₀ + FiLM kernels."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "default"
    config.net = build_hybrid_net(layer_pattern=LAYER_PATTERN, patch_size=PATCH_SIZE)
    apply_learnable_omega_blockdiag_overrides(config)
    apply_film_overrides(config)
    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))
    config.callbacks.append(LazyConfig(OmegaScaleMonitorCallback)(log_every_n_steps=50))
    config.callbacks.append(
        LazyConfig(FiLMMonitorCallback)(
            log_every_n_steps=50,
            num_film_layers=3,
            film_on_pos_embed=True,
            film_after_pos_embed=True,
        )
    )
    config.wandb.job_group = "vit5_hybrid_learnable_omega_blockdiag_film"
    return config
