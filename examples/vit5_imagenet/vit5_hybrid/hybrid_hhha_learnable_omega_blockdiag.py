"""Hybrid Hyena/Attention – 3:1 ratio (HHHA), block-diagonal learnable-ω₀ kernel.

Layout: H H H A H H H A H H H A
         └─────── 3 groups ─────┘

Same architecture as ``hybrid_hhha`` but every Hyena block uses
``BlockDiagonalLearnableOmegaSIRENKernelND`` paired with
``BlockAlignedGaussianModulationND``.  Combines:

1. **Block-diagonal MLP init + per-block ω₀ schedule** — same defaults
   as ``hybrid_hhha_blockdiag`` (``num_blocks=8``, linear ω₀ schedule in
   ``[1, 12]``, ``off_block_scale=0.1``).
2. **Learnable per-row ω₀ scale** — the ``2π·ω₀`` factor is pulled out
   of the first-layer weight init and applied at runtime, multiplied by
   a learnable per-row scale clamped to ``[1e-2, 2]`` (so each row can
   double its effective ω₀ during training while never collapsing to zero).

``apply_lr_scale=True`` attaches ``_lr_scale = 1/(2π·ω₀_max)`` to the
first-layer weight so the per-step update size matches the standard
SIREN init.

Override ``net.patch_size`` to change resolution (default 16).
"""

from examples.vit5_imagenet.v5._base import NUM_BLOCKS, PATCH_SIZE
from examples.vit5_imagenet.vit5_hybrid._base_config import (
    build_hybrid_net,
    get_base_config,
)
from examples.vit5_imagenet.vit5_hybrid._learnable_omega import (
    apply_learnable_omega_blockdiag_overrides,
)
from experiments.callbacks.mask_monitor import MaskMonitorCallback
from experiments.callbacks.omega_scale_monitor import OmegaScaleMonitorCallback
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


LAYER_PATTERN = "HHHA" * (NUM_BLOCKS // 4)


def get_config() -> ExperimentConfig:
    """Build the HHHA hybrid config with block-diagonal learnable-ω₀ kernels."""
    config = get_base_config()
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.net = build_hybrid_net(layer_pattern=LAYER_PATTERN, patch_size=PATCH_SIZE)
    apply_learnable_omega_blockdiag_overrides(config)
    config.callbacks.append(LazyConfig(MaskMonitorCallback)(log_every_n_steps=50))
    config.callbacks.append(LazyConfig(OmegaScaleMonitorCallback)(log_every_n_steps=50))
    config.wandb.job_group = "vit5_hybrid_learnable_omega_blockdiag"
    return config
