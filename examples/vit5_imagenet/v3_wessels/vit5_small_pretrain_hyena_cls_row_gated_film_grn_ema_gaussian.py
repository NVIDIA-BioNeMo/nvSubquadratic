"""ViT-5-Small + Hyena ImageNet-1k — CLS-row, FiLM + GRN + Gaussian mask (no RoPE).

Same as the FiLM + GRN config but adds a learnable Gaussian decay mask on the
continuous convolutional kernel. The mask multiplies the SIREN-generated kernel
with a per-channel Gaussian envelope, encouraging locality while still allowing
the network to learn long-range interactions via the learned std parameters.

- CKConvND (depthwise) with global channel mixing.
- FiLM-conditioned SIREN kernels (input-dependent via register pooling).
- GaussianModulationND mask on the convolutional kernel.
- GRN after mixer output for inter-channel feature competition.
- Dual gating: SiLU (first gate) + Sigmoid (second gate).
- CLS-row architecture: CLS + 13 registers as extra row -> 15x14 grid.
- No RoPE — isolates Gaussian mask contribution without positional encoding in the gate.
"""

from examples.vit5_imagenet.v3_wessels._base_config import HIDDEN_DIM, build_cls_row_network, build_film_cfg, build_hyena_mixer, get_base_config
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.grn import GlobalResponseNorm
from nvsubquadratic.modules.masks_nd import GaussianModulationND


def get_config() -> ExperimentConfig:
    """Return the ViT-5-Small + Hyena CLS-row + FiLM + GRN + Gaussian mask (no RoPE) config."""
    config = get_base_config()

    film_cfg = build_film_cfg()
    mixer_cfg = build_hyena_mixer(film_cfg=film_cfg, use_rope=False)

    # Replace Identity mask with learnable Gaussian decay
    mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(GaussianModulationND)(
        data_dim=2,
        num_channels=HIDDEN_DIM,
        min_std=0.025,
        max_std=1.25,
        init_std_low=0.05,
        init_std_high=1.0,
        parametrization="direct",
    )

    config.net, trainer_overrides = build_cls_row_network(
        mixer_cfg,
        grn_cfg=LazyConfig(GlobalResponseNorm)(dim=HIDDEN_DIM),
    )
    for k, v in trainer_overrides.items():
        setattr(config.trainer, k, v)

    return config
