# TODO: Add license header here

"""TinyImageNet Classification - Hyena with Patchification (ViT-B scale) - Exponential Mask Ablation.

Model Size: ViT-B
- Hidden dim: 768
- Num blocks: 12
- Patchification: patch_size=4 (64/4 = 16x16 = 256 tokens)

Phase 4.3: Replaces the Gaussian modulation mask with ExponentialModulationND.
"""

from examples.imagenet_classification.vit_b_benchmark_tiny_imagenet.hyena_patchify import get_config as get_base_config
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.masks_nd import ExponentialModulationND


def get_config():
    """Return the TinyImageNet classification configuration with Hyena + exponential mask."""
    config = get_base_config()

    # Update WandB config
    config.wandb.job_group = "tinyimagenet_mask_ablation"

    # Override the mask in each block to ExponentialModulationND
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.mask_cfg = LazyConfig(ExponentialModulationND)(
        data_dim="${net.data_dim}",
        num_channels="${net.hidden_dim}",
    )

    return config
