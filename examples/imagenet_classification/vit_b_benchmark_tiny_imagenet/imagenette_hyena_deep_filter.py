# TODO: Add license header here

"""ImageNet Classification - Hyena with Patchification (ViT-B scale, Deep Filter).

Variation: Deep SIREN Kernel
- Num Layers: 5 (Base is 3)

Hypothesis: Deeper MLP allows for more complex spatial priors.
"""

from experiments.default_cfg import ExperimentConfig

# Import get_config from the base file and alias it as get_base_config
try:
    from examples.imagenet_classification.vit_b_benchmark_tiny_imagenet.imagenette_hyena_patchify import get_config as get_base_config
except ImportError:
    import sys
    sys.path.append(".")
    from examples.imagenet_classification.vit_b_benchmark_tiny_imagenet.imagenette_hyena_patchify import get_config as get_base_config

def get_config() -> ExperimentConfig:
    """Return the Imagenette classification configuration with Deep SIREN Kernel."""
    config = get_base_config()
    
    # Set num_layers to 5
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.num_layers = 5
    
    config.wandb.job_group = "imagenette_hyena_ablation_deep_filter"
    
    return config
