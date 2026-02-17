# TODO: Add license header here

"""ImageNet Classification - Hyena with Patchification (ViT-B scale, High Frequency).

Variation: High Frequency SIREN
- Omega_0: 60.0 (Base is 30.0)

Hypothesis: Vision signals require higher frequency filters than NLP.
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
    """Return the Imagenette classification configuration with High Frequency SIREN."""
    config = get_base_config()
    
    # Set omega_0 to 60.0
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.omega_0 = 60.0
    
    config.wandb.job_group = "imagenette_hyena_ablation_hi_freq"

    # Disable W&B checkpoint uploads to prevent memory leak during long runs
    config.trainer.wandb_checkpoint_upload = False

    return config
