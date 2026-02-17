# TODO: Add license header here

"""ImageNet Classification - Hyena with Patchification (ViT-B scale, Split Weight Decay).

Variation: Split Weight Decay
- SIREN Kernel (Hidden + Output): WD = 0.0
- Rest of Model: WD = 0.05

This is achieved by setting `no_weight_decay_on_output=True` in the SIREN kernel config.
"""

from experiments.default_cfg import ExperimentConfig
from examples.imagenet_classification.vit_b_benchmark_tiny_imagenet.imagenette_hyena_patchify import get_config as get_base_config

def get_config() -> ExperimentConfig:
    """Return the Imagenette classification configuration with Split Weight Decay."""
    config = get_base_config()
    
    # Enable no_weight_decay_on_output for SIREN kernel
    # Structure: net -> block_cfg -> sequence_mixer_cfg -> mixer_cfg (Hyena) -> global_conv_cfg (CKConvND) -> kernel_cfg (SIRENKernelND)
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg.no_weight_decay_on_output = True
    
    config.wandb.job_group = "imagenette_hyena_ablation_split_wd"

    # Disable W&B checkpoint uploads to prevent memory leak during long runs
    config.trainer.wandb_checkpoint_upload = False

    return config
