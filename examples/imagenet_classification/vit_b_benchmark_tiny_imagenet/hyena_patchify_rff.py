# TODO: Add license header here

"""TinyImageNet Classification - Hyena with Patchification (ViT-B scale) - RFF Kernel Ablation.

Model Size: ViT-B
- Hidden dim: 768
- Num blocks: 12
- Patchification: patch_size=4 (64/4 = 16x16 = 256 tokens)

This config uses Hyena with a Random Fourier Feature (RFF) kernel instead of SIREN.
"""

import torch

from examples.imagenet_classification.vit_b_benchmark_tiny_imagenet.hyena_patchify import get_config as get_base_config
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.kernels_nd import RandomFourierKernelND


def get_config():
    """Return the TinyImageNet classification configuration with Hyena + RFF kernel."""
    config = get_base_config()

    # Update WandB config
    config.wandb.job_group = "tinyimagenet_kernel_ablation"

    # Override the kernel in each block to use RFF
    # In the base config, it's at net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg
    config.net.block_cfg.sequence_mixer_cfg.mixer_cfg.global_conv_cfg.kernel_cfg = LazyConfig(RandomFourierKernelND)(
        data_dim="${net.data_dim}",
        out_dim="${net.hidden_dim}",
        mlp_hidden_dim=64,  # Matches base KERNEL_MLP_HIDDEN_DIM
        num_layers=3,  # Matches base KERNEL_NUM_LAYERS
        embedding_dim=64,  # Matches base KERNEL_EMBEDDING_DIM
        omega_0=30.0,  # Matches base KERNEL_OMEGA_0
        L_cache=16,  # Matches base L_CACHE
        use_bias=True,
        nonlinear_cfg=LazyConfig(torch.nn.ReLU)(),
    )

    return config
