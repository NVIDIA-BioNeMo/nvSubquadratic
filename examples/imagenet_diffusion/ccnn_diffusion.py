# TODO: Add license header here

"""Config file for ImageNet diffusion using the shared ResNet backbone."""

import os

import torch

from experiments.datamodules.imagenet import ImageNetDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers import DiffusionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.condition_mixer import QKVConditionMixer
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.kernels_nd import RandomFourierKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


PLACEHOLDER = None

data_dim = 2

# Model parameters
batch_size = 16
hidden_dim = 256
num_blocks = 12
dropout_in_rate = 0.0
dropout_rate = 0.1
grid_type = "double"

# Training parameters
training_iterations = 800_000
warmup_iterations_percentage = 0.02
grad_clip = 1.0
weight_decay = 1e-3
learning_rate = 2e-4

# Diffusion parameters
num_train_timesteps = 1_000
beta_start = 1e-4
beta_end = 2e-2
beta_schedule = "linear"
num_inference_steps = 50
num_samples = 8
ema_decay = 0.999
ema_warmup_steps = 1_000
ema_update_every = 1

image_size = 256
final_image_size = 32


def get_config() -> ExperimentConfig:
    """Return the ImageNet diffusion configuration."""

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42

    hf_token = os.environ.get("HF_TOKEN")

    cpu_count = os.cpu_count() or 8
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        num_workers = max(4, cpu_count // torch.cuda.device_count())
    else:
        num_workers = max(4, cpu_count // 2)

    config.dataset = LazyConfig(ImageNetDataModule)(
        data_dir="/media/davidknigge/hard-disk2/huggingface/imagenet",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() and config.device == "cuda",
        seed=config.seed,
        image_size=image_size,
        final_image_size=final_image_size,
        center_crop=True,
        drop_labels=True,
        hf_dataset_name="imagenet-1k",
        hf_dataset_config=None,
        hf_auth_token=hf_token,
    )

    config.net = LazyConfig(ResidualNetwork)(
        in_channels=PLACEHOLDER,
        out_channels=PLACEHOLDER,
        num_blocks=num_blocks,
        hidden_dim=hidden_dim,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=PLACEHOLDER),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=PLACEHOLDER),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim=data_dim,
                        hidden_dim="${net.hidden_dim}",
                        kernel_cfg=LazyConfig(RandomFourierKernelND)(
                            data_dim=data_dim,
                            out_dim="${net.hidden_dim}",
                            mlp_hidden_dim=64,
                            num_layers=3,
                            embedding_dim=64,
                            omega_0=100.0,
                            L_cache=32,
                            use_bias=True,
                            nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                        ),
                        mask_cfg=LazyConfig(GaussianModulationND)(
                            data_dim=data_dim,
                            num_channels="${net.hidden_dim}",
                            min_std=0.02,
                            max_std=1.5,
                            init_std_low=0.05,
                            init_std_high=1.2,
                            parametrization="direct",
                        ),
                        grid_type=grid_type,
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Conv2d)(
                        in_channels="3 * ${net.hidden_dim}",
                        out_channels="3 * ${net.hidden_dim}",
                        kernel_size=3,
                        groups="3 * ${net.hidden_dim}",
                        padding=1,
                        bias=False,
                    ),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.Identity)(),
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.GroupNorm)(
                        num_groups=1,
                        num_channels="${net.hidden_dim}",
                    ),
                    apply_qk_norm=True,
                    use_rope=True,
                    rope_base=10000.0,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=num_blocks),
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            condition_mixer_cfg=LazyConfig(QKVConditionMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(torch.nn.MultiheadAttention)(
                    embed_dim="${net.hidden_dim}",
                    num_heads=4,
                    batch_first=True,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=num_blocks),
            ),
            condition_mixer_norm_cfg="${net.norm_cfg}",
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=2.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=dropout_rate),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=num_blocks),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=dropout_rate),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=dropout_in_rate),
        condition_in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features=PLACEHOLDER, out_features=PLACEHOLDER),
    )

    config.lightning_wrapper_class = LazyConfig(DiffusionWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=training_iterations,
        grad_clip=grad_clip,
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=warmup_iterations_percentage,
        total_iterations="${train.iterations}",
    )

    # Propagate diffusion-specific Lightning wrapper keyword arguments.
    config.lightningwrapper_kwargs = {
        "diffusion": {
            "num_train_timesteps": num_train_timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "beta_schedule": beta_schedule,
            "time_embed_dim": hidden_dim,
        },
        "diffusion_sampling": {
            "num_inference_steps": num_inference_steps,
            "num_samples": num_samples,
            "log_samples": True,
        },
        "diffusion_ema": {
            "enabled": True,
            "decay": ema_decay,
            "warmup_steps": ema_warmup_steps,
            "update_every": ema_update_every,
        },
    }

    config.wandb = WandbConfig(job_group="imagenet-diffusion")

    return config
