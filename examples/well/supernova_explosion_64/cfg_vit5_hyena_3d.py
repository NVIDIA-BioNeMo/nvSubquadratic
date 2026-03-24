"""Supernova ViT5-style 3D Hyena config with registers.

Same as cfg_vit5_hyena.py but uses ViT5HyenaAdapterND to reshape the flattened
token sequence back to a 3D patch grid before applying Hyena. This preserves
3D spatial locality in the short conv (Conv3d) and SIREN kernel (data_dim=3),
rather than operating on a flattened 1D sequence.
"""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_hyena_adapter import ViT5HyenaAdapterND
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_general_purpose import ViT5GeneralPurposeNet
from nvsubquadratic.utils.init import trunc_normal_init_factory
from nvsubquadratic.utils.qk_norm import L2Norm


PLACEHOLDER = None

DATA_DIM = 3
SPATIAL_SIZE = 64
WELL_BASE_PATH = os.environ.get("WELL_DATA_PATH", "/gpfs/scratch1/shared/dwessels2/data/the_well/datasets")
WELL_DATASET_NAME = "supernova_explosion_64"

N_STEPS_INPUT = 4
N_STEPS_OUTPUT = 1
MAX_ROLLOUT_STEPS = 1

BATCH_SIZE = int(os.environ.get("SUPERNOVA_VIT5_HYENA3D_BATCH_SIZE", 2))
HIDDEN_DIM = int(os.environ.get("SUPERNOVA_VIT5_HYENA3D_HIDDEN_DIM", 384))
NUM_BLOCKS = int(os.environ.get("SUPERNOVA_VIT5_HYENA3D_DEPTH", 12))
NUM_REGISTERS = 14
DROPOUT_RATE = 0.0
DROP_PATH_RATE = 0.05
LAYER_SCALE_INIT = 1e-4
MLP_RATIO = 4.0
PATCH_SIZE = int(os.environ.get("SUPERNOVA_VIT5_HYENA3D_PATCH_SIZE", 8))

TRAINING_ITERATIONS = 260_000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
NUM_WORKERS = 8
GRAD_CLIP = 1.0

WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-3

KERNEL_MLP_HIDDEN_DIM = 32
KERNEL_NUM_LAYERS = 3
KERNEL_EMBEDDING_DIM = 32
KERNEL_OMEGA_0 = 10.0
KERNEL_HIDDEN_OMEGA_0 = 1.0

INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


def get_config() -> ExperimentConfig:
    """Return the supernova ViT5-style 3D Hyena config."""
    config = ExperimentConfig()

    config.debug = False
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"
    config.compile_compatible_fftconv = True

    config.dataset = LazyConfig(WellDataModule)(
        well_base_path=WELL_BASE_PATH,
        well_dataset_name=WELL_DATASET_NAME,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_normalization=True,
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        min_dt_stride=1,
        max_dt_stride=1,
        local_staging_dir=None,
    )

    patch_grid_size = SPATIAL_SIZE // PATCH_SIZE  # 8 per dimension (with patch_size=8)

    # 3D Hyena mixer: Conv3d short conv + data_dim=3 SIREN kernel
    hyena_3d_mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Hyena)(
            global_conv_cfg=LazyConfig(CKConvND)(
                data_dim=DATA_DIM,
                hidden_dim=HIDDEN_DIM,
                fft_padding="zero",
                kernel_cfg=LazyConfig(SIRENKernelND)(
                    data_dim=DATA_DIM,
                    out_dim=HIDDEN_DIM,
                    mlp_hidden_dim=KERNEL_MLP_HIDDEN_DIM,
                    num_layers=KERNEL_NUM_LAYERS,
                    embedding_dim=KERNEL_EMBEDDING_DIM,
                    omega_0=KERNEL_OMEGA_0,
                    L_cache=patch_grid_size,
                    use_bias=True,
                    hidden_omega_0=KERNEL_HIDDEN_OMEGA_0,
                ),
                mask_cfg=LazyConfig(torch.nn.Identity)(),
                grid_type="single",
            ),
            short_conv_cfg=LazyConfig(torch.nn.Conv3d)(
                in_channels=3 * HIDDEN_DIM,
                out_channels=3 * HIDDEN_DIM,
                kernel_size=3,
                groups=3 * HIDDEN_DIM,
                padding=1,
                bias=False,
            ),
            gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
            gate_nonlinear_2_cfg=LazyConfig(torch.nn.Sigmoid)(),
            pixelhyena_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
            qk_norm_cfg=LazyConfig(L2Norm)(),
            use_rope=False,
            output_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )

    # Wrap with ViT5HyenaAdapterND: strips registers, reshapes to 3D grid, applies 3D Hyena, re-prepends
    adapter_cfg = LazyConfig(ViT5HyenaAdapterND)(
        inner_mixer_cfg=hyena_3d_mixer_cfg,
        grid_shape=(patch_grid_size,) * DATA_DIM,
        num_prefix_tokens=NUM_REGISTERS,
    )

    block_cfg = LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=adapter_cfg,
        sequence_mixer_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        mlp_cfg=LazyConfig(MLP)(
            dim=HIDDEN_DIM,
            activation="gelu",
            expansion_factor=MLP_RATIO,
            bias=False,
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
            init_method_in=INIT_FN_FACTORY,
            init_method_out=INIT_FN_FACTORY,
        ),
        mlp_norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        hidden_dim=HIDDEN_DIM,
        layer_scale_init=LAYER_SCALE_INIT,
        drop_path_rate=DROP_PATH_RATE,
    )

    config.net = LazyConfig(ViT5GeneralPurposeNet)(
        in_channels=PLACEHOLDER,
        out_channels=PLACEHOLDER,
        hidden_dim=HIDDEN_DIM,
        num_blocks=NUM_BLOCKS,
        data_dim=DATA_DIM,
        patch_size=PATCH_SIZE,
        input_size=SPATIAL_SIZE,
        num_registers=NUM_REGISTERS,
        in_proj_cfg=LazyConfig(Patchify)(
            in_features=PLACEHOLDER,
            out_features=PLACEHOLDER,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        out_proj_cfg=LazyConfig(Unpatchify)(
            in_features=PLACEHOLDER,
            out_features=PLACEHOLDER,
            data_dim=DATA_DIM,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        ),
        block_cfg=block_cfg,
        norm_cfg=LazyConfig(RMSNorm)(dim=HIDDEN_DIM, eps=1e-6),
        dropout_rate=DROPOUT_RATE,
        use_cls_token=False,
        prepend_registers=True,
    )

    config.lightning_wrapper_class = LazyConfig(WELLRegressionWrapper)(
        metadata=PLACEHOLDER,
        n_steps_input=N_STEPS_INPUT,
        n_steps_output=N_STEPS_OUTPUT,
        max_rollout_steps=MAX_ROLLOUT_STEPS,
        metric="MSE",
    )

    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        precision="bf16-mixed",
    )

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    config.wandb = WandbConfig(
        entity="implicit-long-convs",
        project="nvsubquadratic",
        job_group="supernova_explosion_64_vit5_hyena_3d",
    )

    return config
