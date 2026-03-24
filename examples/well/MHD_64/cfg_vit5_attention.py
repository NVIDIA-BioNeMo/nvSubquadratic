"""MHD_64 ViT5-style attention config."""

import os

import torch

from experiments.datamodules.pde.well import WellDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.well_lightning_wrapper import WELLRegressionWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.patchify import Patchify, Unpatchify
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.modules.vit5_residual_block import ViT5ResidualBlock
from nvsubquadratic.networks.vit5_general_purpose import ViT5GeneralPurposeNet
from nvsubquadratic.utils.init import trunc_normal_init_factory


PLACEHOLDER = None

DATA_DIM = 3
SPATIAL_SIZE = 64
WELL_BASE_PATH = os.environ.get("WELL_DATA_PATH", "/gpfs/scratch1/shared/dwessels2/data/the_well/datasets")
WELL_DATASET_NAME = "MHD_64"

N_STEPS_INPUT = 4
N_STEPS_OUTPUT = 1
MAX_ROLLOUT_STEPS = 1

BATCH_SIZE = int(os.environ.get("MHD_VIT5_ATTN_BATCH_SIZE", 1))
HIDDEN_DIM = int(os.environ.get("MHD_VIT5_ATTN_HIDDEN_DIM", 384))
NUM_BLOCKS = int(os.environ.get("MHD_VIT5_ATTN_DEPTH", 12))
PATCH_SIZE = int(os.environ.get("MHD_VIT5_ATTN_PATCH_SIZE", 8))
NUM_HEADS = int(os.environ.get("MHD_VIT5_ATTN_NUM_HEADS", 6))
NUM_REGISTERS = 14
DROPOUT_RATE = 0.0
DROP_PATH_RATE = 0.05
LAYER_SCALE_INIT = 1e-4
MLP_RATIO = 4.0

TRAINING_ITERATIONS = 260_000
WARMUP_ITERATIONS_PERCENTAGE = 0.1
NUM_WORKERS = 8
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-5
LEARNING_RATE = 1e-4

INIT_FN_FACTORY = trunc_normal_init_factory(std=0.02)


def get_config() -> ExperimentConfig:
    """Return the MHD_64 ViT5-style attention config."""
    config = ExperimentConfig()

    config.debug = False
    config.compile = True
    config.compile_mode = "max-autotune-no-cudagraphs"

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

    mixer_cfg = LazyConfig(QKVSequenceMixer)(
        hidden_dim=HIDDEN_DIM,
        mixer_cfg=LazyConfig(Attention)(
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            apply_qk_norm=True,
            use_rope=True,
            is_causal=False,
            attn_dropout=0.0,
            rope_base=10000.0,
        ),
        qkv_bias=False,
        out_proj_bias=False,
        init_method_in=INIT_FN_FACTORY,
        init_method_out=INIT_FN_FACTORY,
    )

    block_cfg = LazyConfig(ViT5ResidualBlock)(
        sequence_mixer_cfg=mixer_cfg,
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
        job_group="MHD_64_vit5_attention",
    )

    return config
