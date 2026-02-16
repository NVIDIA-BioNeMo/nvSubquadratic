import os
import torch
from experiments.datamodules.lra_datamodule import LRADataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.classification_wrapper import ClassificationWrapper
from nvsubquadratic.lazy_config import PLACEHOLDER, LazyConfig
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.classification_resnet import ClassificationResNet

# Dataset parameters
INPUT_CHANNELS = 1 # Byte-level 
NUM_CLASSES = 2
DATA_DIM = 1
SEQ_LENGTH = 4096

# Training parameters
BATCH_SIZE = 32
TRAINING_ITERATIONS = 50_000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

# Model parameters
NUM_HIDDEN_CHANNELS = 128 # Smaller hidden dim for longer sequence
NUM_BLOCKS = 4

def get_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False
    
    config.dataset = LazyConfig(LRADataModule)(
        task="text",
        batch_size=BATCH_SIZE,
        num_workers=4,
        max_length=SEQ_LENGTH,
    )

    config.net = LazyConfig(ClassificationResNet)(
        in_channels=INPUT_CHANNELS,
        out_channels=NUM_CLASSES,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        in_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Hyena)(
                    global_conv_cfg=LazyConfig(CKConvND)(
                        data_dim="${net.data_dim}",
                        hidden_dim="${net.hidden_dim}",
                        kernel_cfg=LazyConfig(SIRENKernelND)(
                            data_dim="${net.data_dim}",
                            out_dim="${net.hidden_dim}",
                            num_layers=3,
                            omega_0=30.0,
                            L_cache=SEQ_LENGTH,
                        ),
                        mask_cfg=LazyConfig(torch.nn.Identity)(),
                        grid_type="double",
                        fft_padding="zero",
                    ),
                    short_conv_cfg=LazyConfig(torch.nn.Identity)(),
                    gate_nonlinear_cfg=LazyConfig(torch.nn.SiLU)(),
                    pixelhyena_norm_cfg=LazyConfig(torch.nn.Identity)(),
                    apply_qk_norm=True,
                    use_rope=False,
                ),
                init_method_in=small_init,
                init_method_out=LazyConfig(partial_wang_init_fn_with_num_layers)(num_layers="${net.num_blocks}"),
            ),
            sequence_mixer_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=2.0,
            ),
            mlp_norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=0.1),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Dropout)(p=0.0),
    )

    config.lightning_wrapper_class = LazyConfig(ClassificationWrapper)()
    config.optimizer = LazyConfig(torch.optim.AdamW)(params=PLACEHOLDER, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    config.train = TrainConfig(batch_size=BATCH_SIZE, iterations=TRAINING_ITERATIONS, precision="bf16-mixed")
    config.scheduler = SchedulerConfig(name="cosine", total_iterations="${train.iterations}")
    config.wandb = WandbConfig(job_group="lra_text", project="nvsubquadratic")

    return config
