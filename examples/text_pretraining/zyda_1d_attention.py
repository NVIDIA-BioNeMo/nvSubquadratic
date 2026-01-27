"""Config file for Text Pretraining using 1D Attention on Zyda-2."""

import torch
from pytorch_lightning import Trainer

from experiments.datamodules.zyda_datamodule import ZydaDataModule
from experiments.default_cfg import (
    SchedulerConfig,
    TextGenerationConfig,
    TextPretrainingExperimentConfig,
    TrainConfig,
    WandbConfig,
)
from experiments.lightning_wrappers.text_pretraining_wrapper import TextPretrainingWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.attention import Attention
from nvsubquadratic.modules.init_functions import partial_wang_init_fn_with_num_layers, small_init
from nvsubquadratic.modules.mlp import MLP
from nvsubquadratic.modules.residual_block import ResidualBlock
from nvsubquadratic.modules.sequence_mixer import QKVSequenceMixer
from nvsubquadratic.networks.general_purpose_resnet import ResidualNetwork


class EmbeddingAdapter(torch.nn.Embedding):
    """Adapter for Embedding to accept in_features and out_features."""

    def __init__(self, in_features: int, out_features: int, **kwargs):
        """Initialize the adapter."""
        super().__init__(num_embeddings=in_features, embedding_dim=out_features, **kwargs)


PLACEHOLDER = None
WANDB_ENTITY = "dafidofff"
DATA_DIM = 1

# Dataset
BATCH_SIZE = 4
MAX_LENGTH = 1024
VOCAB_SIZE = 131072  # Mistral-NeMo-Minitron vocab size
# Tokenization is done in the dataloader, so with to many workers
# we might run out of cpu memory
NUM_WORKERS = 4

# Model
# Keeping similar param count to Hyena config:
# - Same hidden_dim (256), same num_blocks (6)
# - Attention has fewer params in the mixer itself (no global conv),
#   but we keep num_heads=8 for a head_dim of 32
NUM_HIDDEN_CHANNELS = 256
NUM_BLOCKS = 6
NUM_HEADS = 8  # head_dim = 256/8 = 32
DROPOUT_RATE = 0.1

# Optimisation
TRAINING_ITERATIONS = 50_000
WARMUP_ITERATIONS_PERCENTAGE = 0.02
WEIGHT_DECAY = 0.1
LEARNING_RATE = 6e-4
GRAD_CLIP = 1.0
VAL_CHECK_INTERVAL = 10000
LIMIT_VAL_BATCHES = 100


def get_config() -> TextPretrainingExperimentConfig:
    """Return the Text Pretraining configuration."""
    config = TextPretrainingExperimentConfig()
    config.debug = False
    config.seed = 42

    # Dataset
    config.dataset = LazyConfig(ZydaDataModule)(
        dataset_name="Zyphra/Zyda-2",
        tokenizer_name="nvidia/Mistral-NeMo-Minitron-8B-Base",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        streaming=True,
    )

    # Model
    config.net = LazyConfig(ResidualNetwork)(
        in_channels=VOCAB_SIZE,
        out_channels=VOCAB_SIZE,
        num_blocks=NUM_BLOCKS,
        hidden_dim=NUM_HIDDEN_CHANNELS,
        data_dim=DATA_DIM,
        # Input projection is Embedding layer via Adapter
        in_proj_cfg=LazyConfig(EmbeddingAdapter)(in_features="${net.in_channels}", out_features="${net.hidden_dim}"),
        # Output projection is Linear layer
        out_proj_cfg=LazyConfig(torch.nn.Linear)(in_features="${net.hidden_dim}", out_features="${net.out_channels}"),
        norm_cfg=LazyConfig(torch.nn.LayerNorm)(normalized_shape="${net.hidden_dim}"),
        block_cfg=LazyConfig(ResidualBlock)(
            sequence_mixer_cfg=LazyConfig(QKVSequenceMixer)(
                hidden_dim="${net.hidden_dim}",
                mixer_cfg=LazyConfig(Attention)(
                    hidden_dim="${net.hidden_dim}",
                    num_heads=NUM_HEADS,
                    apply_qk_norm=True,
                    use_rope=True,
                    is_causal=True,  # Causal attention for autoregressive LM
                    attn_dropout=0.0,
                    rope_base=10000.0,
                ),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            sequence_mixer_norm_cfg="${net.norm_cfg}",
            # No conditioning for pretraining
            condition_mixer_cfg=LazyConfig(torch.nn.Identity)(),
            condition_mixer_norm_cfg=LazyConfig(torch.nn.Identity)(),
            mlp_cfg=LazyConfig(MLP)(
                dim="${net.hidden_dim}",
                activation="glu",
                expansion_factor=2.0,
                dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
                init_method_in=small_init,
                init_method_out=partial_wang_init_fn_with_num_layers(num_layers=NUM_BLOCKS),
            ),
            mlp_norm_cfg="${net.norm_cfg}",
            dropout_cfg=LazyConfig(torch.nn.Dropout)(p=DROPOUT_RATE),
        ),
        dropout_in_cfg=LazyConfig(torch.nn.Identity)(),  # No dropout on input indices
        condition_in_proj_cfg=None,
    )

    # Text Generation Config
    config.text_generation = LazyConfig(TextGenerationConfig)(
        enabled=True,
        every_n_epochs=1,
        num_samples=4,
        max_new_tokens=50,
        temperature=0.8,
        top_k=50,
    )

    # Lightning Wrapper
    config.lightning_wrapper_class = LazyConfig(TextPretrainingWrapper)(
        vocab_size=VOCAB_SIZE,
    )

    # Optimizer
    config.optimizer = LazyConfig(torch.optim.AdamW)(
        params=PLACEHOLDER,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),  # Standard for LLMs
    )

    # Training
    config.train = TrainConfig(
        batch_size="${dataset.batch_size}",
        iterations=TRAINING_ITERATIONS,
        grad_clip=GRAD_CLIP,
        track_grad_norm=2,  # Track L2 norm of gradients
    )

    # Trainer
    config.trainer = LazyConfig(Trainer)(
        val_check_interval=VAL_CHECK_INTERVAL,
        limit_val_batches=LIMIT_VAL_BATCHES,
    )

    # Scheduler
    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=WARMUP_ITERATIONS_PERCENTAGE,
        total_iterations="${train.iterations}",
        mode="min",
    )

    # WandB
    config.wandb = WandbConfig(
        job_group="text-pretraining-attention",
        entity=WANDB_ENTITY,
    )

    return config
