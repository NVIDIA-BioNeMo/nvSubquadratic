import math

import torch

from examples.arc._base import LEARNING_RATE, NUM_EPOCHS, NUM_GPUS, PLACEHOLDER
from experiments.datamodules.arc import ARCDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.arc_wrapper import ARCWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.arc_vit import ARCViT


EMBED_DIM = 512
DEPTH = 10
NUM_HEADS = 8
MLP_DIM = 512
PATCH_SIZE = 1 # Changed to 1
MAX_SIZE = 32

BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1

NUM_TRAINING_SAMPLES_REARC = 413_020


def get_config():  # noqa: D103
    training_iterations = math.ceil(NUM_EPOCHS * NUM_TRAINING_SAMPLES_REARC / (BATCH_SIZE * NUM_GPUS))

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42

    config.dataset = LazyConfig(ARCDataModule)(
        data_dir="data/arc/data",
        rearc_dir="data/re_arc",
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        seed=config.seed,
        max_size=MAX_SIZE,
        num_color_permutations=9,
        rearc_num_color_permutations=0,
        val_task_split="training",
        val_subset="test",
    )

    config.lightning_wrapper_class = LazyConfig(ARCWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(params=PLACEHOLDER, lr=LEARNING_RATE, weight_decay=0.0)

    config.train = TrainConfig(batch_size="${dataset.batch_size}", iterations=training_iterations, grad_clip=1.0, accumulate_grad_steps=1)

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="max",
    )
    config.trainer.checkpoint_monitor = "val/exact_match"
    config.compile = True
    config.compile_mode = "max-autotune"
    config.trainer.precision = "bf16-mixed"

    config.net = LazyConfig(ARCViT)(
        num_tasks=400,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        dropout=0.1,
        patch_size=PATCH_SIZE,
        max_size=MAX_SIZE,
    )

    config.wandb = WandbConfig(entity="implicit-long-convs", project="nvsubquadratic", job_group="arc")

    return config
