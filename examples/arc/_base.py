import torch

from experiments.datamodules.arc import ARCDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.arc_wrapper import ARCWrapper
from nvsubquadratic.lazy_config import LazyConfig


# Constants
TRAINING_ITERATIONS = 50_000
NUM_WORKERS = 8
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
PLACEHOLDER = None


def get_base_config(
    *, data_dir: str, batch_size: int, learning_rate: float, weight_decay: float = 0.0
) -> ExperimentConfig:
    config = ExperimentConfig()
    config.debug = False

    config.seed = 42

    config.dataset = LazyConfig(ARCDataModule)(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        seed=config.seed,
        max_size=32,
        num_color_permutations=9,
    )

    config.lightning_wrapper_class = LazyConfig(ARCWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(params=PLACEHOLDER, lr=learning_rate, weight_decay=weight_decay)

    config.train = TrainConfig(batch_size="${dataset.batch_size}", iterations=TRAINING_ITERATIONS, grad_clip=1.0)

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="max",
    )
    config.trainer.checkpoint_monitor = "val/exact_match"

    config.wandb = WandbConfig(entity="implicit-long-convs", project="nvsubquadratic", job_group="arc")

    return config
