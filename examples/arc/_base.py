import math

import torch

from experiments.callbacks.arc_ttt_validation import ARCTTTValidationCallback
from experiments.datamodules.arc import ARCDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.arc_wrapper import ARCWrapper
from nvsubquadratic.lazy_config import LazyConfig


# ── Dataset size estimate ─────────────────────────────────────────────────────
# ARC-AGI-1 training split: ~400 tasks × ~3 examples × (1 + 9 colour perms)
# All 400 training tasks are now used for training (no hold-out val split).
NUM_TRAINING_SAMPLES = 12_000

# ── Training schedule ─────────────────────────────────────────────────────────
NUM_EPOCHS = 10  # matches VARC offline-training protocol
NUM_WORKERS = 8
BATCH_SIZE = 128  # 128 per GPU × 2 GPUs = 256 global batch size (matches VARC's global BS)
LEARNING_RATE = 3e-4
PLACEHOLDER = None


def get_base_config(
    *, data_dir: str, batch_size: int, learning_rate: float, weight_decay: float = 0.0
) -> ExperimentConfig:
    training_iterations = math.ceil(NUM_EPOCHS * NUM_TRAINING_SAMPLES / batch_size)

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

    # TTT validation: fine-tune task token for 100 steps on eval tasks every 5 epochs.
    # Logs val_ttt/exact_match — the same metric used at final test time.
    config.callbacks = [
        LazyConfig(ARCTTTValidationCallback)(
            ttt_val_tasks=20,
            ttt_val_every_n_epochs=5,
            ttt_steps=100,
            ttt_lr=3e-4,
        )
    ]

    config.lightning_wrapper_class = LazyConfig(ARCWrapper)()

    config.optimizer = LazyConfig(torch.optim.AdamW)(params=PLACEHOLDER, lr=learning_rate, weight_decay=weight_decay)

    config.train = TrainConfig(batch_size="${dataset.batch_size}", iterations=training_iterations, grad_clip=1.0)

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="max",
    )
    config.trainer.checkpoint_monitor = "val/exact_match"

    config.wandb = WandbConfig(entity="implicit-long-convs", project="nvsubquadratic", job_group="arc")

    return config
