"""Visual Mamba (Vim) baseline for ARC-AGI, matching the reference ViT training setup.

Data, batch size, LR, val split, and precision are identical to cfg_vit_rearc.py so
results are directly comparable.  The backbone swaps the transformer encoder for
bidirectional Mamba SSM blocks.

Key design choices vs ARCViT:
- Task token is prepended to the patch sequence (same as ARCViT).
- Bidirectional scan: even layers run forward, odd layers run on the reversed
  sequence; outputs are summed per pair (Vim ``if_bidirectional=True`` strategy).
- Absolute positional embedding on patch tokens only (not task token).
- Pure-PyTorch selective scan — no mamba_ssm package required.
- ``compile = False``: the sequential scan loop is not friendly to torch.compile
  in its current form; remove or set to True once a parallel scan is available.
"""

import math

import torch

from examples.arc._base import LEARNING_RATE, NUM_EPOCHS, NUM_GPUS, PLACEHOLDER
from experiments.datamodules.arc import ARCDataModule
from experiments.default_cfg import ExperimentConfig, SchedulerConfig, TrainConfig, WandbConfig
from experiments.lightning_wrappers.arc_wrapper import ARCWrapper
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.arc_vim import ARCVim


# ── Architecture ──────────────────────────────────────────────────────────────
# Parameter budget roughly matches the ARCViT-S config:
# ARCViT: embed=512, depth=10, heads=8 → ~28 M params
# ARCVim: embed=384, depth=12, expand=2 → ~26 M params
EMBED_DIM = 384
DEPTH = 12  # must be even (bidirectional pairing)
PATCH_SIZE = 2
MAX_SIZE = 32  # 32×32 canvas → 256 patch tokens at patch_size=2
D_STATE = 16  # Mamba SSM state size
D_CONV = 4  # Mamba depthwise-conv width
EXPAND = 2  # Mamba channel-expand factor

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
GRAD_ACCUM_STEPS = 1
NUM_TRAINING_SAMPLES_REARC = 413_020


def get_config() -> ExperimentConfig:
    """ARCVim baseline trained on ARC + RE-ARC, matching the reference ViT setup."""
    training_iterations = math.ceil(NUM_EPOCHS * NUM_TRAINING_SAMPLES_REARC / (BATCH_SIZE * NUM_GPUS))

    config = ExperimentConfig()
    config.debug = False
    config.seed = 42

    config.dataset = LazyConfig(ARCDataModule)(
        data_dir="data/arc/data",
        rearc_dir="/home/dwessel/code/VARC_info/raw_data/re_arc",
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

    config.train = TrainConfig(batch_size="${dataset.batch_size}", iterations=training_iterations, grad_clip=1.0)

    config.scheduler = SchedulerConfig(
        name="cosine",
        warmup_iterations_percentage=0.05,
        total_iterations="${train.iterations}",
        mode="max",
    )
    config.trainer.checkpoint_monitor = "val/exact_match"
    # compile=False: the sequential scan loop is not torch.compile-friendly yet.
    config.compile = False
    config.trainer.precision = "bf16-mixed"

    config.net = LazyConfig(ARCVim)(
        num_tasks=400,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        patch_size=PATCH_SIZE,
        max_size=MAX_SIZE,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        drop_path_rate=0.1,
        dropout=0.1,
    )

    config.wandb = WandbConfig(entity="implicit-long-convs", project="nvsubquadratic", job_group="arc")

    return config
