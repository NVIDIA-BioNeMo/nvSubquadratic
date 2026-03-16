r"""ViT-5-Small ImageNet-1k — Validation only.

Loads the v3 attention pretrain config and overrides it for validation:
  - ``train.do = False`` — skip training
  - ``start_from_checkpoint`` — download weights from W&B run ``2y06y121``
  - ``debug = True`` — offline wandb (no logging to remote)

The model architecture is imported directly from the training config so it
stays in sync automatically.

Run (inside SLURM container):
    source .env && PYTHONPATH=. python experiments/run.py \
        --config examples/vit5_imagenet/v1/VALIDATION_vit5_small_dali_fused.py
"""

from examples.vit5_imagenet.v3.vit5_small_pretrain_attention_ema import get_config as _train_get_config
from experiments.default_cfg import AutoResumeConfig, ExperimentConfig, StartFromCheckpointConfig, TrainConfig
from experiments.utils.checkpointing import StripCompiledPrefix
from nvsubquadratic.lazy_config import LazyConfig


def get_config() -> ExperimentConfig:
    """Return validation-only config for the v3 attention model."""
    config = _train_get_config()

    # Disable training
    config.train = TrainConfig(
        do=False,
        batch_size="${dataset.batch_size}",
        iterations=0,
        precision="bf16-mixed",
    )

    # Load weights from W&B run 2y06y121
    config.start_from_checkpoint = StartFromCheckpointConfig(
        load=True,
        run_path="implicit-long-convs/nvsubquadratic/2y06y121",
        alias="latest",
        strict=True,
        callbacks=[LazyConfig(StripCompiledPrefix)()],
    )

    # Run offline, no logging
    config.debug = True
    config.compile = False
    config.autoresume = AutoResumeConfig(enabled=False)

    return config
