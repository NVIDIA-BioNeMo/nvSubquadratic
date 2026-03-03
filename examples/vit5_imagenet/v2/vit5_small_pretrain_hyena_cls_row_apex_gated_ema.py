"""ViT-5-Small + Hyena CLS-row gated pretrain with EMA — thin wrapper.

Loads the base CLS-row gated config and adds LabeledEMAWeightAveraging (decay=0.99996).
Validation metrics are logged as ``val/acc_ema``, ``val/loss_ema``.
"""

from examples.vit5_imagenet.v2.vit5_small_pretrain_hyena_cls_row_apex_gated import get_config as _base_get_config
from experiments.callbacks.model_ema import LabeledEMAWeightAveraging
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


def get_config() -> ExperimentConfig:
    config = _base_get_config()
    config.callbacks = [LazyConfig(LabeledEMAWeightAveraging)(decay=0.99996)]
    config.trainer.checkpoint_monitor = "val/acc_ema"
    return config
