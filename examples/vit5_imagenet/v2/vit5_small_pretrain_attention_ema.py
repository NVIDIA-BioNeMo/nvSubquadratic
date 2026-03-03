"""ViT-5-Small attention baseline pretrain with EMA — thin wrapper.

Loads the base DALI-fused attention config and adds LabeledEMAWeightAveraging
(decay=0.99996).  Validation metrics are logged as ``val/acc_ema``,
``val/loss_ema``.
"""

from examples.vit5_imagenet.vit5_small_pretrain_apex_dali_fused import get_config as _base_get_config
from experiments.callbacks.model_ema import LabeledEMAWeightAveraging
from experiments.default_cfg import ExperimentConfig
from nvsubquadratic.lazy_config import LazyConfig


def get_config() -> ExperimentConfig:
    config = _base_get_config()
    config.callbacks = [LazyConfig(LabeledEMAWeightAveraging)(decay=0.99996)]
    config.trainer.checkpoint_monitor = "val/acc_ema"
    return config
