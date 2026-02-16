"""Ablation: Small Attention with LR=1e-4."""
from examples.language_modeling.small_attention import get_config as base_get_config
def get_config():
    config = base_get_config()
    config.optimizer.lr = 1e-4
    return config
