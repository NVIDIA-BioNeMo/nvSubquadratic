from examples.arc._base import BATCH_SIZE, LEARNING_RATE, get_base_config
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.networks.baselines.arc_vit import ARCViT


EMBED_DIM = 256
DEPTH = 6
NUM_HEADS = 8
MLP_DIM = 512
PATCH_SIZE = 2
MAX_SIZE = 32


def get_config():
    """Return experiment config for the ARCViT (~18M param) baseline."""
    config = get_base_config(
        data_dir="data/arc/data",
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
    )

    config.net = LazyConfig(ARCViT)(
        num_tasks=800,  # 400 training + 400 evaluation tasks
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        dropout=0.1,
        patch_size=PATCH_SIZE,
        max_size=MAX_SIZE,
    )
    return config
