"""WSD finetuning ablation — cosine schedule baseline (original ViT-5 finetune recipe).

Replicates the original finetune_vit5_small_attention.py recipe but from
pretrained checkpoint qyjyx58f. Key comparison: is WSD or the checkpoint
the issue?
"""

from examples.vit5_imagenet.wsd_ft_ablation._base import get_config as _base


def get_config():
    """Return config with original cosine finetune recipe."""
    return _base(
        lr=1e-5,
        wd=0.1,
        scheduler_name="cosine",
        warmup_pct=0.25,
        stable_pct=0.0,
    )
