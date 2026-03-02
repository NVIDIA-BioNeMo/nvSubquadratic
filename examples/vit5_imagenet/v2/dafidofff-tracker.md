# ViT-5 ImageNet v2 — Experiment Tracker

## Runs

### Run: `vit5_small_pretrain_multihead_hyena_cls_row_apex_fix_init`
- **Config:** `examples/vit5_imagenet/v2/vit5_small_pretrain_multihead_hyena_cls_row_apex_fix_init.py`
- **Script:** `scripts/run_imagenet_vit5_hyena.sh`
- **Partition:** `cees`, 8× RTX A5000
- **Effective batch size:** 2048 (128 per GPU × 8 GPUs, no grad accum)
- **Epochs:** 800
- **Optimizer:** Apex FusedLAMB, lr=4e-3, wd=0.05
- **SLURM job ID:** 143520 (restarted from 143515 which crashed on wandb empty run_id)
- **Wandb run ID:** `fo9x4gbj`
- **Wandb link:** https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/fo9x4gbj
- **Reason:** Ablate initialization — using default PyTorch init (Kaiming uniform) instead of `small_init` (in-proj) + Wang init (out-proj) for both QKVSequenceMixer and MLP.
