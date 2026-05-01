# Gray Scott Reaction Diffusion v2 Patch-Size Ablation

## Goal

Compare **CNextU-net** (baseline), **Attention**, and **Hyena + Gaussian mask**
on `gray_scott_reaction_diffusion` (128x128).  Attention and Hyena+G are
ablated across patch sizes (2, 4, 8, 16).

All runs share the same training recipe (24 hours, AdamW lr=1e-4, cosine schedule with 5% warmup, bf16-mixed, grad_clip=1.0).
Default batch size is **64** on **1 GPU**. We ended up not using gradient
accumulation for patch_size 2 because the training curve was stable at batch
size 16 and we traded off for speed.

## Patch-size overrides (Attention & Hyena+G only)

Stride, out_proj patch_size, and kernel `L_cache` are derived via OmegaConf
interpolators from `net.in_proj_cfg.patch_size`.  Only one override is needed:

```
net.in_proj_cfg.patch_size=P
```

## Run order

1. CNextU-net (baseline, no patch size)
1. Attention p16 → p8 → p4 → p2
1. Hyena+G   p16 → p8 → p4 → p2

## Results — CNextU-net (baseline)

| Batch/GPU | GPUs | WandB Run | val/VRMSE            | it/s | total steps |
| --------- | ---- | --------- | -------------------- | ---- | ----------- |
| 256       | 1    |           | 0.034676797688007355 | 1.39 | 110,000     |

## Results — Attention

| Patch | Tokens/img | Batch/GPU | GPUs | WandB Run | val/VRMSE            | it/s | total steps |
| ----- | ---------- | --------- | ---- | --------- | -------------------- | ---- | ----------- |
| 16    | 64         | 256       | 1    |           | 0.06180257350206375  | 1.40 | 110,000     |
| 8     | 256        | 256       | 1    |           | 0.058588799089193344 | 1.43 | 110,000     |
| 4     | 1,024      | 256       | 1    |           | 0.06164788082242012  | 1.33 | 102,480     |
| 2     | 4,096      | 256       | 1    |           | 0.09142962843179704  | 0.43 | 36,605      |

## Results — Hyena + Gaussian Mask

| Patch | Tokens/img | Batch/GPU | GPUs | WandB Run | val/VRMSE            | it/s | total steps |
| ----- | ---------- | --------- | ---- | --------- | -------------------- | ---- | ----------- |
| 16    | 64         | 256       | 1    |           | 0.012452345341444016 | 1.38 | 110,000     |
| 8     | 256        | 256       | 1    |           | 0.007334186229854822 | 1.39 | 110,000     |
| 4     | 1,024      | 256       | 1    |           | 0.007981576956808567 | 1.36 | 110,000     |
| 2     | 4,096      | 1256      | 1    |           | 0.00703924847766757  | 2.63 | 110,000     |
