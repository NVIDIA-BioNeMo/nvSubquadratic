# Active Matter — v2 Patch-Size Ablation

## Goal

Compare **CNextU-net** (baseline), **Attention**, and **Hyena + Gaussian mask**
on `active_matter` (256×256, 11 fields, periodic BC).  Attention and Hyena+G are
ablated across patch sizes (2, 4, 8, 16).

All runs share the same training recipe (110k iterations, AdamW lr=5e-3,
wd=1e-4, cosine schedule with 5% warmup, bf16-mixed, grad_clip=1.0).
Default batch size is **64** on **1 GPU**.

## Model sizes

| Model           | Params |
| --------------- | ------ |
| CNextU-net      | 18.59M |
| Attention (p16) | 17.80M |
| Hyena + G (p16) | 18.34M |

## Configs

| File                     | Model                 | Compile                    | FFT backend |
| ------------------------ | --------------------- | -------------------------- | ----------- |
| `unet_convnext.py`       | CNextU-net (baseline) | max-autotune-no-cudagraphs | —           |
| `attention.py`           | Attention (QKV+RoPE)  | max-autotune-no-cudagraphs | —           |
| `hyena_gaussian_mask.py` | Hyena + Gaussian mask | max-autotune-no-cudagraphs | —           |

## Patch-size overrides (Attention & Hyena+G only)

Stride, out_proj patch_size, and kernel `L_cache` are derived via OmegaConf
interpolators from `net.in_proj_cfg.patch_size`.  Only one override is needed:

```
net.in_proj_cfg.patch_size=P
```

If OOM at batch 64, add:
`dataset.batch_size=<micro_batch> trainer.accumulate_grad_batches=<accum>`
(keep `micro_batch * accum = 64`).

## Run order

1. CNextU-net (baseline, no patch size)
1. Attention p16 → p8 → p4 → p2
1. Hyena+G   p16 → p8 → p4 → p2

## Results — CNextU-net (baseline)

| Batch/GPU | Grad Accum | GPUs | WandB Run | val/MSE | it/s  | WANDB                                                               |
| --------- | ---------- | ---- | --------- | ------- | ----- | ------------------------------------------------------------------- |
| 64        | 1          | 1    |           |         | 1.308 | UB_examples_well_v2_active_matter_unet_convnext_2026-04-03-15-36-52 |

## Results — Attention

| Patch | Tokens/img | Batch/GPU | Grad Accum | GPUs | WandB Run                                                                    | val/MSE | it/s  |
| ----- | ---------- | --------- | ---------- | ---- | ---------------------------------------------------------------------------- | ------- | ----- |
| 16    | 256        | 64        | 1          | 1    | UB_examples_well_v2_active_matter_attention_2026-04-03-17-41-55              |         | 0.983 |
| 8     | 1,024      | 64        | 1          | 1    | UB_examples_well_v2_active_matter_attention_patch_size_8_2026-04-03-18-50-30 | 0.067   | 1.213 |
| 4     | 4,096      | 64        | 1          | 1    | UB_examples_well_v2_active_matter_attention_patch_size_4_2026-04-03-18-54-10 |         |       |
| 2     | 16,384     | 32        | 2          | 1    |                                                                              |         |       |

## Results — Hyena + Gaussian Mask

| Patch | Tokens/img | Batch/GPU | Grad Accum | GPUs | WandB Run                                                                               | val/MSE | it/s  |
| ----- | ---------- | --------- | ---------- | ---- | --------------------------------------------------------------------------------------- | ------- | ----- |
| 16    | 256        | 64        | 1          | 1    | UB_examples_well_v2_active_matter_hyena_gaussian_mask_patch_size_16_2026-04-03-17-08-22 | 0.0198  | 1.258 |
| 8     | 1,024      | 64        | 1          | 1    | UB_examples_well_v2_active_matter_hyena_gaussian_mask_patch_size_8_2026-04-03-18-46-37  |         | 1.145 |
| 4     | 4,096      | 64        | 1          | 1    | UB_examples_well_v2_active_matter_hyena_gaussian_mask_patch_size_4_2026-04-03-18-53-17  |         |       |
| 2     | 16,384     | 16        | 4          | 1    |                                                                                         |         |       |

## Memory Probing Notes (1×H100 80GB, with torch.compile max-autotune-no-cudagraphs)

| Patch | Batch 64   | Batch 32         | Batch 16      |
| ----- | ---------- | ---------------- | ------------- |
| 16    | OK (both)  | —                | —             |
| 8     | OK (both)  | —                | —             |
| 4     | OK (both)  | —                | —             |
| 2     | OOM (both) | Attn OK, H+G OOM | **OK (both)** |

Grad accumulation set so `batch * accum = 64`.
