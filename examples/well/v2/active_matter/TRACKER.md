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

| Batch/GPU | Grad Accum | GPUs | WandB Run | val/MSE | it/s | WANDB                                                                                                                                                                                   |
| --------- | ---------- | ---- | --------- | ------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 64        | 1          | 1    |           |         |      | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_cfg_unet_convnext_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_run_start_time_177_2026-04-10-07-57-59 |

## Results — Attention

| Patch | Tokens/img | Batch/GPU | Grad Accum | GPUs | WandB Run                                                                                                                                                                               | val/MSE | it/s |
| ----- | ---------- | --------- | ---------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---- |
| 16    | 256        | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_attention_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_16_run_start_ti_2026-04-10-07-53-21 |         |      |
| 8     | 1,024      | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_attention_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_8_run_start_tim_2026-04-10-07-53-30 |         |      |
| 4     | 4,096      | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_attention_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_4_run_start_tim_2026-04-10-07-53-41 |         |      |
| 2     | 16,384     | 32        | 2          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_attention_bs_32_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_2_run_sta_2026-04-10-07-54-08 |         |      |

## Results — Hyena + Gaussian Mask

| Patch | Tokens/img | Batch/GPU | Grad Accum | GPUs | WandB Run                                                                                                                                                                               | val/MSE | it/s |
| ----- | ---------- | --------- | ---------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---- |
| 16    | 256        | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_hyena_gaussian_mask_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_16_ru_2026-04-10-07-41-33 |         |      |
| 8     | 1,024      | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_hyena_gaussian_mask_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_8_run_2026-04-10-07-42-04 |         |      |
| 4     | 4,096      | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_hyena_gaussian_mask_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_4_run_2026-04-10-07-42-12 |         |      |
| 2     | 16,384     | 16        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_hyena_gaussian_mask_bs_16_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_2026-04-10-07-44-00 |         |      |

## Memory Probing Notes (1×H100 80GB, with torch.compile max-autotune-no-cudagraphs)

| Patch | Batch 64   | Batch 32         | Batch 16      |
| ----- | ---------- | ---------------- | ------------- |
| 16    | OK (both)  | —                | —             |
| 8     | OK (both)  | —                | —             |
| 4     | OK (both)  | —                | —             |
| 2     | OOM (both) | Attn OK, H+G OOM | **OK (both)** |

Grad accumulation set so `batch * accum = 64`.
