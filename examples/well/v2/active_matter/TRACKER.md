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

| Batch/GPU | Grad Accum | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE            | it/s | total steps |
| --------- | ---------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---- | ----------- |
| 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_cfg_unet_convnext_enabled_True_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_run_start_time_1775922336_run_time_limit_hour | 0.034676797688007355 | 1.39 | 110,000     |

## Results — Attention

| Patch | Tokens/img | Batch/GPU | Grad Accum | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE            | it/s | total steps |
| ----- | ---------- | --------- | ---------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---- | ----------- |
| 16    | 256        | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_attention_enabled_True_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_16_run_start_time_1775918134_run_time_limi | 0.06180257350206375  | 1.40 | 110,000     |
| 8     | 1,024      | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_attention_enabled_True_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_8_run_start_time_1775919205_run_time_limit | 0.058588799089193344 | 1.43 | 110,000     |
| 4     | 4,096      | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_attention_enabled_True_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_4_run_start_time_1775922307_run_time_limit | 0.06164788082242012  | 1.33 | 102,480     |
| 2     | 16,384     | 32        | 2          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_attention_bs_32_enabled_True_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_2_run_start_time_1775922311_run_time | 0.09142962843179704  | 0.43 | 36,605      |

## Results — Hyena + Gaussian Mask

| Patch | Tokens/img | Batch/GPU | Grad Accum | GPUs | WandB Run                                                                                                                                                                                                    | val/VRMSE            | it/s | total steps |
| ----- | ---------- | --------- | ---------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------- | ---- | ----------- |
| 16    | 256        | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_hyena_gaussian_mask_enabled_True_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_16_run_start_time_1775914506_run  | 0.012452345341444016 | 1.38 | 110,000     |
| 8     | 1,024      | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_hyena_gaussian_mask_enabled_True_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_8_run_start_time_1775918046_run\_ | 0.007334186229854822 | 1.39 | 110,000     |
| 4     | 4,096      | 64        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_hyena_gaussian_mask_enabled_True_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_4_run_start_time_1775918057_run\_ | 0.007981576956808567 | 1.36 | 110,000     |
| 2     | 16,384     | 16        | 1          | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_active_matter_hyena_gaussian_mask_bs_16_enabled_True_experiment_dir\_/workspace/results_lr_0.005_num_nodes_1_patch_size_2_run_start_time_177587628  | 0.00703924847766757  | 2.63 | 110,000     |

## Memory Probing Notes (1×H100 80GB, with torch.compile max-autotune-no-cudagraphs)

| Patch | Batch 64   | Batch 32         | Batch 16      |
| ----- | ---------- | ---------------- | ------------- |
| 16    | OK (both)  | —                | —             |
| 8     | OK (both)  | —                | —             |
| 4     | OK (both)  | —                | —             |
| 2     | OOM (both) | Attn OK, H+G OOM | **OK (both)** |

Grad accumulation set so `batch * accum = 64`.
