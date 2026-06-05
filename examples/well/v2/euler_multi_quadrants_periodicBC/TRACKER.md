# Euler Multi Quadrants v2

## Goal

Compare **CNextU-net** (baseline), **Attention**, and **Hyena + Gaussian mask**
on `euler_multi_quadrants_periodicBC` (512x512).

All runs share the same training recipe (24 hours, AdamW, cosine schedule with 5% warmup, bf16-mixed, grad_clip=1.0).

## Run order

1. CNextU-net (baseline, no patch size)
1. Attention   p8
1. Hyena+G    p8, p4, p2

## Results — CNextU-net (baseline)

| Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE | it/s | total steps |
| --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---- | ----------- |
| 42        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_euler_multi_quadrants_periodicBC_cfg_unet_convnext_bs_42_enabled_True_experiment_dir\_/workspace/results_lr_0.003_num_nodes_1_run_start_time_17806 | 0.0332    | 1.52 | 110,000     |

## Results — Attention

Total params:  13,747,979

| Patch | Tokens/img | Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                    | val/VRMSE | it/s | total steps |
| ----- | ---------- | --------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | ---- | ----------- |
| 8     | 2,048      | 42        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_euler_multi_quadrants_periodicBC_attention_bs_42_enabled_True_experiment_dir\_/workspace/results_lr_0.003_num_nodes_1_patch_size_8_run_start_time\_ | 0.1293    | 1.38 | 110,000     |
| 4     | 4,096      | 42        | 1    | OOM at bs=42                                                                                                                                                                                                 | —         | —    | —           |
| 2     | 8,192      | 12        | 1    | OOM at bs=12                                                                                                                                                                                                 | —         | —    | —           |

## Results — Hyena + Gaussian Mask

Total params: 14,297,099

| Patch | Tokens/img | Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE | it/s   | total steps |
| ----- | ---------- | --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ----------- |
| 8     | 2,048      | 42        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_euler_multi_quadrants_periodicBC_hyena_gaussian_mask_bs_42_experiment_dir\_/workspace/results_lr_0.003_num_2026-06-03-06-49-37                     | 0.0311    | 1.33   | 110,000     |
| 4     | 4,096      | 42        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_euler_multi_quadrants_periodicBC_hyena_gaussian_mask_bs_42_enabled_True_experiment_dir\_/workspace/results_lr_0.003_num_nodes_1_patch_size_4_run_s | 0.0378    | 0.5915 | 44,787      |
| 2     | 8,192      | 12        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_euler_multi_quadrants_periodicBC_hyena_gaussian_mask_bs_12_enabled_True_experiment_dir\_/workspace/results_lr_0.003_num_nodes_1_patch_size_2_run_s | 0.1088    | 0.4749 | 40,949      |
