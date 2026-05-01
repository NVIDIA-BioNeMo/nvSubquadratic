# Euler Multi Quadrants v2

## Goal

Compare **CNextU-net** (baseline), **Attention**, and **Hyena + Gaussian mask**
on `euer_multi_quadrants` (512x512).

All runs share the same training recipe (24 hours, AdamW, cosine schedule with 5% warmup, bf16-mixed, grad_clip=1.0).

## Run order

1. CNextU-net (baseline, no patch size)
1. Attention   p2
1. Hyena+G    p2

## Results — CNextU-net (baseline)

| Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                  | val/VRMSE         | it/s | total steps |
| --------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---- | ----------- |
| 256       | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_gray_scott_reaction_diffusion_cfg_unet_convnext_enabled_True_experiment_dir\_/workspace/results_lr_0.0001_num_nodes_1_run_start_time_1777331397_r | 0.231915682554245 | 6.00 | 110,000     |

## Results — Attention

Total params:  13,747,979

| Patch | Tokens/img | Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE            | it/s | total steps |
| ----- | ---------- | --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---- | ----------- |
| 8     | 2,048      | 256       | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_gray_scott_reaction_diffusion_attention_enabled_True_experiment_dir\_/workspace/results_lr_0.0005_num_nodes_1_patch_size_8_run_start_time_17773947 | 0.052010852843523026 | 1.4  | 110,000     |
| 4     | 4,096      | 256       | 1    |                                                                                                                                                                                                             | 0.007981576956808567 | 1.36 | 110,000     |
| 2     | 8,192      | 16        | 1    |                                                                                                                                                                                                             | 0.00703924847766757  | 2.63 | 110,000     |

## Results — Hyena + Gaussian Mask

Total params: 14,297,099

| Patch | Tokens/img | Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE            | it/s | total steps |
| ----- | ---------- | --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---- | ----------- |
| 8     | 2,048      | 256       | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_gray_scott_reaction_diffusion_hyena_gaussian_mask_enabled_True_experiment_dir\_/workspace/results_lr_0.0005_num_nodes_1_patch_size_8_run_start_tim | 0.009229527786374092 | 1.36 | 88200       |
| 4     | 4,096      | 256       | 1    |                                                                                                                                                                                                             | 0.007981576956808567 | 1.36 | 110,000     |
| 2     | 8,192      | 16        | 1    |                                                                                                                                                                                                             | 0.00703924847766757  | 2.63 | 110,000     |
