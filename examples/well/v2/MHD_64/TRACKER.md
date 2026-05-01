# MHD_64

## Goal

Compare **CNextU-net** (baseline), **Attention**, and **Hyena + Gaussian mask**

All runs share the same training recipe ( AdamW lr=2-5e-3,
wd=1e-4, cosine schedule with 5% warmup, bf16-mixed, grad_clip=1.0).
Default batch size is **64** on **1 GPU**.

## Run order

1. CNextU-net (baseline, no patch size)
1. Attention p16 → p8 → p4 → p2
1. Hyena+G   p16 → p8 → p4 → p2

## Results — CNextU-net (baseline)

| Batch/GPU | GPUs | WandB Run                                                                                                                                                                                           | val/VRMSE           | it/s         | total steps |
| --------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------ | ----------- |
| 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_MHD_64_unet_convnext_enabled_True_experiment_dir\_/workspace/results_lr_0.002_num_nodes_1_run_start_time_1776680521_run_time_limit_hours_4 | 0.21076937019824984 | 0.1066574978 | 9,151       |

## Results — Attention

| Patch | Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                    | val/VRMSE           | it/s         | total steps |
| ----- | --------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------- | ------------ | ----------- |
| 8     | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_MHD_64_attention_enabled_True_experiment_dir\_/workspace/results_lr_0.002_num_nodes_1_patch_size_8_run_start_time_1777096814_run_time_limit_hours\_ | 0.30443859100341797 | 0.7517476852 | 64951       |
| 4     | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_MHD_64_attention_enabled_True_experiment_dir\_/workspace/results_lr_0.002_num_nodes_1_patch_size_4_run_start_time_1777036508_run_time_limit_hours\_ | 0.21639029681682587 | 0.3653587963 | 31567       |
| 2     | 16        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_MHD_64_attention_bs_16_enabled_True_experiment_dir\_/workspace/results_lr_0.002_num_nodes_1_patch_size_2_run_start_time_1777036405_run_time_limit\_ | 0.30372998118400574 | 0.0803125    | 6939        |

## Results — Hyena + Gaussian Mask

All experiments run with lr=2e-3

| Patch | Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE           | it/s         | total steps |
| ----- | --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------ | ----------- |
| 8     | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_MHD_64_hyena_gaussian_mask_enabled_True_experiment_dir\_/workspace/results_lr_0.002_num_nodes_1_patch_size_8_run_start_time_1776680370_run_time_li | 0.2809585630893707  | 0.7427864124 | 63610       |
| 4     | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_MHD_64_hyena_gaussian_mask_enabled_True_experiment_dir\_/workspace/results_lr_0.002_num_nodes_1_patch_size_4_run_start_time_1776680398_run_time_li | 0.10876571387052536 | 0.5321398562 | 45598       |
| 2     | 16        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_MHD_64_hyena_gaussian_mask_bs_8_enabled_True_experiment_dir\_/workspace/results_lr_0.002_num_nodes_1_patch_size_2_run_start_time_1776682943_run_ti | 0.05432766303420067 | 0.8076215108 | 69323       |
