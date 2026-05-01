# Acoustic Scattering Maze

## Goal

Compare **CNextU-net** (baseline), **Attention**, and **Hyena + Gaussian mask**

All runs share the same training recipe (24 hours, AdamW lr=5e-3,
wd=1e-4, cosine schedule with 5% warmup, bf16-mixed, grad_clip=1.0).
Default batch size is **64** on **1 GPU**.

## Run order

1. CNextU-net (baseline, no patch size)
1. Attention p16 → p8 → p4 → p2
1. Hyena+G   p16 → p8 → p4 → p2

## Results — CNextU-net (baseline)

| Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE            | it/s         | total steps |
| --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------ | ----------- |
| 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_acoustic_scattering_maze_cfg_unet_convnext_enabled_True_experiment_dir\_/workspace/results_lr_0.001_num_nodes_1_run_start_time_1776227052_run_time | 0.008186010643839836 | 5.3794992175 | 110,000     |

## Results — Attention

| Patch | Tokens/img | Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE            | it/s         | total steps |
| ----- | ---------- | --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------ | ----------- |
| 16    | 256        | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_acoustic_scattering_maze_attention_enabled_True_experiment_dir\_/workspace/results_lr_0.001_num_nodes_1_run_start_time_1776226723_run_time_limit_h | 0.04768187180161476  | 5.8760683761 | 110,000     |
| 8     | 1,024      | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_acoustic_scattering_maze_attention_enabled_True_experiment_dir\_/workspace/results_lr_0.001_num_nodes_1_patch_size_8_run_start_time_1776335992_run | 0.045643627643585205 | 5.7651991614 | 110,000     |
| 4     | 4,096      | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_acoustic_scattering_maze_attention_enabled_True_experiment_dir\_/workspace/results_lr_0.001_num_nodes_1_patch_size_4_run_start_time_1776340302_run | 0.056851577013731    | 2.3595023595 | 110,000     |
| 2     | 16,384     | 16        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_acoustic_scattering_maze_attention_bs_16_enabled_True_experiment_dir\_/workspace/results_lr_0.001_num_nodes_1_patch_size_2_run_start_time_17763786 | 0.10574174672365189  | 0.8702652948 | 69,019      |

## Results — Hyena + Gaussian Mask

| Patch | Tokens/img | Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE            | it/s         | total steps |
| ----- | ---------- | --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------ | ----------- |
| 16    | 256        | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_acoustic_scattering_maze_hyena_enabled_True_experiment_dir\_/workspace/results_lr_0.001_num_nodes_1_run_start_time_1776335625_run_time_limit_hours | 0.0260219257324934   | 5.8873902805 | 110,000     |
| 8     | 1,024      | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_acoustic_scattering_maze_hyena_gaussian_mask_enabled_True_experiment_dir\_/workspace/results_lr_0.001_num_nodes_1_patch_size_8_run_start_time_1776 | 0.00858149304986     | 4.2379411311 | 110,000     |
| 4     | 4,096      | 64        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_acoustic_scattering_maze_hyena_gaussian_mask_enabled_True_experiment_dir\_/workspace/results_lr_0.001_num_nodes_1_patch_size_4_run_start_time_1776 | 0.006847195327281952 | 1.6516516517 | 110,000     |
| 2     | 16,384     | 16        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_acoustic_scattering_maze_hyena_gaussian_mask_bs_16_enabled_True_experiment_dir\_/workspace/results_lr_0.001_num_nodes_1_patch_size_2_run_start_tim | 0.006232426967471838 | 1.5774680204 | 110,000     |
