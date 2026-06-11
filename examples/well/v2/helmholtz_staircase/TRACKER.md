# Helmholtz Staircase v2

## Goal

Compare **CNextU-net** (baseline), **Attention**, and **Hyena + Gaussian mask**
on `helmholtz_staircase` (1024×256, open BCs).

All runs share the same training recipe (AdamW, cosine schedule with 5% warmup, bf16-mixed, grad_clip=1.0).

Paper baseline: CNextU-net VRMSE = 0.02758, FNO = 0.00046 (dominant).

## Run order

1. CNextU-net (baseline, no patch size) ✓
1. Attention   p8, p4 (p2 OOM)
1. Hyena+G    p8, p4, p2

## Tokens per image

| Patch | Tokens/img |
| ----- | ---------- |
| 8     | 4,096      |
| 4     | 16,384     |
| 2     | 65,536     |

## Results — CNextU-net (baseline)

| Batch/GPU | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE | it/s   | total steps |
| --------- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ----------- |
| 42        | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_helmholtz_staircase_cfg_unet_convnext_bs_42_enabled_True_experiment_dir\_/workspace/results_lr_0.003_num_nodes_1_run_start_time_1781084507_run_tim | 0.004504  | 0.8347 | 72,115      |

## Results — Attention

| Patch | Tokens/img | Batch/GPU | LR   | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE | it/s   | total steps |
| ----- | ---------- | --------- | ---- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ----------- |
| 8     | 4,096      | 68        | 3e-3 | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_helmholtz_staircase_attention_bs_64_enabled_True_experiment_dir\_/workspace/results_lr_0.003_num_nodes_1_patch_size_8_run_start_time_1780763270_ru | 0.004970  | 0.5900 | 50,973      |
| 4     | 16,384     | 36        | 3e-3 | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_helmholtz_staircase_attention_bs_36_enabled_True_experiment_dir\_/workspace/results_lr_0.003_num_nodes_1_patch_size_4_run_start_time_1780777733_ru | 0.014651  | 0.2120 | 18,318      |
| 2     | 65,536     | —         | —    | —    | OOM at bs≤12                                                                                                                                                                                                | —         | —      | —           |

## Results — Hyena + Gaussian Mask

| Patch | Tokens/img | Batch/GPU | LR   | GPUs | WandB Run                                                                                                                                                                                                   | val/VRMSE | it/s   | total steps |
| ----- | ---------- | --------- | ---- | ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------ | ----------- |
| 8     | 4,096      | 48        | 3e-4 | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_helmholtz_staircase_hyena_gaussian_mask_bs_48_enabled_True_experiment_dir\_/workspace/results_lr_0.0003_num_nodes_1_patch_size_8_run_start_time_17 | 0.006125  | 0.7212 | 49,979      |
| 4     | 16,384     | 48        | 3e-4 | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_helmholtz_staircase_hyena_gaussian_mask_bs_48_enabled_True_experiment_dir\_/workspace/results_lr_0.0003_num_nodes_1_patch_size_4_run_start_time_17 | 0.011989  | 0.2432 | 17,511      |
| 2     | 65,536     | 12        | 3e-4 | 1    | OV\_\_workspaces_nvSubquadratic-private_examples_well_v2_helmholtz_staircase_hyena_gaussian_mask_bs_12_enabled_True_experiment_dir\_/workspace/results_lr_0.0003_num_nodes_1_patch_size_2_run_start_time_17 | 0.031442  | 0.3148 | 22,099      |

## Notes

- **Attention p2**: OOM at all tested batch sizes (≤12); not run.
- **Hyena+G uses gradient checkpointing** (`GRADIENT_CHECKPOINTING = True` in `hyena.py`) to fit larger patch sizes in memory.
- **LR difference**: Attention converged best at lr=3e-3; Hyena+G at lr=3e-4 — a 10× gap vs Attention. This mirrors the supernova finding where Hyena+G preferred lower LRs.
- **Patch-size trend (Hyena+G)**: VRMSE degrades as patch size decreases (p8: 0.006 → p4: 0.012 → p2: 0.031). This is the **opposite** of the supernova_explosion_64 trend, consistent with helmholtz being a high-frequency problem where the large effective receptive field at coarser patches is actually beneficial. Smaller patches give more tokens but reduce the kernel span relative to the wavelength.
- **Attention p4 step count** (18,318) and **Hyena+G p4** (17,511) are much lower than p8 (~50k) — these may not be fully converged; flag if comparing head-to-head.
- **CNextU-net** val/VRMSE = 0.004504 at 72,115 steps — slightly better than Attention p8 (0.004970) and well below the paper baseline (0.02758).
