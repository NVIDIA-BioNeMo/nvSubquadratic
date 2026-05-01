# ViT-5-Small ImageNet-1k — Omega-Zero and Block-Diagonal Kernel Tracker

W&B project: [nvsubquadratic](https://wandb.ai/implicit-long-convs/nvsubquadratic)

Related report: [`reports/ckconv_block_diagonal_kernel/REPORT.md`](../../../reports/ckconv_block_diagonal_kernel/REPORT.md)

## Goal

Track the kernel-specific ablations for the full-Hyena ViT-5 model. These runs
are separate from the Hyena/Attention ratio sweep in [`TRACKER.md`](TRACKER.md):
they test how the CKConv SIREN kernel is initialized and trained.

All runs below use:

- Architecture: all-Hyena `H×12`
- Dataset: ImageNet-1k
- Patch size: 8
- Training budget: 250k iterations / 400 epochs for completed runs
- Kernel focus: scalar omega-zero, block-diagonal multi-omega-zero, and
  learnable block-diagonal omega-zero scales

## Configs

| Variant                             | Config / override                                                            | Kernel / mask                                                                   |
| ----------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Scalar SIREN, higher omega-zero     | `full_hyena.py`, `w0=20`                                                     | Baseline scalar `SIRENKernelND` + `GaussianModulationND`                        |
| Block-diagonal multi-omega-zero     | `full_hyena_blockdiag.py`, `omega_0_min=1`, `omega_0_max=24`                 | `BlockDiagonalMultiOmegaSIRENKernelND` + `BlockAlignedGaussianModulationND`     |
| Learnable block-diagonal omega-zero | `full_hyena_learnable_omega_blockdiag.py`, `omega_0_min=1`, `omega_0_max=24` | `BlockDiagonalLearnableOmegaSIRENKernelND` + `BlockAlignedGaussianModulationND` |

## Results

| Variant                                               | State    | Epoch | Steps   | WandB Run                                                                     | val/acc_ema | val/loss_ema | test/acc    | test/loss | Notes                                 |
| ----------------------------------------------------- | -------- | ----- | ------- | ----------------------------------------------------------------------------- | ----------- | ------------ | ----------- | --------- | ------------------------------------- |
| Scalar SIREN, higher omega-zero                       | finished | 400   | 250,000 | [xlujiniz](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xlujiniz) | 0.83500     | 0.65311      | 0.83594     | 0.64828   | Scalar omega-zero comparison point.   |
| Block-diagonal multi-omega-zero                       | finished | 400   | 250,000 | [5kn6nrzs](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5kn6nrzs) | 0.83258     | 0.66757      | 0.83376     | 0.65809   | Fixed block-wise omega-zero schedule. |
| Learnable block-diagonal omega-zero, LR scale enabled | finished | 400   | 250,000 | [d2t0692n](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/d2t0692n) | **0.83556** | 0.65991      | **0.83610** | 0.64948   | Best finished run in this set.        |
| Learnable block-diagonal omega-zero, no LR scale      | running  | 265   | 165,750 | [xdu8ox9f](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xdu8ox9f) | 0.82280     | 0.69072      | —           | —         | In progress; no test metrics yet.     |

## Conclusion

The fixed block-diagonal multi-omega-zero initialization underperforms the
stronger scalar omega-zero baseline in this setting. Making the block-diagonal
omega-zero scales learnable recovers that gap and slightly improves the
finished scalar baseline.

The no-LR-scale ablation is still running and is behind at the current
checkpoint, so the LR-scale effect should remain unresolved until that run
finishes.
