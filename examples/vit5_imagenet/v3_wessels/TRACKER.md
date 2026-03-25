# v3_wessels — Hyena Expressivity Ablations

W&B project: [`implicit-long-convs/nvsubquadratic`](https://wandb.ai/implicit-long-convs/nvsubquadratic)

## Summary

Close the ~0.5% ImageNet-1k accuracy gap between the ViT-5-Small Hyena variant (~81.7%) and the attention baseline (~82.2%). All experiments target the sequence mixer block only — the ViT-5 backbone (12 blocks, dim 384, patch 16, CLS-row layout) is unchanged.

Four expressivity gaps were identified in the current best Hyena config:

1. **No positional encoding in gating**: The `Q * SiLU(K)` gate has zero positional information (`use_rope=False`), unlike attention which uses 2D RoPE on Q and K.
2. **Narrow SIREN kernel + FiLM bottleneck**: The SIREN kernel network (32 hidden dim, 3 layers) and FiLM conditioning bottleneck (64 dim) may limit the kernel's function approximation capacity.
3. **Depthwise-only global convolution**: CKConvND convolves each of 384 channels independently, unlike attention's per-head dense mixing across `head_dim=64` channels.
4. **Feature collapse in depthwise conv** (ConvNeXt V2, Woo et al. 2023): Depthwise convolutions produce redundant/dead channel features. GRN (Global Response Normalization) addresses this via divisive normalization across channels.

## Shared Training Recipe

ViT-5-Small (12 blocks, dim 384, patch 16, 224x224), CLS-row (15x14 grid, 13 registers), Apex FusedLAMB lr=4e-3, wd=0.05, batch 2048, 800 epochs, cosine schedule, 5 warmup epochs, 3-Augment, Mixup 0.8 + CutMix 1.0, soft-target CE loss, DropPath 0.05, LayerScale 1e-4, bf16-mixed, `torch.compile(mode="max-autotune")`.

______________________________________________________________________

## Results

> Last updated: 2026-03-25. See W&B for live curves.

| Config | Conv | RoPE | FiLM | GRN | Head | W&B | Best val/acc_ema | Status |
|--------|------|------|------|-----|------|-----|------------------|--------|
| `…hyena_cls_row_gated_film_grn_ema` | Depthwise | No | 32h/64d | Yes | CLS | [nxm3i7g6](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/nxm3i7g6) | **0.8173** (ep800) | Finished |
| `…hyena_cls_row_gated_film_ema` | Depthwise | No | 32h/64d | No | CLS | [69gwd0xm](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/69gwd0xm) | **0.8168** (ep800) | Finished |
| `…grouped_hyena_cls_row_gated_film_ema` | Grouped (6g) | No | 32h/64d | No | CLS | [qmvsolpk](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/qmvsolpk) | **0.815** (ep800) | Finished |
| `…hyena_cls_row_reghead_film_ema` | Depthwise | No | 32h/64d | No | Reg14 | [oml9thv5](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/oml9thv5) | **0.8131** (ep800) | Finished |
| `…hyena_cls_row_reghead_film_ema` | Depthwise | No | 32h/64d | No | Reg14 | [21cjexko](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/21cjexko) | **0.809** (ep519) | Stopped |
| `…hyena_gap_gated_film_ema` | Depthwise | No | 32h/64d | No | GAP | [tiwoypi8](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/tiwoypi8) | **0.808** (ep527) | Cancelled |
| `…hyena_cls_row_gated_film_grn_ema_gaussian` | Depthwise | No | 32h/64d | Yes | CLS | [wcyddd5s](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/wcyddd5s) | 0.758 (ep203) | Running (21105380) |
| `…multihead_hyena_cls_row_gated_film_ema` | MH-LR (6h, r=8) | No | 32h/64d | No | CLS | [ebhmpzc9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ebhmpzc9) | 0.736 (ep173) | Running (21074646) |
| `…hyena_cls_row_reghead_film_ema` | Depthwise | No | 32h/64d | No | Reg14 | [awstn87e](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/awstn87e) | 0.575 (ep87) | Stopped |
| `…hyena_cls_row_gated_film_rope_grn_ema` | Depthwise | Yes | 32h/64d | Yes | CLS | [swr33zw4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/swr33zw4) | 0.276 (ep63) | Cancelled |
| `…hyena_cls_row_gated_film_rope_ema` | Depthwise | Yes | 32h/64d | No | CLS | [o2ojne5x](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o2ojne5x) | 0.194 (ep47) | Cancelled |
| `…multihead_hyena_cls_row_gated_film_ema` | MH (6h, d=64) | No | 32h/64d | No | CLS | [ladnbcd9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ladnbcd9) | 0.040 (ep26) | Stopped |
| `…distributed_reg_crossattn_film_ema` | Depthwise | No | 32h/64d | No | DistReg14+XAttn | [vcmnt9ey](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vcmnt9ey) | — | Failed |
| `…distributed_reg_localpool_film_ema` | Depthwise | No | 32h/64d | No | DistReg14+Pool | [406qsc1o](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/406qsc1o) | — | Failed |
| `…apex_gated_ema` | Depthwise | No | No | No | CLS | — | — | Not started |
| `…multihead_…_gated_ema` | MH (6h, d=64) | No | No | No | CLS | — | — | Not started |
| `…gated_film_rope_wider_ema` | Depthwise | Yes | 64h/128d/4L | No | CLS | — | — | Not started |
| `…gated_film_rope_wider_grn_ema` | Depthwise | Yes | 64h/128d/4L | Yes | CLS | — | — | Not started |
| `…multihead_…_gated_rope_ema` | MH (6h, d=64) | Yes | No | No | CLS | — | — | Not started |
| `…multihead_…_gated_film_rope_ema` | MH (6h, d=64) | Yes | 32h/64d | No | CLS | — | — | Not started |
| `…multihead_…_gated_film_rope_grn_ema` | MH (6h, d=64) | Yes | 32h/64d | Yes | CLS | — | — | Not started |
| `…hyena_cls_row_reghead_ema` | Depthwise | No | No | No | Reg14 | — | — | Not started |
| `…multihead_…_reghead_ema` | MH (6h, d=64) | No | No | No | Reg14 | — | — | Not started |
| `…multihead_…_reghead_film_ema` | MH (6h, d=64) | No | 32h/64d | No | Reg14 | — | — | Not started |

______________________________________________________________________

## Finetune Results

> ViT-5 paper finetune recipe (Table 13): AdamW lr=1e-5, wd=0.1, 20 epochs, cosine schedule (25% warmup), batch 512, RandAugment, Mixup 0.8 + CutMix 1.0, label smoothing 0.1, EMA 0.99996.

| Config | Description | Pretrain W&B | Pretrain val/acc_ema | FT W&B | FT val/acc_ema | Status |
|--------|-------------|-------------|---------------------|--------|----------------|--------|
| `finetune_…film_ema` | Depthwise Hyena + FiLM, CLS head | [69gwd0xm](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/69gwd0xm) | 0.8168 | [3fppm25l](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/3fppm25l) | **0.8194** (ep8) | Finished |
| `finetune_…film_grn_ema` | Depthwise Hyena + FiLM + GRN, CLS head | [nxm3i7g6](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/nxm3i7g6) | 0.8173 | [3ovy9b1t](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/3ovy9b1t) | **0.8189** (ep3) | Finished |
| `finetune_…reghead_film_ema` | Depthwise Hyena + FiLM, Reg14 head | [oml9thv5](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/oml9thv5) | 0.8131 | [m0zqeo2p](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/m0zqeo2p) | **0.8168** (ep5) | Finished |

### Finetune Launch Commands

```bash
# FiLM + GRN (CLS head)
sbatch --job-name=ft-film-grn examples/vit5_imagenet/v3_wessels/submit_2gpu.sh \
  examples/vit5_imagenet/v3_wessels/finetune_hyena_cls_row_gated_film_grn_ema.py train.accumulate_grad_steps=1

# FiLM only (CLS head)
sbatch --job-name=ft-film examples/vit5_imagenet/v3_wessels/submit_2gpu.sh \
  examples/vit5_imagenet/v3_wessels/finetune_hyena_cls_row_gated_film_ema.py train.accumulate_grad_steps=1

# Register head + FiLM
sbatch --job-name=ft-reghead-film examples/vit5_imagenet/v3_wessels/submit_2gpu.sh \
  examples/vit5_imagenet/v3_wessels/finetune_hyena_cls_row_reghead_film_ema.py train.accumulate_grad_steps=1
```

> Effective batch size = 256/GPU × 2 GPUs = 512 (Table 13). `train.accumulate_grad_steps=1` overrides submit_2gpu.sh's default of 4.

______________________________________________________________________

## Job Submission Log

| Date | Job ID | Config | Notes |
|------|--------|--------|-------|
| 2026-03-08 | 20579151 | `…hyena_cls_row_gated_film_ema` | FiLM only, no RoPE |
| 2026-03-08 | 20583567 | `…hyena_cls_row_gated_film_rope_ema` | Cancelled — RoPE mismatched to elementwise gating |
| 2026-03-08 | 20583569 | `…hyena_cls_row_gated_film_rope_grn_ema` | Cancelled — same RoPE issue |
| 2026-03-08 | 20583570 | `…hyena_cls_row_reghead_film_ema` | Early register-head run; superseded by 20600257/20600260 |
| 2026-03-08 | 20583571 | `…multihead_hyena_cls_row_gated_film_ema` | Failed — disk quota / SIGTERM during compile |
| 2026-03-10 | 20600257 | `…hyena_cls_row_reghead_film_ema` | Rerun, plain launch without autoresume |
| 2026-03-10 | 20600260 | `…hyena_cls_row_reghead_film_ema` | Rerun, with `autoresume.enabled=True` — still started fresh |
| 2026-03-10 | 20609833 | `…hyena_gap_gated_film_ema` | GAP head + FiLM |
| 2026-03-12 | 20694240 | `…hyena_cls_row_gated_film_ema` | Resumed from local `last.ckpt` (was 20579151) |
| 2026-03-12 | 20696138 | `…hyena_gap_gated_film_ema` | Resumed from local `last.ckpt` (was 20609833) — **Cancelled**: val accuracy consistently degrading |
| 2026-03-13 | 20708657 | `…hyena_cls_row_gated_film_grn_ema` | Resumed from local `last.ckpt` — FiLM + GRN (no RoPE) |
| 2026-03-13 | 20724897 | `…hyena_cls_row_reghead_film_ema` | Resumed from local `last.ckpt` (ep519, was oml9thv5) |
| 2026-03-13 | 20725092 | `…hyena_cls_row_reghead_film_ema` | Resumed from local `last.ckpt` — ran to completion (ep800) |
| 2026-03-16 | 20770048 | `finetune_…film_grn_ema` | Finetune FiLM + GRN (CLS head) from nxm3i7g6 |
| 2026-03-16 | 20770254 | `finetune_…film_ema` | Finetune FiLM (CLS head) from 69gwd0xm |
| 2026-03-16 | 20770256 | `finetune_…reghead_film_ema` | Finetune register head + FiLM from oml9thv5 |
| 2026-03-16 | 20784221 | `…multihead_hyena_cls_row_gated_film_ema` | Resubmit after k_norm DDP fix |
| 2026-03-17 | 20807577 | `…multihead_hyena_cls_row_gated_film_ema` | MH + FiLM, 4 GPU run |
| 2026-03-17 | 20842258 | `…grouped_hyena_cls_row_gated_film_ema` | Grouped conv (6 groups) + FiLM |
| 2026-03-18 | 20880555 | `…distributed_reg_crossattn_film_ema` | Distributed registers + cross-attention (Mamba-R style) |
| 2026-03-18 | 20880577 | `…distributed_reg_localpool_film_ema` | Distributed registers + local pooling (Mamba-R style) |
| 2026-03-18 | 20884969–20887626 | `…distributed_reg_*` | Multiple resubmits — all crashed (exit code 1 / SIGTERM) |
| 2026-03-18 | 20893133 | `…multihead_hyena_cls_row_gated_film_ema` | MH + FiLM + low-rank (rank=8), 4 GPU |
| 2026-03-20 | 21005597 | `…grouped_hyena_cls_row_gated_film_ema` | Resumed from 20842258 |
| 2026-03-22 | 21074566 | `…grouped_hyena_cls_row_gated_film_ema` | Resumed, ran to completion (ep800, test/acc=0.815) |
| 2026-03-23 | 21074646 | `…multihead_hyena_cls_row_gated_film_ema` | MH low-rank resumed from 20893133 |
| 2026-03-24 | 21105380 | `…hyena_cls_row_gated_film_grn_ema_gaussian` | FiLM + GRN + Gaussian mask |

______________________________________________________________________

## Notes

### Key Findings

- **FiLM + GRN (no RoPE) is the best Hyena config at 81.73%.** GRN provides a small but consistent improvement over FiLM alone (+0.05%), confirming that addressing feature collapse in depthwise conv helps. Gap to attention baseline (82.2%) is now **0.47%**.
- **FiLM alone (no RoPE) finishes at 81.68%.** Stable throughout training, 0.52% below attention baseline.
- **Grouped conv (6 groups) + FiLM finishes at 81.5%** (test/acc=0.815, ep800). Weight-sharing across groups works but trails depthwise by ~0.2%, suggesting the extra per-channel kernel capacity matters.
- **Register head + FiLM** (Mamba-R style, 14 registers) finishes at 81.31%, ~0.4% behind CLS head variants.
- **RoPE is architecturally mismatched with elementwise gating.** In attention, RoPE works because rotations cancel in the dot product `QKᵀ`. In Hyena's `Q ⊙ SiLU(K)` gate, the operation is elementwise — rotations don't simplify and add structured noise. Both RoPE runs were cancelled after val/acc_ema collapsed.
- **GAP head + FiLM** peaked at ~0.808 (ep527) before degrading; cancelled.
- **Multi-head full-rank (ladnbcd9) failed to converge** — val/acc_ema=0.040 at ep26, stopped. Full 6h×64d dense kernels appear too expensive / numerically unstable without low-rank factorization.
- **Multi-head low-rank (rank=8) is training well** (ebhmpzc9) — val/acc_ema=0.736 at ep173, currently running. Low-rank factorization resolved the convergence issue; tracking to finish ~80%+ if trend continues.
- **FiLM + GRN + Gaussian mask** (wcyddd5s) at ep203, val/acc_ema=0.758, running. Learnable Gaussian decay envelope on the SIREN kernel; tracking slightly behind base FiLM+GRN at the same epoch.
- **Finetuning results**: FiLM-only finetune (81.94%) slightly outperforms FiLM+GRN (81.89%), both improving ~0.2% over pretrain. Register head finetune reaches 81.68% (+0.37% over pretrain).
- **Distributed registers (Mamba-R) failed to launch** — both cross-attention and local pooling variants crashed across multiple resubmits (exit code 1 / SIGTERM). Needs debugging before retry.

### TODOs

- [x] Resubmit multi-head + FiLM run with Triton cache fix — resubmitted as 20784221 (fixed unused k_norm params in DDP)
- [x] Launch FiLM + GRN (no RoPE) — finished at 81.73% (nxm3i7g6)
- [x] Launch finetune for top-3 pretrain configs — all finished (see Finetune Results)
- [x] Launch distributed register experiments (Mamba-R style) — failed, needs debugging
- [x] Grouped conv (6 groups) + FiLM — finished at 81.5% (qmvsolpk), ~0.2% behind depthwise
- [x] Multi-head low-rank (rank=8) — launched as 20893133, resumed as 21074646
- [x] FiLM + GRN + Gaussian mask — launched as 21105380
- [ ] Monitor MH low-rank run (21074646, ebhmpzc9) — at ep173/0.736, trending well
- [ ] Monitor Gaussian mask run (21105380, wcyddd5s) — at ep203/0.758, tracking behind base FiLM+GRN
- [ ] Debug distributed register crashes (both cross-attn and local pool) and resubmit
- [ ] Launch baseline runs (depthwise and multi-head without FiLM)
- [ ] After initial results (~100-200 epochs), compare learning curves against baselines on W&B
- [ ] If wider SIREN helps: try even wider (128h) or deeper (5-6 layers) variants
- [ ] **GRN placement revisit**: In ConvNeXt V2, GRN is *inside* the MLP (after expansion + activation, before projection) and *replaces* LayerScale. Our current config adds GRN after the mixer output while keeping LayerScale. If GRN shows promise, test: (a) moving GRN inside MLP, (b) dropping LayerScale, or (c) both.
- [ ] **Register recycling — MAP variant**: If register head shows promise, try Multi-head Attention Pooling (MAP) over registers. MAP uses a learnable query that cross-attends to all 14 register outputs (used in CoCa, DINOv2).

### How to Launch

```bash
sbatch --job-name=<NAME> scripts/submit.sh examples/vit5_imagenet/v3_wessels/<CONFIG>.py
```

______________________________________________________________________

**Last Updated**: 2026-03-25
