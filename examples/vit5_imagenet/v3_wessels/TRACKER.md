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

> Last updated: 2026-03-12 ~15:10. See W&B for live curves.

| Config | Conv | RoPE | FiLM | GRN | Head | W&B | Best val/acc_ema | Status |
|--------|------|------|------|-----|------|-----|------------------|--------|
| `…hyena_cls_row_gated_film_ema` | Depthwise | No | 32h/64d | No | CLS | [69gwd0xm](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/69gwd0xm) | **0.811** (ep527) | Running |
| `…hyena_cls_row_reghead_film_ema` | Depthwise | No | 32h/64d | No | Reg14 | [oml9thv5](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/oml9thv5) | **0.810** (ep519) | Stopped |
| `…hyena_cls_row_reghead_film_ema` | Depthwise | No | 32h/64d | No | Reg14 | [21cjexko](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/21cjexko) | **0.809** (ep519) | Stopped |
| `…hyena_gap_gated_film_ema` | Depthwise | No | 32h/64d | No | GAP | [tiwoypi8](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/tiwoypi8) | **0.808** (ep527) | Running |
| `…hyena_cls_row_reghead_film_ema` | Depthwise | No | 32h/64d | No | Reg14 | [awstn87e](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/awstn87e) | 0.575 (ep87) | Stopped |
| `…hyena_cls_row_gated_film_rope_grn_ema` | Depthwise | Yes | 32h/64d | Yes | CLS | [swr33zw4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/swr33zw4) | 0.276 (ep63) | Cancelled |
| `…hyena_cls_row_gated_film_rope_ema` | Depthwise | Yes | 32h/64d | No | CLS | [o2ojne5x](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o2ojne5x) | 0.194 (ep47) | Cancelled |
| `…multihead_hyena_cls_row_gated_film_ema` | MH (6h, d=64) | No | 32h/64d | No | CLS | [r4iwlaxt](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/r4iwlaxt) | — | Failed |
| `…apex_gated_ema` | Depthwise | No | No | No | CLS | — | — | Not started |
| `…multihead_…_gated_ema` | MH (6h, d=64) | No | No | No | CLS | — | — | Not started |
| `…hyena_cls_row_gated_film_grn_ema` | Depthwise | No | 32h/64d | Yes | CLS | — | — | Not started |
| `…gated_film_rope_wider_ema` | Depthwise | Yes | 64h/128d/4L | No | CLS | — | — | Not started |
| `…gated_film_rope_wider_grn_ema` | Depthwise | Yes | 64h/128d/4L | Yes | CLS | — | — | Not started |
| `…multihead_…_gated_rope_ema` | MH (6h, d=64) | Yes | No | No | CLS | — | — | Not started |
| `…multihead_…_gated_film_rope_ema` | MH (6h, d=64) | Yes | 32h/64d | No | CLS | — | — | Not started |
| `…multihead_…_gated_film_rope_grn_ema` | MH (6h, d=64) | Yes | 32h/64d | Yes | CLS | — | — | Not started |
| `…hyena_cls_row_reghead_ema` | Depthwise | No | No | No | Reg14 | — | — | Not started |
| `…multihead_…_reghead_ema` | MH (6h, d=64) | No | No | No | Reg14 | — | — | Not started |
| `…multihead_…_reghead_film_ema` | MH (6h, d=64) | No | 32h/64d | No | Reg14 | — | — | Not started |

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
| 2026-03-12 | 20696138 | `…hyena_gap_gated_film_ema` | Resumed from local `last.ckpt` (was 20609833) |

______________________________________________________________________

## Notes

### Key Findings So Far

- **FiLM alone (no RoPE) trains stably** and is currently leading at 81.1% with ~130 epochs remaining.
- **RoPE is architecturally mismatched with elementwise gating.** In attention, RoPE works because rotations cancel in the dot product `QKᵀ`. In Hyena's `Q ⊙ SiLU(K)` gate, the operation is elementwise — rotations don't simplify and add structured noise. Both RoPE runs were cancelled after val/acc_ema collapsed.
- **Register head + FiLM** (Mamba-R style, 14 registers) performs comparably to CLS head + FiLM (~0.810 vs ~0.811).
- **GAP head + FiLM** is slightly behind (~0.808) but still competitive.
- **Multi-head + FiLM** failed on infra (disk quota during Triton compile), needs resubmit.

### TODOs

- [ ] Resubmit multi-head + FiLM run with Triton cache fix
- [ ] Launch FiLM + GRN (no RoPE) — replaces cancelled RoPE+GRN run
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

**Last Updated**: 2026-03-12
