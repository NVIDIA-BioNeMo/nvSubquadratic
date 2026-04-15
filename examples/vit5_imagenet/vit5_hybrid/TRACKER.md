# ViT-5-Small ImageNet-1k — Hybrid (Interleaved Hyena + Attention)

## Goal

Ablate the ratio of Hyena-to-Attention layers in a hybrid ViT-5-Small on
ImageNet-1k pretraining.  All runs share the v5 training recipe (800 epochs,
LAMB lr=4e-3, wd=0.05, cosine schedule, 3-Augment, Mixup/CutMix, EMA 0.99996).

Key differences from v5:

- Uniform trunc_normal(std=0.02) initialization for **all** layer types.
- No FiLM. Learnable `GaussianModulationND` mask on Hyena kernel output.
- CLS readout with 4 registers for all configs.
- Token layout: `[patches, CLS, registers, padding]`.
  Padding is stripped for Attention blocks and kept for Hyena blocks.
- GRN (Global Response Normalization) on Hyena blocks.

## Configs

| File                | Pattern    | Hyena:Attn | Compile                    |
| ------------------- | ---------- | ---------- | -------------------------- |
| `full_attention.py` | `A×12`     | 0:12       | max-autotune-no-cudagraphs |
| `hybrid_ha.py`      | `(HA)×6`   | 6:6        | max-autotune-no-cudagraphs |
| `hybrid_hhha.py`    | `(HHHA)×3` | 9:3        | max-autotune-no-cudagraphs |
| `full_hyena.py`     | `H×12`     | 12:0       | max-autotune-no-cudagraphs |

All configs are patch-size agnostic. Override `net.patch_size=P` to change
resolution (default 16).

______________________________________________________________________

## Patch 16

196 tokens/image.

| Config                | Params (M) | GFLOPs (train) |
| --------------------- | ---------- | -------------- |
| Full Attention (A×12) | 22.01      | 9.41           |
| Hybrid HA (HA×6)      | 22.17      | 9.69           |
| Hybrid HHHA (HHHA×3)  | 22.25      | 9.84           |
| Full Hyena (H×12)     | 22.33      | 9.98           |

| Config                | WandB Run | val/acc_ema | test/acc | it/s (1 GPU) |
| --------------------- | --------- | ----------- | -------- | ------------ |
| Full Attention (A×12) |           |             |          |              |
| Hybrid HA (HA×6)      |           |             |          |              |
| Hybrid HHHA (HHHA×3)  |           |             |          |              |
| Full Hyena (H×12)     |           |             |          |              |

______________________________________________________________________

## Patch 8

784 tokens/image. Replacing Attention with Hyena saves significant FLOPs:
Full Hyena is **14.3% cheaper** than Full Attention (38.72 vs 45.16 GFLOPs).

| Config                | Params (M) | GFLOPs (train) |
| --------------------- | ---------- | -------------- |
| Full Attention (A×12) | 22.01      | 45.16          |
| Hybrid HA (HA×6)      | 22.18      | 41.94          |
| Hybrid HHHA (HHHA×3)  | 22.26      | 40.33          |
| Full Hyena (H×12)     | 22.34      | 38.72          |

| Config                | WandB Run | val/acc_ema | test/acc | it/s (1 GPU) |
| --------------------- | --------- | ----------- | -------- | ------------ |
| Full Attention (A×12) |           |             |          |              |
| Hybrid HA (HA×6)      |           |             |          |              |
| Hybrid HHHA (HHHA×3)  |           |             |          |              |
| Full Hyena (H×12)     |           |             |          |              |
