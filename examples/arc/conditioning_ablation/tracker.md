# Conditioning Ablation — ARC Hyena ResNet

> **Goal:** Find a conditioning mechanism that lets the Hyena ResNet make full use of the
> per-task Task Token, closing the gap to ARCViT (≥72% val exact match) without relying
> on attention.

______________________________________________________________________

## Background & Motivation

The VARC paper embeds a learnable **Task Token** per ARC task.  In ARCViT the token lives
inside the sequence and participates in every attention layer — each spatial token directly
attends to (and updates) the task token at every depth.  This is why ARCViT reaches ~72%
val exact match.

Our Hyena ResNet currently uses one of two naive approaches:

| Approach                 | Where task_tok is used           | What it modulates               |
| ------------------------ | -------------------------------- | ------------------------------- |
| **Broadcast** (baseline) | Input embedding only, added once | Spatial activations at layer 0  |
| **FiLM on SIREN kernel** | Inside CKConvND per block        | Convolution *filter shape* only |

Neither approach gives the task token ongoing, direct influence over the residual stream
feature activations across all 12 blocks.  FiLM on the SIREN kernel is a *filter-level*
conditioner: it changes what Hyena looks for spatially, but not what it does with the
result.  The residual stream normalization, gating, and MLP branches remain unconditional.

**Current performance (as of 2026-04-15):**

| Config               |     Val exact match | Notes                                                                                                      |
| -------------------- | ------------------: | ---------------------------------------------------------------------------------------------------------- |
| ARCViT (reference)   |  **83.6%** (ep 439) | Task token in sequence, full attention · SLURM 154955, still running · still improving                     |
| Hyena broadcast      | **66.4%** (ep ~500) | SLURM 154555 · completed (OOM at ep ~500) · best ckpt ep=356                                               |
| Hyena FiLM (SIREN)   | **59.1%** (ep ~193) | SLURM 154232 · hit wall-time · best ckpt ep=191 · underperforms broadcast                                  |
| Hyena AdaLN          | **73.0%** (ep ~431) | SLURM 154569 · hit wall-time · **clear win** over broadcast (+6.6 pp)                                      |
| Hyena AdaLN (stable) |  **70.4%** (ep 338) | SLURM 154949 · running · tanh gate + cond WD=1e-4 · still improving                                        |
| Hyena seq_concat     |  **64.8%** (ep 323) | SLURM 155568 · running · task_tok as spatial row 0 (17×16) · plateau ~64-65%; underperforms AdaLN by ~8 pp |

FiLM on the SIREN kernel is currently underperforming plain broadcast.  This rules it out
as a reliable improvement and makes a clean slate necessary before stacking conditioners.

______________________________________________________________________

## Conditioning Options

### Option A — Multi-layer Broadcast *(cheap, ~zero params)*

Re-inject the task token additively at the input of every residual block instead of only
at the embedding stage.

```python
for block in self.blocks:
    x = x + task_tok[:, None, None, :]  # re-add before each block
    x = block(x, condition=None)
```

**Pros:** trivial to implement, zero extra parameters, strictly more signal than
single-injection broadcast.
**Cons:** only additive shift; no scale or gate; the same token is replicated identically
at every depth, so later blocks get no "depth-aware" task signal.

______________________________________________________________________

### Option B — AdaLN-Zero *(recommended first experiment)*

Use `AdaLNZeroResidualBlock` (already implemented in `residual_block.py`) instead of
`ResidualBlock`.  This is the DiT-style approach:

```
task_tok → SiLU → Linear(d, 6d)
         → [γ_seq, β_seq, α_seq,  γ_mlp, β_mlp, α_mlp]

x = x + α_seq · Hyena( norm(x)·(1+γ_seq) + β_seq )
x = x + α_mlp · MLP  ( norm(x)·(1+γ_mlp) + β_mlp )
```

Every layer modulates scale, shift, **and output gate** for both the Hyena mixer and the
MLP branch.  Zero-initialised linear head → training starts as unconditional baseline
(same gradient landscape as plain Hyena at step 0).

**Pros:** directly modulates the residual stream, not just the filter; proven effective
in DiT and class-conditional generation; no training instability risk; uses existing
block implementation.
**Cons:** adds one `Linear(d, 6d)` per block (6 × 384² ≈ 3.4M params for 12 blocks —
~18% overhead); still a one-way channel (task → features, not bidirectional).

Config to create: `cfg_hyena_rearc_adaln_subq_ops.py`

______________________________________________________________________

### Option C — FiLM on SIREN kernel *(already tried, currently underperforming)*

Modulates the Hyena convolution kernel shape via `KernelFiLMGenerator` inside
`SIRENKernelND`.

**Status:** Running (SLURM 154232).  Val exact match 47.93% at ep 88 — below broadcast
baseline.  Do **not** stack on top of other options until it recovers or the run finishes.

______________________________________________________________________

### Option D — Task token in spatial sequence *(medium effort, highest fidelity to ViT)*

Prepend the task token as an extra spatial "row" to the 2D feature map before the ResNet
blocks, then strip it at the end.  The 3×3 short-conv in Hyena lets top-edge spatial
positions directly "see" the task row.

```
patchify: [B, 16, 16, d]
prepend:  [B, 17, 16, d]   ← task_tok broadcast across the 16 columns of row 0
run blocks
strip:    [B, 16, 16, d]   ← discard row 0 before out_proj
```

**Pros:** task token participates structurally as a sequence member, closest analog to
ViT; the short-conv (local receptive field) naturally lets nearby patches attend to it.
**Cons:** requires changes to `ARCResNet.forward()` and careful handling of
`Patchify/Unpatchify` sizes; the FFT conv treats it as a real spatial row, which may
introduce boundary artefacts unless padding is adjusted.

______________________________________________________________________

### Option E — Cross-attention condition_mixer *(high effort, highest expressivity)*

Use the existing `condition_mixer` slot in `ResidualBlock` with a lightweight
cross-attention module.  Spatial tokens (queries) attend to the task token (key/value).

**Pros:** bidirectional information flow; the task token can route different spatial
regions to different sub-computations.
**Cons:** adds `O(seq_len)` compute per block; requires implementing/selecting a
cross-attention module; most expensive option.

______________________________________________________________________

### Option F — AdaLN-Zero + FiLM on SIREN *(combined, to try after B is validated)*

Stack AdaLN-Zero (residual-stream modulation) with FiLM on the SIREN kernel (filter
modulation).  These are complementary in principle — one changes "what Hyena looks for",
the other changes "what it does with the result".

**Condition for running this:** Option B must first show a clear improvement over plain
broadcast.  Stacking before validating B makes the attribution ambiguous.

______________________________________________________________________

## Experiment Plan

| Priority | Option                     | Config                                                           | Status                     | Notes                                                                                                                                                                                                                                                                  |
| :------: | -------------------------- | ---------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    ✅    | **Broadcast (baseline)**   | `cfg_hyena_rearc_subq_ops.py`                                    | ✅ Done (66.4%, ep ~500)   | Reference point                                                                                                                                                                                                                                                        |
|    ✅    | **FiLM on SIREN**          | `cfg_hyena_rearc_film_subq_ops.py`                               | ✅ Done (59.1%, ep ~193)   | Underperforms broadcast — ruled out as primary conditioning                                                                                                                                                                                                            |
|    ✅    | **AdaLN-Zero**             | `conditioning_ablation/cfg_hyena_rearc_adaln_subq_ops.py`        | ✅ Done (73.0%, ep ~431)   | **Clear winner** — +6.6 pp over broadcast; near-parity with ARCViT (73.4%)                                                                                                                                                                                             |
|    ✅    | **AdaLN-Zero (stable)**    | `conditioning_ablation/cfg_hyena_rearc_adaln_stable_subq_ops.py` | 🔄 Running (SLURM 154949)  | Stabilised variant (tanh gate, WD=1e-4); 70.4% at ep 338, still improving                                                                                                                                                                                              |
|    ❌    | **Multi-layer broadcast**  | configs deleted                                                  | ❌ Done (9.84%, ep ~14097) | SLURM 21931841/21931842 · ungated and gated both failed catastrophically · task_tok accumulated 12× in residual stream, dominating features · no code bug, design is flawed · **code reverted; see commit `535e814` to restore `AdditiveCondResidualBlock` + configs** |
|    1     | **ARCViT AdaLN**           | `conditioning_ablation/cfg_vit_rearc_adaln.py`                   | ⬜ Not started             | **Expected to stagger**: removes task token from sequence, replaces with side-channel AdaLN · ViT's strength is in-sequence participation · validates that AdaLN gain is Hyena-specific                                                                                |
|    2     | **Task token in sequence** | `cfg_hyena_rearc_seqtoken_subq_ops.py`                           | 🔄 Running (SLURM 155568)  | ViT5-style concat: prepend task token as extra spatial row in 2D feature map (Option D) · best 64.8% (ep 323), currently ep ~337, val=64.4% · **underperforming vs AdaLN (73.0%)** — plateau visible; candidate for cancellation                                       |
|    3     | **AdaLN-Zero + FiLM**      | `cfg_hyena_rearc_adaln_film.py` (to create)                      | ⬜ Not started             | Stack AdaLN (residual-stream) + FiLM (filter-shape) · condition: AdaLN validated ✓                                                                                                                                                                                     |
|    4     | **Cross-attention**        | `cfg_hyena_rearc_crossattn.py` (to create)                       | ⬜ Not started             | Last resort if simpler options plateau                                                                                                                                                                                                                                 |

______________________________________________________________________

## Decision Criteria (updated 2026-04-15)

- **AdaLN-Zero wins** ✅ — confirmed at 73.0% (ep ~431) vs 66.4% broadcast.
  AdaLN-Zero is now the primary conditioning for Hyena. Stable variant at 70.4% (ep 338) still running.
- **FiLM on SIREN** ❌ — underperforms broadcast (59.1% vs 66.4%). Ruled out as standalone conditioning.
  Still eligible as a stack on top of AdaLN (Option F) now that AdaLN is validated.
- **Next decision gate:** does multi-layer broadcast (Option A, priority 1) close most of the gap to AdaLN
  at zero parameter cost? If yes → further conditioning experiments may have diminishing returns.
- **ViT AdaLN stagger expected:** removing ViT's task token from the sequence and replacing with AdaLN
  side-channel is expected to hurt. Run to confirm, not to improve.
- **Task-in-sequence (Option D) underperforms AdaLN** — seq_concat peaks at 64.8% (ep 323) vs AdaLN 73.0%.
  The 3×3 short-conv receptive field is too limited for the task token row to propagate useful information
  deeply into the spatial features. Not recommended as primary conditioning. Candidate for cancellation.
- **Single-grid circular padding** is the best grid+padding combo observed so far (66.1% > single-grid zero ~66.4% broadcast baseline at same epoch range; double-grid zero 68.4%). Keep running.

______________________________________________________________________

## Notes on Implementation

### Creating `cfg_hyena_rearc_adaln_subq_ops.py`

1. Copy `cfg_hyena_rearc_subq_ops.py`.
1. Replace `block_cfg=LazyConfig(ResidualBlock)(...)` with
   `block_cfg=LazyConfig(AdaLNZeroResidualBlock)(...)`.
   - Remove `condition_mixer_cfg`, `condition_mixer_norm_cfg`, `pass_condition_to_sequence_mixer`.
   - Add `condition_norm_cfg=LazyConfig(RMSNorm)(dim=EMBED_DIM)`.
   - Add `hidden_dim=EMBED_DIM`.
1. Keep `task_injection="film"` in `ARCResNet` — this routes `task_tok` as the
   `condition` tensor through `ResidualNetwork.forward()` into each block.
1. No changes to `ARCResNet`, `ResidualNetwork`, or `AdaLNZeroResidualBlock` required.
