# WikiText-103 Experiment Tracker

**Goal**: Compare causal language modeling quality (PPL) of Hyena vs Attention at scale.
**Metric**: Validation Perplexity (log_e(loss)).

## Planned Experiments

### 1. Scaling Tiers (Primary)
**Why**: Validate scaling laws and memory efficiency.
**Config**: WikiText-103, GPT-2 tokenizer.
**Variations**:
- **Debug**: 2M params, 256 seq, 5K steps
- **Small**: 25M params, 512 seq, 50K steps
- **Medium**: 125M params, 1024 seq, 100K steps
- **Scale**: 350M params, 2048 seq, 200K steps

### 2. Ablation: RoPE on Hyena
**Why**: Hyena currently uses `use_rope=False` (relying on implicit position encoding from causal conv), while Attention uses `use_rope=True`. This ablation will add RoPE to Hyena to see if explicit position encoding helps.
**Variations**: `use_rope` ∈ {False (default), True}

### 3. Ablation: Learning Rate Sensitivity
**Why**: Ensure fair comparison (one architecture might prefer higher/lower LR).
**Config**: Small tier.
**Variations**: `lr` ∈ {1e-4, 3e-4 (default), 1e-3}

### 4. Ablation: Weight Tying
**Why**: Standard practice for small LMs. Enabled by default (`tie_weights=True`).
**Variations**: Compare with `tie_weights=False` to see parameter efficiency gain.

---

## Results

### 1. Perplexity by Tier

| Tier | Hyena PPL | Attention PPL | Hyena Run ID | Attention Run ID | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Debug** | 367.48 | 494.06 | 132739 | 132740 | ✅ Done | |
| **Small** | 37.06 | 52.50 | 134930 | 134931 | ✅ Done | 2 GPU, all6000, ~3h each |
| **Medium** | 27.97 | 36.21 | 134307 | 134716 | ✅ Done | |
| **Scale** | - | - | - | - | 📅 Planned | Requires 32 GPUs (4 nodes × 8), not feasible on all6000 |

### 2. Ablations (Small Tier) — all6000

| Experiment | Hyena PPL | Attention PPL | Run IDs | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** (RoPE=F, LR=3e-4) | 37.06 | 52.50 | Hy: 134930, At: 134931 | ✅ Done |
| **Hyena + RoPE** | - | N/A | Hy: 136569 | � Running |
| **LR=1e-4** | - | - | Hy: 136570, At: 136571 | � Running |
| **LR=1e-3** | - | - | Hy: 136572, At: 136573 | � Running |

---

## Technical Notes
- **Weight Tying**: Enabled by default (`tie_weights=True`). `out_proj` bias disabled.
- **Dropout**: `dropout_in` set to 0.0 (applied before embedding), dropout inside blocks remains active.
- **Effective Batch Sizes** (important for fair comparisons):
  - Debug: 64 × 1 GPU × 1 accum = **64**
  - Small: 32 × 2 GPU × 2 accum = **128**
  - Medium: 16 × 1 GPU × 1 accum = **16** ⚠️
  - Scale: 8 × 32 GPU × 4 accum = **1024**

> [!WARNING]
> Batch size affects LM performance (larger → more stable gradients → often better PPL). Within-tier comparisons (Hyena vs Attention) are fair since both use the same effective batch size. Cross-tier comparisons are confounded. When running new model variants, **always match the effective batch size** of the tier baseline.
