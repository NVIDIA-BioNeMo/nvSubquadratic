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

| Tier | Hyena PPL | Attention PPL | Hyena Run ID | Attention Run ID | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Debug** | 367.48 | 494.06 | 132739 | 132740 | Completed (valid PPL) |
| **Medium** | 27.97 | 36.21 | 134307 | 134716 | Hyena significantly outperforms Attention at this scale |
| **Small** | - | - | 134868 | 134869 | Baseline sweep |
| **Small (Abl)** | - | - | 134870-72 | 134873-74 | RoPE and LR ablations |
| **Scale** | - | - | | | |

### 2. Ablations (Small Tier)

| Experiment | Hyena PPL | Attention PPL | Run IDs |
| :--- | :--- | :--- | :--- |
| **Baseline** (RoPE=F, LR=3e-4) | - | - | Hy: 134868, At: 134869 |
| **Hyena + RoPE** | - | N/A | 134870 |
| **LR=1e-4** | - | - | Hy: 134872, At: 134874 |
| **LR=1e-3** | - | - | Hy: 134871, At: 134873 |

---

## Technical Notes
- **Weight Tying**: Enabled by default (`tie_weights=True`). `out_proj` bias disabled.
- **Dropout**: `dropout_in` set to 0.0 (applied before embedding), dropout inside blocks remains active.
