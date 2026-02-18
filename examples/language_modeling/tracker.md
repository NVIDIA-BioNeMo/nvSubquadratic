# Hyena vs Attention — WikiText-103 Causal LM Tracker

Systematic comparison of Hyena and Attention on causal language modeling at multiple scales.

## Task Description

- **Dataset**: WikiText-103
- **Tokenizer**: GPT-2 (50,257 vocab)
- **Task**: Causal language modeling
- **Metric**: Validation Perplexity (exp(loss), lower is better)
- **Objective**: Validate that Hyena matches or beats Attention on LM quality across model scales and training hyperparameters

## Compute Resources

| Partition | GPUs | Type | Max Time | Account |
| :--- | :--- | :--- | :--- | :--- |
| `all6000` | 8 × RTX 6000 (24 GB) | Training | 7 days | `all6000users` |

> [!IMPORTANT]
> **Fixed batch-size rule**: All experiments within a tier use the **same effective batch size = 128**. When running on fewer GPUs, compensate with gradient accumulation. Example: 4 GPUs × bs 16 × accum 2 = 128.

## Model Architectures

Both Hyena and Attention use the same `ResidualNetwork` backbone (embedding → residual blocks → LM head) and only differ in the sequence mixer inside each block.

### Scaling Tiers

| Tier | Blocks | Hidden Dim | Params | Seq Len | Steps | BS/GPU | GPUs | Accum | Eff. BS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Debug | 4 | 128 | ~2M | 256 | 5K | 64 | 1 | 1 | 64 |
| Small | 8 | 384 | ~25M | 512 | 50K | 32 | 2 | 2 | 128 |
| Medium | 12 | 768 | ~125M | 1024 | 100K | 16 | 4 | 2 | 128 |
| Scale | — | — | ~3.5B | 2048 | 200K | — | 32 | — | 1024 |

> [!NOTE]
> Debug tier has effective batch size 64 (not 128) — within-tier comparisons (Hyena vs Attention) are still fair.

### Common Hyperparameters

- **Optimizer**: AdamW, `weight_decay=0.1`, `grad_clip=1.0`
- **Scheduler**: Cosine with 5% warmup
- **Precision**: bf16-mixed
- **Dropout**: 0.1 inside blocks, 0.0 on embedding input
- **Weight tying**: Enabled (`tie_weights=True`), `out_proj` bias disabled

______________________________________________________________________

## 🔬 Experimental Phases

### Phase 1: Scaling Tiers

**Goal**: Validate that Hyena outperforms Attention on LM perplexity across model scales.

| # | Experiment | Config | Partition | GPUs | Eff. BS | Status | Hyena PPL | Attn PPL | Hyena Job ID | Attn Job ID | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1.1 | Debug tier | `debug_{hyena,attention}.py` | all6000 | 1 | 64 | ✅ Done | 367.48 | 494.06 | 132739 | 132740 | |
| 1.2 | Small tier | `small_{hyena,attention}.py` | all6000 | 2 | 128 | ✅ Done | 37.06 | 52.50 | 134930 | 134931 | ~3h each |
| 1.3 | Medium tier (BS=16) | `medium_{hyena,attention}.py` | all6000 | 1 | 16 ⚠️ | ✅ Done | 27.97 | 36.21 | 134307 | 134716 | Batch size too small |
| 1.4 | Medium tier (BS=128) | `medium_{hyena,attention}.py` | all6000 | 4 | 128 | 🔄 Running | — | — | 138242 | 138243 | Re-run with corrected batch size |
| 1.5 | Scale tier | `scale_{hyena,attention}.py` | — | 32 | 1024 | 📅 Planned | — | — | — | — | Requires 4 nodes × 8 GPUs; not feasible on all6000 |

---

### Phase 2: Ablations (Small Tier)

**Goal**: Understand sensitivity to architectural choices and training hyperparameters.

| # | Experiment | Variable | Config | Status | Hyena PPL | Attn PPL | Hyena Job ID | Attn Job ID | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2.1 | **Baseline** | — | `small_{hyena,attention}.py` | ✅ Done | 37.06 | 52.50 | 134930 | 134931 | LR=3e-4, RoPE=False |
| 2.2 | Hyena + RoPE | `use_rope=True` | `small_hyena_rope.py` | ✅ Done | 40.08 | N/A | 136569 | — | RoPE hurts Hyena (+3 PPL) |
| 2.3 | LR = 1e-4 | `learning_rate=1e-4` | `small_{hyena,attention}_lr1e4.py` | ✅ Done | 95.63 | 148.17 | 136570 | 136571 | Too low; both models underfit |
| 2.4 | LR = 1e-3 | `learning_rate=1e-3` | `small_{hyena,attention}_lr1e3.py` | 🔄 Running | 24.57 | — | 136572 | 136573 | Hyena done; Attention running |

______________________________________________________________________

## Running Experiments

```bash
# Activate environment
source ~/.bashrc
conda activate nvsubq
cd /home/dwessel/code/nvSubquadratic-private
export PYTHONPATH=.
source .env

# Submit medium tier re-runs (corrected BS=128)
sbatch examples/language_modeling/run_medium_hyena.sh
sbatch examples/language_modeling/run_medium_attention.sh
```

______________________________________________________________________

## 🏆 Results

### Perplexity by Tier (Hyena vs Attention)

| Tier | Eff. BS | Hyena PPL | Attn PPL | Δ PPL | Winner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Debug | 64 | 367.48 | 494.06 | −126.58 | **Hyena** |
| Small | 128 | 37.06 | 52.50 | −15.44 | **Hyena** |
| Medium (BS=16 ⚠️) | 16 | 27.97 | 36.21 | −8.24 | **Hyena** |
| Medium (BS=128) | 128 | — | — | — | — |

### Learning Rate Ablation (Small Tier, Hyena)

| LR | Hyena PPL | Attn PPL | Notes |
| :--- | :--- | :--- | :--- |
| 1e-4 | 95.63 | 148.17 | Underfitting |
| **3e-4 (default)** | **37.06** | **52.50** | Baseline |
| 1e-3 | 24.57 | — | Best Hyena so far; Attention pending |

______________________________________________________________________

## Job Submission Log

> [!IMPORTANT]
> **Always update this log when submitting a job.** Record the job ID, config, and phase so we can trace results back to specific runs.

| Date | Job ID | Phase | Config | Partition | GPUs | Eff. BS | Status | PPL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| — | 132739 | 1.1 | `debug_hyena.py` | all6000 | 1 | 64 | ✅ Done | 367.48 | |
| — | 132740 | 1.1 | `debug_attention.py` | all6000 | 1 | 64 | ✅ Done | 494.06 | |
| — | 134930 | 1.2 | `small_hyena.py` | all6000 | 2 | 128 | ✅ Done | 37.06 | ~3h |
| — | 134931 | 1.2 | `small_attention.py` | all6000 | 2 | 128 | ✅ Done | 52.50 | ~3h |
| — | 134307 | 1.3 | `medium_hyena.py` | all6000 | 1 | 16 ⚠️ | ✅ Done | 27.97 | Too-small batch size |
| — | 134716 | 1.3 | `medium_attention.py` | all6000 | 1 | 16 ⚠️ | ✅ Done | 36.21 | Too-small batch size |
| — | 136569 | 2.2 | `small_hyena_rope.py` | all6000 | 2 | 128 | ✅ Done | 40.08 | RoPE ablation |
| — | 136570 | 2.3 | `small_hyena_lr1e4.py` | all6000 | 2 | 128 | ✅ Done | 95.63 | LR=1e-4 |
| — | 136571 | 2.3 | `small_attention_lr1e4.py` | all6000 | 2 | 128 | ✅ Done | 148.17 | LR=1e-4 |
| — | 136572 | 2.4 | `small_hyena_lr1e3.py` | all6000 | 2 | 128 | ✅ Done | 24.57 | LR=1e-3 |
| — | 136573 | 2.4 | `small_attention_lr1e3.py` | all6000 | 2 | 128 | 🔄 Running | — | LR=1e-3 |
| 2026-02-18 | 138242 | 1.4 | `medium_hyena.py` | all6000 | 4 | 128 | 🔄 Running | — | Corrected BS re-run |
| 2026-02-18 | 138243 | 1.4 | `medium_attention.py` | all6000 | 4 | 128 | 🔄 Running | — | Corrected BS re-run |

______________________________________________________________________

## 📊 Observations & Insights

*   **2026-02-18**: Medium tier re-runs submitted (Phase 1.4) — jobs 138242 (Hyena) and 138243 (Attention). Corrected effective batch size to 128 (4 GPUs × bs 16 × accum 2). Previous runs (134307, 134716) used eff. BS=16.
*   **2026-02-18**: LR=1e-3 Hyena (job 136572) finished at 24.57 PPL — best small-tier result so far. Attention (136573) still running.
*   Hyena consistently outperforms Attention across all completed tiers and LR settings.
*   RoPE hurts Hyena (+3 PPL on small tier): the causal convolution already provides implicit positional information; adding RoPE is redundant and slightly harmful.
*   LR=1e-3 substantially improves Hyena (37.06 → 24.57); worth re-running medium tier with this LR after the BS fix is confirmed.

______________________________________________________________________

**Last Updated**: 2026-02-18
**Status**: Phase 1.4 (medium re-run BS=128) pending submission; Phase 2.4 (Attention LR=1e-3) running
