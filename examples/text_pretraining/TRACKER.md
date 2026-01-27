# Experiment Tracker - Text Pretraining (FineWeb Edu 10BT)

## Experiment Overview

**Task**: Next-Token Prediction on FineWeb Edu 10BT
**Data**: `fineweb_edu_10bt_shuffled`
**Objective**: Pretrain subquadratic architectures (Hyena) and compare against standard Transformer baselines.
**Tokenizer**: `nvidia/Mistral-NeMo-Minitron-8B-Base`

______________________________________________________________________

## Tracked Parameters

Key parameters to vary across experiments:

- **Model Type**: Hyena (SSM-based) vs. Attention (Transformer)
- **Scale**: Dimension (`dim`), Number of layers (`n_layers`), Number of heads (`n_heads`)
- **Sequence Length**: `seq_len` (default: 2048)
- **Optimization**: Learning rate (`lr`), Warmup steps, Gradient accumulation steps

______________________________________________________________________

## Metrics to Monitor

Defined in \[metrics_summary.md\](file:///home/dwessel/code/nvSubquadratic-private/examples/text_pretraining/metrics_summary.md):

1. **Loss** (`train/loss`, `val/loss`): Primary objective. Lower is better.
1. **Perplexity** (`train/ppl`, `val/ppl`): $e^{\\text{loss}}$. Easier to interpret for scale comparison.
1. **Throughput** (`wps`): Tokens per second per device. Higher is better for scaling.
1. **Validation Benchmarks**: Zero-shot performance on `hellaswag` and `arc_easy`.

______________________________________________________________________

## Ongoing & Completed Experiments

| Experiment Name             | Model       | Dim  | Layers | Heads | BS  | Status  | Result / Note                |
| :-------------------------- | :---------- | :--- | :----- | :---- | :-- | :------ | :--------------------------- |
| `hyena_fineweb_train`       | Hyena       | 1024 | 8      | 8     | 8   | Running | Chkpt 27k reached. Loss ~5.6 |
| `transformer_fineweb_train` | Transformer | 768  | 12     | 12    | 2   | Setup   | (~150M params baseline)      |

______________________________________________________________________

## Key Insights & Notes

- **Hyena Initialization**: Using `wang_init` for `out_proj` and `small_init` for others as defined in `zyda_1d_hyena.py`.
- **Memory Optimization**: Transformer uses selective activation checkpointing to fit in memory with FineWeb context.
- **Evaluation**: Log-likelihood evaluations are performed every 1000 steps.
