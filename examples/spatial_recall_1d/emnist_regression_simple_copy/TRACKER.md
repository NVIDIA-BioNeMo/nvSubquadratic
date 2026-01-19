# Spatial Recall 1D - EMNIST Simple Copy - Experiment Tracker

## Task Description

1D version of spatial recall where:

- Images are flattened FIRST (16×16 → 256 elements)
- Flattened image placed as contiguous segment in 1D canvas (4096 elements)
- Model must recall the flattened image from a **causal** perspective

Key difference from 2D: Models are **causal** (can only see past, not future).

## Model Configurations

### XS (Extra-Small) Models

| Model     | Hidden Dim | Heads/Headdim         | Params | Notes                                |
| --------- | ---------- | --------------------- | ------ | ------------------------------------ |
| Attention | 160        | 8 heads (head_dim=20) | ~719K  | Causal attention with RoPE           |
| Mamba     | 128        | headdim=32, expand=2  | ~738K  | Unidirectional (bidirectional=False) |

**Note**: Mamba hidden_dim must be multiple of 16 for Mamba2 compatibility.

## Dataset Configuration

- **Target size**: 16×16 (flattened to 256)
- **Canvas size**: 64×64 (flattened to 4096)
- **Placement**: fixed
- **Num items**: 1
- **With mask**: False

## Experiments

### Spatial Recall (Regression) Experiments

| Job ID | Model        | readout_value | Status  | Step | Notes      |
| ------ | ------------ | ------------- | ------- | ---- | ---------- |
| 172326 | Attention XS | 0.0           | Running | 16k  | 100k iters |
| 172328 | Attention XS | -1.0          | Running | 42k  | 100k iters |
| 172337 | Mamba XS     | 0.0           | Running | 60k  | 100k iters |
| 172338 | Mamba XS     | -1.0          | Running | 60k  | 100k iters |

### Autoregressive Pretraining Experiments

| Job ID | Model                 | readout_value | Status   | Step | W&B Run    | Notes     |
| ------ | --------------------- | ------------- | -------- | ---- | ---------- | --------- |
| 172343 | Attention XS Pretrain | 0.0           | Finished | 20k  | `iefl9ab8` | 20k iters |
| 172351 | Mamba XS Pretrain     | 0.0           | Finished | 20k  | `q1wklbij` | 20k iters |
| 172352 | Attention XS Pretrain | -1.0          | Finished | 20k  | `n328nxa7` | 20k iters |
| 172353 | Mamba XS Pretrain     | -1.0          | Finished | 20k  | `aepgk6og` | 20k iters |

### Fine-tuning from Pretrained Checkpoints

| Job ID | Model        | readout_value | Pretrain Run | Status  | Step | Notes                       |
| ------ | ------------ | ------------- | ------------ | ------- | ---- | --------------------------- |
| 172373 | Mamba XS     | 0.0           | `q1wklbij`   | Running | -    | 80k iters, from AR pretrain |
| 172374 | Attention XS | 0.0           | `iefl9ab8`   | Running | -    | 80k iters, from AR pretrain |
| 172377 | Mamba XS     | -1.0          | `aepgk6og`   | Running | -    | 80k iters, from AR pretrain |
| 172378 | Attention XS | -1.0          | `n328nxa7`   | Running | -    | 80k iters, from AR pretrain |

### Experiment Variants

1. **readout_value=0.0** (default): Readout region filled with zeros
1. **readout_value=-1.0**: Readout region explicitly marked with -1, so model knows where to output
1. **Fine-tuning from pretrain**: Start from AR-pretrained weights and fine-tune on recall task (80k iters)

## Results

### Spatial Recall Results (in progress)

| Model        | readout_value | Val Loss   | Step | Notes                         |
| ------------ | ------------- | ---------- | ---- | ----------------------------- |
| Attention XS | 0.0           | 0.2735     | 16k  | Still training                |
| Attention XS | -1.0          | **0.1256** | 42k  | readout marker helps!         |
| Mamba XS     | 0.0           | 0.7402     | 60k  | Much worse than Attention     |
| Mamba XS     | -1.0          | 0.6640     | 60k  | readout marker helps slightly |

**Early observations:**

- Attention >> Mamba for this task
- readout_value=-1.0 helps both models
- Mamba struggles significantly with the recall task

### Autoregressive Pretraining Results

| Model                 | readout_value | Val Loss    | Step          | Notes                |
| --------------------- | ------------- | ----------- | ------------- | -------------------- |
| Attention XS Pretrain | 0.0           | 0.00644     | 4k (finished) | Good AR modeling     |
| Attention XS Pretrain | -1.0          | 0.00815     | 2.8k          | Still training       |
| Mamba XS Pretrain     | 0.0           | **0.00335** | 4k            | Best AR performance! |
| Mamba XS Pretrain     | -1.0          | 0.00361     | 4k            | Also excellent       |

**Observations:**

- Mamba outperforms Attention on autoregressive pretraining!
- This is expected: Mamba is designed for sequential/causal modeling
- Interesting contrast: Mamba is better at AR but worse at recall

## WandB

- **Group (Regression)**: `spatial_recall_1d_emnist_simple_copy_xs`
- **Group (Pretraining)**: `spatial_recall_1d_emnist_simple_copy_pretrain_xs`
- **Project**: `nvsubquadratic`
- **Entity**: `implicit-long-convs`

## Notes

- Mamba unidirectional (bidirectional=False) has fewer params than bidirectional, so we increased hidden_dim from 96 to 128 to match Attention's param count.
- The `readout_value=-1.0` experiment tests whether explicitly marking the output region helps the model.
- Autoregressive pretraining uses continuous mode (MSE loss) to predict next element in the sequence.
- **Key finding**: Mamba excels at next-token prediction (AR) but struggles with "find and recall" tasks where explicit position attention is needed.
