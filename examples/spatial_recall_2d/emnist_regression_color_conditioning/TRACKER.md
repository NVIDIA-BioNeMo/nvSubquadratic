# Experiment Tracker - EMNIST Spatial Recall 2D (Color Conditioning)

## Experiment Overview

**Task**: EMNIST Spatial Recall 2D Regression with Color Conditioning

- **Input**: 64×64 canvas with 4 EMNIST digits in RGB (3 channels) with colored bounding boxes
- **Target**: 16×16 RGB region containing the digit colored with its frame color
- **Objective**: Regress the target region AND its color based on the frame indicator

______________________________________________________________________

## Key Differences from Color Selection

| Aspect         | Color Selection           | Color Conditioning                   |
| -------------- | ------------------------- | ------------------------------------ |
| **Input**      | 3ch RGB (colored frames)  | 3ch RGB (colored frames)             |
| **Output**     | 1ch grayscale digit       | 3ch RGB digit (in frame color)       |
| **Selection**  | Colored bounding box      | Colored bounding box                 |
| **Task**       | Recall digit by color cue | Recall digit AND reproduce its color |
| **Hypothesis** | Color discrimination only | Color discrimination + color memory  |

______________________________________________________________________

## Configs Available

| Config                 | Architecture | Patchify | Size | Patch | Status        |
| ---------------------- | ------------ | -------- | ---- | ----- | ------------- |
| ccnn_hyena_xs          | Hyena        | No       | XS   | -     | ✅ 0.0218     |
| ccnn_hyena_s           | Hyena        | No       | S    | -     | ✅ 0.0139     |
| ccnn_hyena_m           | Hyena        | No       | M    | -     | ✅ **0.0028** |
| ccnn_mamba_xs          | Mamba        | No       | XS   | -     | ✅ 0.0754     |
| ccnn_mamba_s           | Mamba        | No       | S    | -     | ✅ 0.0778     |
| ccnn_mamba_m           | Mamba        | No       | M    | -     | ✅ 0.0253     |
| ccnn_attn_xs           | Attention    | No       | XS   | -     | ✅ 0.2738     |
| ccnn_attn_s            | Attention    | No       | S    | -     | ✅ 0.2667     |
| ccnn_attn_m            | Attention    | No       | M    | -     | ✅ 0.2731     |
| ccnn_hyena_patchify_xs | Hyena        | Yes      | XS   | p=2   | ✅ 0.0103     |
| ccnn_mamba_patchify_xs | Mamba        | Yes      | XS   | p=4   | ✅ 0.0159     |
| ccnn_attn_patchify_xs  | Attention    | Yes      | XS   | p=8   | ✅ 0.1835     |
| ccnn_hyena_patchify_s  | Hyena        | Yes      | S    | p=2   | ✅ 0.0078     |
| ccnn_hyena_patchify_m  | Hyena        | Yes      | M    | p=2   | ✅ 0.0043     |
| ccnn_mamba_patchify_s  | Mamba        | Yes      | S    | p=4   | ✅ 0.0231     |
| ccnn_mamba_patchify_m  | Mamba        | Yes      | M    | p=4   | ✅ 0.0050     |
| ccnn_attn_patchify_s   | Attention    | Yes      | S    | p=8   | ✅ 0.1256     |
| ccnn_attn_patchify_m   | Attention    | Yes      | M    | p=8   | ✅ 0.0743     |
| ccnn_delta_hyena_xs    | Delta-Hyena  | No       | XS   | -     | ❌ OOM        |
| ccnn_delta_hyena_s     | Delta-Hyena  | No       | S    | -     | ❌ OOM        |
| ccnn_delta_hyena_m     | Delta-Hyena  | No       | M    | -     | ❌ OOM        |
| ccnn_delta_hyena_patchify_xs | Delta-Hyena | Yes | XS | p=2 | 🔄 Running (0.045) |
| ccnn_delta_hyena_patchify_s  | Delta-Hyena | Yes | S | p=2 | 🔄 Running |
| ccnn_delta_hyena_patchify_m  | Delta-Hyena | Yes | M | p=2 | ❌ OOM |
| ccnn_reasoning_delta_hyena_xs | Reasoning Delta | No | XS* | r=4 | Not started |

______________________________________________________________________

## Results

### Top 10 Results (50k iterations)

| Rank | Architecture | Size | Patch | Val Loss   | WandB ID | Notes                     |
| ---- | ------------ | ---- | ----- | ---------- | -------- | ------------------------- |
| 1    | **Hyena**    | M    | none  | **0.0028** | tqbnoevm | ✅ **BEST!** Non-patchify |
| 2    | Hyena        | M    | p=2   | 0.0043     | 8mm3bhl5 | ✅ Best patchify          |
| 3    | Mamba        | M    | p=4   | 0.0050     | tnpmnw72 | ✅                        |
| 4    | Hyena        | S    | p=2   | 0.0078     | vu5flsj1 | ✅                        |
| 5    | Hyena        | XS   | p=2   | 0.0103     | hchvn3u7 | ✅ Best XS                |
| 6    | Hyena        | S    | none  | 0.0139     | lxbrmb5e | ✅ Non-patchify (resumed) |
| 7    | Mamba        | XS   | p=4   | 0.0159     | 103dn9dy | ✅                        |
| 8    | Hyena        | XS   | none  | 0.0218     | pps0ivor | ✅ Non-patchify           |
| 9    | Mamba        | S    | p=4   | 0.0231     | w0julwcz | ✅                        |
| 10   | Mamba        | M    | none  | 0.0253     | y6k3jm36 | ✅ Non-patchify           |

### S/M Results Comparison (50k iterations)

| Arch      | Size | Patch | Val Loss   | WandB ID |
| --------- | ---- | ----- | ---------- | -------- |
| **Hyena** | M    | p=2   | **0.0043** | 8mm3bhl5 |
| Mamba     | M    | p=4   | 0.0050     | tnpmnw72 |
| Hyena     | S    | p=2   | 0.0078     | vu5flsj1 |
| Mamba     | S    | p=4   | 0.0231     | w0julwcz |
| Attention | M    | p=8   | 0.0743     | 66h54vmw |
| Attention | S    | p=8   | 0.1256     | lrcy9uhf |

### All Results by Architecture

#### Hyena + Patchify

| Size | Patch | Val Loss   | WandB Run | SLURM ID | Notes      |
| ---- | ----- | ---------- | --------- | -------- | ---------- |
| XS   | p=2   | **0.0103** | hchvn3u7  | 174189   | ✅ Best XS |
| S    | p=2   | **0.0078** | vu5flsj1  | 174402   | ✅ Best S  |
| M    | p=2   | **0.0043** | 8mm3bhl5  | 174403   | ✅ Best M  |

#### Mamba + Patchify

| Size | Patch | Val Loss | WandB Run | SLURM ID | Notes              |
| ---- | ----- | -------- | --------- | -------- | ------------------ |
| XS   | p=4   | 0.0159   | 103dn9dy  | 174169   | ✅                 |
| S    | p=4   | 0.0231   | w0julwcz  | 174404   | ✅                 |
| M    | p=4   | 0.0050   | tnpmnw72  | 174405   | ✅ Close to Hyena! |

#### Attention + Patchify

| Size | Patch | Val Loss | WandB Run | SLURM ID | Notes        |
| ---- | ----- | -------- | --------- | -------- | ------------ |
| XS   | p=8   | 0.1835   | qpngs2v3  | 174149   | ✅           |
| S    | p=8   | 0.1256   | lrcy9uhf  | 174406   | ✅           |
| M    | p=8   | 0.0743   | 66h54vmw  | 174407   | ✅ Improving |

______________________________________________________________________

## Experiment Log

### XS Experiments (50k iterations)

- **Date**: 2026-01-21
- **SLURM IDs**: 174189, 174169, 174149
- **Iterations**: 50k
- **Status**: ✅ All 3 completed!

| SLURM ID | Config                 | Architecture  | Val Loss   | WandB    | Status  |
| -------- | ---------------------- | ------------- | ---------- | -------- | ------- |
| 174189   | ccnn_hyena_patchify_xs | Hyena p=2     | **0.0103** | hchvn3u7 | ✅ Best |
| 174169   | ccnn_mamba_patchify_xs | Mamba p=4     | 0.0158     | 103dn9dy | ✅      |
| 174149   | ccnn_attn_patchify_xs  | Attention p=8 | 0.1806     | qpngs2v3 | ✅      |

**History**:

- Initial jobs (174003-174005) crashed due to tensor `.view()` bug. Fixed with `.reshape()`.
- Jobs 174147, 174148 preempted at step ~3300. Resubmitted as 174168, 174169.
- Job 174168 (Hyena) preempted again at step ~19628. Resubmitted as 174189.
- **All 3 XS completed** on 2026-01-21.

### S & M Experiments (50k iterations)

- **Date**: 2026-01-22
- **Iterations**: 50k
- **Status**: ✅ All completed!

| SLURM ID | Config                | Architecture  | Size | Val Loss   | WandB ID |
| -------- | --------------------- | ------------- | ---- | ---------- | -------- |
| 174402   | ccnn_hyena_patchify_s | Hyena p=2     | S    | **0.0078** | vu5flsj1 |
| 174403   | ccnn_hyena_patchify_m | Hyena p=2     | M    | **0.0043** | 8mm3bhl5 |
| 174404   | ccnn_mamba_patchify_s | Mamba p=4     | S    | 0.0231     | w0julwcz |
| 174405   | ccnn_mamba_patchify_m | Mamba p=4     | M    | 0.0050     | tnpmnw72 |
| 174406   | ccnn_attn_patchify_s  | Attention p=8 | S    | 0.1256     | lrcy9uhf |
| 174407   | ccnn_attn_patchify_m  | Attention p=8 | M    | 0.0743     | 66h54vmw |

______________________________________________________________________

## Notes

- Input: 3 channels (RGB with colored frames)
- Output: 3 channels (RGB digit colored with frame color)
- `num_items=4`, `placement="random"`, `use_colored_frames=True`, `colored_label=True`
- Callback shows: `[canvas | prediction | label]` (all RGB)
- Patch sizes based on best results from color_selection experiments:
  - Hyena: p=2 (best overall)
  - Mamba: p=4 (best for Mamba)
  - Attention: p=8 (best for Attention, though still fails)

______________________________________________________________________

### XS Patch Size Sweep (50k iterations)

- **Date**: 2026-01-22
- **Purpose**: Validate optimal patch sizes across all architectures
- **Status**: ✅ All 12 completed!

| Patch | Hyena                 | Mamba                 | Attention         |
| ----- | --------------------- | --------------------- | ----------------- |
| p=2   | **0.0103** (hchvn3u7) | 0.0659 (rp9kz8fy)     | 0.2719 (drj5ob1h) |
| p=4   | 0.0255 (t2jpyelg)     | **0.0159** (103dn9dy) | 0.2513 (b9ow0zck) |
| p=8   | 0.2077 (i9bll5eg)     | 0.0453 (0z6flse6)     | 0.1835 (qpngs2v3) |
| p=16  | 0.1912 (6d0zjka3)     | 0.1350 (q5yojque)     | 0.1717 (lr2h1xv5) |

**Non-patchify XS**:

| Architecture | Val Loss | WandB ID |
| ------------ | -------- | -------- |
| Hyena        | 0.0218   | pps0ivor |
| Mamba        | 0.0754   | qd9sm7u2 |
| Attention    | 0.2738   | twgy8s8l |

______________________________________________________________________

## Key Findings

1. **Hyena M non-patchify is BEST** (0.0028) - outperforms patchify p=2 (0.0043)!
1. **Patchify helps for smaller models** (XS/S), but not required at M size for Hyena
1. **Mamba benefits from patchify** - p=4 patchify (0.005) beats non-patchify (0.025) at M
1. **Attention struggles** with color conditioning regardless of configuration (best: 0.0743)
1. **Scaling helps significantly** - M consistently beats S beats XS for all architectures
1. **Architecture ranking**: Hyena >> Mamba >> Attention for color conditioning task

______________________________________________________________________

### Non-Patchify S & M Experiments (50k iterations)

- **Date**: 2026-01-25
- **Purpose**: Compare non-patchify models at larger sizes
- **Status**: ✅ All 6 completed!

| Config       | Architecture | Size | Hidden | Val Loss   | WandB ID | Status            |
| ------------ | ------------ | ---- | ------ | ---------- | -------- | ----------------- |
| ccnn_hyena_s | Hyena        | S    | 256    | **0.0139** | lxbrmb5e | ✅ Resumed & done |
| ccnn_hyena_m | Hyena        | M    | 416    | **0.0028** | tqbnoevm | ✅ Best!          |
| ccnn_mamba_s | Mamba        | S    | 160    | 0.0778     | 3agn9x3e | ✅                |
| ccnn_mamba_m | Mamba        | M    | 256    | 0.0253     | y6k3jm36 | ✅                |
| ccnn_attn_s  | Attention    | S    | 256    | 0.2667     | rteejbf3 | ✅                |
| ccnn_attn_m  | Attention    | M    | 384    | 0.2731     | bzgvkbob | ✅                |

### Non-Patchify Results Comparison

| Arch      | XS                | S                     | M                     |
| --------- | ----------------- | --------------------- | --------------------- |
| Hyena     | 0.0218 (pps0ivor) | **0.0139** (lxbrmb5e) | **0.0028** (tqbnoevm) |
| Mamba     | 0.0754 (qd9sm7u2) | 0.0778 (3agn9x3e)     | 0.0253 (y6k3jm36)     |
| Attention | 0.2738 (twgy8s8l) | 0.2667 (rteejbf3)     | 0.2731 (bzgvkbob)     |

**Observations:**

- Hyena M non-patchify (0.0028) outperforms Hyena M patchify p=2 (0.0043)!
- Hyena scales very well: XS (0.022) → S (0.014) → M (0.003) = **7x improvement!**
- Mamba scales well with size (M is 3x better than S/XS)
- Attention shows no improvement with scale for non-patchify

______________________________________________________________________

______________________________________________________________________

**Last Updated**: 2026-02-01
**Status**: 🔄 Delta-Hyena experiments on `geodude` partition (RTX A5000, 24GB VRAM).

### Delta-Hyena Experiment Log

**Latest SLURM Job Run (122061-122066)**:

| Config                       | SLURM ID | Status       | Val Loss  | Notes                           |
| ---------------------------- | -------- | ------------ | --------- | ------------------------------- |
| ccnn_delta_hyena_xs          | 122061   | ❌ OOM       | -         | Crashed in delta_rule_parallel  |
| ccnn_delta_hyena_s           | 122062   | ❌ OOM       | -         | Crashed in delta_rule_parallel  |
| ccnn_delta_hyena_m           | 122063   | ❌ OOM       | -         | Crashed in fftconv2d            |
| ccnn_delta_hyena_patchify_xs | 122064   | 🔄 Running   | **0.045** | Epoch 3 @ 55%, v_num=xlg1       |
| ccnn_delta_hyena_patchify_s  | 122065   | 🔄 Running   | ~0.010 train | Epoch 0 complete, v_num=xdw9 |
| ccnn_delta_hyena_patchify_m  | 122066   | ❌ OOM       | -         | Crashed in delta_rule_parallel  |

**Observations**:
- Non-patchify Delta-Hyena models run out of memory on 24GB GPUs
- Only patchify variants (XS/S) can train on RTX A5000
- Patchify_m also OOM - may need smaller batch size or larger GPU
