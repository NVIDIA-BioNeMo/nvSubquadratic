# Experiment Tracker - EMNIST Spatial Recall 2D (Mask Selection)

## Experiment Overview

**Task**: EMNIST Spatial Recall 2D Regression with Mask Selection

- **Input**: 64×64 canvas with 4 EMNIST digits + binary mask (2 channels)
- **Target**: 16×16 region containing the masked digit
- **Objective**: Regress the target region for the digit indicated by the mask

______________________________________________________________________

## 🚀 KEY FINDING: "Never Train from Scratch" Confirmed!

With 5x more training (100k vs 20k iterations), **Attention beats Hyena!**

| Config          | Patch | 20k iter | 100k iter | Improvement    |
| --------------- | ----- | -------- | --------- | -------------- |
| attn_patchify_m | 8     | 0.096    | **0.016** | **6x better**  |
| attn_patchify_m | 4     | 0.352    | **0.021** | **17x better** |

**Conclusion**: Attention lacks inductive biases → needs more training to converge.
Hyena/Mamba converge faster due to built-in spatial priors.

______________________________________________________________________

## Final Results (93 runs completed)

### Top 15 by Val Loss 🏆

| Rank | Config              | Patch | Iterations | Val Loss   |
| ---- | ------------------- | ----- | ---------- | ---------- |
| 🥇   | **attn_patchify_m** | **8** | **100k**   | **0.0161** |
| 🥈   | **attn_patchify_m** | **4** | **100k**   | **0.0213** |
| 🥉   | hyena_patchify_m    | 4     | 20k        | 0.0218     |
| 4    | hyena_m (no patch)  | -     | 20k        | 0.0232     |
| 5    | hyena_s (no patch)  | -     | 20k        | 0.0234     |
| 6    | hyena_patchify_m    | 1     | 20k        | 0.0302     |
| 7    | hyena_patchify_m    | 2     | 20k        | 0.0313     |
| 8    | hyena_patchify_xs   | 2     | 40k        | 0.0364     |
| 9    | hyena_patchify_xs   | 4     | 40k        | 0.0374     |
| 10   | mamba_patchify_m    | 4     | 20k        | 0.0443     |
| 11   | mamba_patchify_m    | 8     | 20k        | 0.0487     |
| 12   | hyena_xs (no patch) | -     | 20k        | 0.0512     |
| 13   | hyena_patchify_s    | 4     | 20k        | 0.0542     |
| 14   | hyena_patchify_s    | 2     | 20k        | 0.0561     |
| 15   | hyena_patchify_xs   | 1     | 20k        | 0.0686     |

______________________________________________________________________

## Non-Patchify Models (Final)

| Config      | Size | Val Loss   | Notes                  |
| ----------- | ---- | ---------- | ---------------------- |
| **hyena_m** | M    | **0.0232** | ✅ Works great         |
| **hyena_s** | S    | **0.0234** | ✅ Works great         |
| hyena_xs    | XS   | 0.0512     | ✅ Works               |
| attn_xs     | XS   | 0.5775     | ❌ Fails (as expected) |
| mamba_xs    | XS   | 0.7234     | ❌ Fails (as expected) |

**Key insight**: Non-patchify Hyena works as well as patchify! Global convolution handles spatial recall natively.

______________________________________________________________________

## Extended Training Results

### Hyena + Patchify XS (40k iterations)

| Patch | 20k iter | 40k iter  | Improvement |
| ----- | -------- | --------- | ----------- |
| 1     | 0.077    | 0.099     | ❌ worse    |
| 2     | 0.070    | **0.036** | 1.9x better |
| 4     | 0.079    | **0.037** | 2.1x better |
| 8     | 0.145    | **0.088** | 1.6x better |
| 16    | 0.299    | **0.194** | 1.5x better |
| 32    | 0.417    | **0.328** | 1.3x better |

### Attention + Patchify M (100k iterations)

| Patch | 20k iter | 100k iter | Improvement       |
| ----- | -------- | --------- | ----------------- |
| 8     | 0.096    | **0.016** | **6x better** 🏆  |
| 4     | 0.352    | **0.021** | **17x better** 🏆 |

______________________________________________________________________

## Val Loss by Patch Size (20k iterations)

### Hyena + Patchify

| Size   | p=1   | p=2       | p=4       | p=8   | p=16  | p=32  |
| ------ | ----- | --------- | --------- | ----- | ----- | ----- |
| **M**  | 0.030 | 0.031     | **0.022** | 0.091 | 0.148 | 0.298 |
| **S**  | -     | 0.056     | **0.054** | 0.136 | 0.223 | 0.361 |
| **XS** | 0.077 | **0.070** | 0.079     | 0.145 | 0.299 | 0.417 |

### Mamba + Patchify

| Size   | p=1 | p=2   | p=4       | p=8       | p=16  | p=32  |
| ------ | --- | ----- | --------- | --------- | ----- | ----- |
| **M**  | -   | 0.159 | **0.044** | 0.049     | 0.122 | 0.277 |
| **S**  | -   | 0.372 | 0.112     | **0.083** | 0.186 | 0.331 |
| **XS** | -   | 0.554 | 0.217     | **0.147** | 0.256 | 0.379 |

### Attention + Patchify

| Size   | p=1 | p=2   | p=4   | p=8       | p=16  | p=32  |
| ------ | --- | ----- | ----- | --------- | ----- | ----- |
| **M**  | -   | 0.392 | 0.352 | **0.096** | 0.141 | 0.313 |
| **S**  | -   | 0.473 | 0.379 | **0.148** | 0.196 | 0.353 |
| **XS** | -   | 0.522 | 0.432 | **0.216** | 0.296 | 0.415 |

______________________________________________________________________

## Key Findings Summary

1. **"Never Train from Scratch" Confirmed** 🎯

   - Attention with 100k iter beats Hyena with 20k iter
   - Attention p=4 improved 17x with 5x more training
   - Confirms: Attention needs more data/training due to lack of inductive biases

1. **Non-patchify Hyena works great**

   - hyena_s/m achieve 0.023 without patchification
   - Global convolution handles spatial recall natively

1. **Mamba + Patchify works**

   - Best: 0.044 at p=4 (M size)
   - Patchification enables Mamba for 2D tasks

1. **Optimal patch sizes (at 20k iter)**:

   - Hyena: p=2-4 (handles long sequences)
   - Mamba: p=4-8
   - Attention: p=8 (needs short sequences to converge quickly)

1. **Architecture efficiency** (at 20k iter):

   - Hyena > Mamba > Attention
   - But with enough training, Attention catches up!

______________________________________________________________________

______________________________________________________________________

## 100k Iteration Runs (In Progress)

**52 new runs launched** to test if all architectures improve with 5x more training.

| Architecture          | SLURM IDs     | Status             |
| --------------------- | ------------- | ------------------ |
| Attention M (4 runs)  | 171995-171998 | 🔄 Running/Pending |
| Attention S (6 runs)  | 171999-172004 | 🔄 Running/Pending |
| Attention XS (6 runs) | 172005-172010 | 🔄 Running/Pending |
| Mamba M (6 runs)      | 172011-172016 | 🔄 Running/Pending |
| Mamba S (6 runs)      | 172017-172022 | 🔄 Running/Pending |
| Mamba XS (6 runs)     | 172023-172028 | 🔄 Running/Pending |
| Hyena M (6 runs)      | 172029-172034 | 🔄 Running/Pending |
| Hyena S (6 runs)      | 172035-172040 | 🔄 Running/Pending |
| Hyena XS (6 runs)     | 172041-172046 | 🔄 Running/Pending |

**Note**: Attention M p=4,8 already completed (171964-171965).

______________________________________________________________________

## Notes

- Input: 2 channels (grayscale + mask), Output: 1 channel
- `num_items=4`, `placement="random"`, `with_mask=True`
- Callback shows: `[canvas | mask | prediction | label]`
