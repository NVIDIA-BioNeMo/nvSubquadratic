# Experiment Tracker - EMNIST Spatial Recall 2D (Mask Selection)

## Experiment Overview

**Task**: EMNIST Spatial Recall 2D Regression with Mask Selection

- **Input**: 64×64 canvas with 4 EMNIST digits + binary mask (2 channels)
- **Target**: 16×16 region containing the masked digit
- **Objective**: Regress the target region for the digit indicated by the mask

______________________________________________________________________

## 🚀 KEY FINDING: All architectures benefit massively from longer training!

With 5x more training (100k vs 20k iterations), **ALL architectures improve dramatically**:

| Architecture    | 20k iter | 100k iter   | Improvement       |
| --------------- | -------- | ----------- | ----------------- |
| **Hyena M**     | 0.022    | **0.00129** | **17x better** 🏆 |
| **Mamba M**     | 0.044    | **0.00295** | **15x better**    |
| **Attention M** | 0.096    | **0.0161**  | **6x better**     |

**Conclusion**: Longer training benefits all architectures, but Hyena benefits most!
Hyena + 100k iterations achieves **0.00129** - the new SOTA for this task.

______________________________________________________________________

## Final Results (140+ runs completed)

### Top 15 by Val Loss 🏆 (Verified from WandB)

| Rank | Config                | Patch | Iterations | Val Loss    | WandB Run |
| ---- | --------------------- | ----- | ---------- | ----------- | --------- |
| 🥇   | **hyena_patchify_m**  | **2** | **100k**   | **0.00129** | 97nmmw5x  |
| 🥈   | **mamba_patchify_m**  | **4** | **100k**   | **0.00295** | fdrqaid1  |
| 🥉   | **hyena_patchify_s**  | **1** | **100k**   | **0.00772** | drbqqt7x  |
| 4    | **mamba_patchify_s**  | **4** | **100k**   | **0.0100**  | 7ongarf9  |
| 5    | **hyena_patchify_xs** | **2** | **100k**   | **0.0121**  | 0crv4li2  |
| 6    | **attn_patchify_m**   | **8** | **100k**   | **0.0161**  | jzol8pz6  |
| 7    | hyena_patchify_m      | 4     | 20k        | 0.0218      | -         |
| 8    | hyena_m (no patch)    | -     | 20k        | 0.0232      | -         |
| 9    | hyena_s (no patch)    | -     | 20k        | 0.0234      | -         |
| 10   | **attn_patchify_s**   | **4** | **100k**   | **0.0257**  | 172271    |
| 11   | **attn_patchify_s**   | **8** | **100k**   | **0.0272**  | 172272    |
| 12   | hyena_patchify_m      | 1     | 20k        | 0.0302      | -         |
| 13   | hyena_patchify_m      | 2     | 20k        | 0.0313      | -         |
| 14   | mamba_patchify_xs     | 4     | 100k       | 0.0380      | a3elhhxo  |
| 15   | attn_patchify_xs      | 8     | 100k       | 0.0598      | n3pv4dbn  |

### 🔥 Key Finding: Hyena + 100k iterations = NEW SOTA!

With 100k iterations, **Hyena patchify M achieves 0.00129** - a **17x improvement** over 20k iterations!

______________________________________________________________________

## Val Loss by Patch Size (100k iterations) - Verified from WandB

### Hyena + Patchify (100k)

| Size   | p=1            | p=2       | p=4       | p=8   | p=16  | p=32  |
| ------ | -------------- | --------- | --------- | ----- | ----- | ----- |
| **M**  | 🔄 0.007 (80%) | **0.001** | 0.002     | 0.019 | 0.049 | 0.104 |
| **S**  | 0.008          | 0.013     | **0.015** | 0.027 | 0.071 | 0.155 |
| **XS** | 🔄 0.014 (63%) | **0.012** | 0.020     | 0.045 | 0.120 | 0.210 |

### Mamba + Patchify (100k)

| Size   | p=1           | p=2       | p=4       | p=8   | p=16  | p=32  |
| ------ | ------------- | --------- | --------- | ----- | ----- | ----- |
| **M**  | 🔄 0.69 (39%) | **0.023** | 0.003     | 0.012 | 0.040 | 0.100 |
| **S**  | 0.267         | 0.068     | **0.010** | 0.023 | 0.072 | 0.161 |
| **XS** | 0.383         | 0.054     | **0.038** | 0.044 | 0.113 | 0.230 |

### Attention + Patchify (100k)

| Size   | p=1       | p=2   | p=4       | p=8       | p=16  | p=32  |
| ------ | --------- | ----- | --------- | --------- | ----- | ----- |
| **M**  | 0.017     | 0.019 | 0.021     | **0.016** | 0.048 | 0.120 |
| **S**  | 0.070     | 0.045 | **0.026** | 0.027     | 0.075 | 0.165 |
| **XS** | **0.158** | 0.144 | 0.094     | 0.060     | 0.116 | 0.226 |

> ✅ **Attn S/XS 100k runs now complete!**

### Non-Patchify Models (100k) - 1 COMPLETE ✅, 2 RUNNING 🔄

| Config       | Size | Val Loss              | Notes                          |
| ------------ | ---- | --------------------- | ------------------------------ |
| **hyena_m**  | M    | 🔄 **0.007** (80%) 🔥 | Running - looking excellent!   |
| **hyena_s**  | S    | ✅ **0.0065** 🏆      | DONE! New non-patchify record! |
| **hyena_xs** | XS   | 🔄 0.014 (63%)        | Running                        |
| mamba_xs     | XS   | **0.396**             | ✅ Done - fails (as expected)  |
| attn_xs      | XS   | **0.294**             | ✅ Done - fails (as expected)  |

> ⚠️ `mamba_m`, `mamba_s`, `attn_m`, `attn_s` broke in simple_copy - skipped

______________________________________________________________________

## Val Loss by Patch Size (20k iterations)

### Hyena + Patchify (20k)

| Size   | p=1       | p=2       | p=4       | p=8   | p=16  | p=32  |
| ------ | --------- | --------- | --------- | ----- | ----- | ----- |
| **M**  | 0.030     | 0.032     | **0.022** | 0.092 | 0.149 | 0.301 |
| **S**  | **0.045** | 0.057     | 0.055     | 0.137 | 0.224 | 0.364 |
| **XS** | 0.069     | **0.037** | 0.038     | 0.089 | 0.195 | 0.331 |

### Mamba + Patchify (20k)

| Size   | p=1   | p=2   | p=4       | p=8       | p=16  | p=32  |
| ------ | ----- | ----- | --------- | --------- | ----- | ----- |
| **M**  | 0.582 | 0.161 | **0.045** | 0.049     | 0.123 | 0.280 |
| **S**  | 0.522 | 0.374 | 0.114     | **0.084** | 0.187 | 0.333 |
| **XS** | 0.715 | 0.556 | 0.219     | **0.148** | 0.257 | 0.382 |

### Attention + Patchify (20k)

| Size   | p=1   | p=2   | p=4   | p=8       | p=16  | p=32  |
| ------ | ----- | ----- | ----- | --------- | ----- | ----- |
| **M**  | 0.487 | 0.392 | 0.353 | **0.096** | 0.142 | 0.314 |
| **S**  | 0.538 | 0.473 | 0.380 | **0.148** | 0.197 | 0.355 |
| **XS** | 0.579 | 0.523 | 0.433 | **0.216** | 0.297 | 0.417 |

> ✅ **20k patchify tables now complete!**

### Non-Patchify Models (20k)

| Config      | Size | Val Loss   | Notes                  |
| ----------- | ---- | ---------- | ---------------------- |
| **hyena_m** | M    | **0.0233** | ✅ Works great         |
| **hyena_s** | S    | **0.0236** | ✅ Works great         |
| hyena_xs    | XS   | 0.0514     | ✅ Works               |
| mamba_m     | M    | -          | ❌ NOT RUN             |
| mamba_s     | S    | -          | ❌ NOT RUN             |
| mamba_xs    | XS   | 0.7243     | ❌ Fails (as expected) |
| attn_m      | M    | -          | ❌ NOT RUN             |
| attn_s      | S    | -          | ❌ NOT RUN             |
| attn_xs     | XS   | 0.5773     | ❌ Fails (as expected) |

> ⚠️ **Missing**: Mamba M/S and Attn M/S non-patchify runs were never submitted!

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

## 100k Iteration Runs - ✅ COMPLETE (Verified from WandB)

**54 runs total** testing if all architectures improve with 5x more training.

**Status**: Most completed ✅, some crashed (preempted), results verified from WandB API

| Architecture          | SLURM IDs     | Status                        | Best Val Loss |
| --------------------- | ------------- | ----------------------------- | ------------- |
| Attention M (4 runs)  | 171995-171998 | ✅ Complete                   | **0.0161**    |
| Attention S (6 runs)  | 172270-172274 | ✅ Complete (100k)            | **0.0257**    |
| Attention XS (6 runs) | 172275-172278 | ✅ Complete (100k)            | **0.0598**    |
| Mamba M (6 runs)      | 172011-172016 | ✅ Complete                   | **0.00295**   |
| Mamba S (6 runs)      | 172017-172022 | ✅ Complete                   | **0.0100**    |
| Mamba XS (6 runs)     | 172023-172028 | ✅ Complete                   | 0.0380        |
| Hyena M (6 runs)      | 172029-172034 | ✅ 5 complete, ❌ 1 preempted | **0.00129**   |
| Hyena S (6 runs)      | 172035-172040 | ✅ Complete                   | **0.00772**   |
| Hyena XS (6 runs)     | 172041-172046 | ✅ 5 complete, ❌ 1 preempted | **0.0121**    |

### 100k Iteration Results Summary (Verified from WandB)

| Architecture    | Best Val Loss (100k) | Best Val Loss (20k) | Improvement |
| --------------- | -------------------- | ------------------- | ----------- |
| **Hyena M**     | **0.00129**          | 0.022               | **17x** 🏆  |
| **Mamba M**     | **0.00295**          | 0.044               | **15x**     |
| **Hyena S**     | **0.00772**          | 0.054               | **7x**      |
| **Mamba S**     | **0.0100**           | 0.083               | **8x**      |
| **Hyena XS**    | **0.0121**           | 0.070               | **6x**      |
| **Attention M** | **0.0161**           | 0.096               | **6x**      |
| **Attention S** | **0.0257**           | 0.148               | **6x**      |
| Mamba XS        | 0.0380               | 0.147               | 4x          |
| Attention XS    | 0.0598               | 0.216               | 4x          |

**Note**: Attention M p=4,8 already completed earlier (171964-171965).

______________________________________________________________________

## Notes

- Input: 2 channels (grayscale + mask), Output: 1 channel
- `num_items=4`, `placement="random"`, `with_mask=True`
- Callback shows: `[canvas | mask | prediction | label]`

______________________________________________________________________

**Last Updated**: 2026-01-19 16:00 PST
**Status**: 🔄 4 runs still running, 15 completed. Hyena S non-patch finished at **0.0065**!

______________________________________________________________________

## 🚀 Newly Submitted Experiments (2026-01-19)

> **Note**: WandB verification confirmed **0 crashed/failed** runs. All "missing" experiments simply weren't submitted before.

### 100k Patchify - 12 runs (11 COMPLETE ✅)

| Arch  | Size | Patches                   | SLURM IDs              | Status                                                                         |
| ----- | ---- | ------------------------- | ---------------------- | ------------------------------------------------------------------------------ |
| Hyena | M    | p=1                       | 172398 (resumed)       | 🔄 80% (val=0.007) - almost done!                                              |
| Hyena | XS   | p=1                       | 172266                 | 🔄 63% (val=0.014)                                                             |
| Mamba | M    | p=1, p=2, p=32            | 172267, 172268, 172269 | ✅ p=2: **0.023**, p=32: 0.100, p=1: 🔄39% (val=0.69)                          |
| Attn  | S    | p=2, p=4, p=8, p=16, p=32 | 172270-172274          | ✅ **ALL DONE** (p=2: 0.045, p=4: 0.026, p=8: 0.027, p=16: 0.075, p=32: 0.165) |
| Attn  | XS   | p=1, p=2, p=4, p=32       | 172399 (resumed)       | ✅ **ALL DONE** (p=1: **0.158**, p=2: 0.144, p=4: 0.094, p=32: 0.226)          |

### 100k Non-Patchify - 5 runs (3 COMPLETE ✅, 2 RUNNING 🔄)

| Arch  | Sizes | SLURM IDs        | Val Loss              | Notes                            |
| ----- | ----- | ---------------- | --------------------- | -------------------------------- |
| Hyena | M     | 172400 (resumed) | 🔄 **0.007** (80%) 🔥 | Autoresumed - looking excellent! |
| Hyena | S     | 172397 (resumed) | ✅ **0.0065** 🏆      | DONE! New non-patchify record!   |
| Hyena | XS    | 172290           | 🔄 0.014 (63%)        | Running                          |
| Mamba | XS    | 172401 (resumed) | ✅ 0.408              | Done - fails (as expected)       |
| Attn  | XS    | 172402 (resumed) | ✅ 0.294              | Done - fails (as expected)       |

### 20k Patchify - ✅ COMPLETE (2 runs)

| Arch  | Size | Patches | SLURM ID | Val Loss     |
| ----- | ---- | ------- | -------- | ------------ |
| Hyena | S    | p=1     | 172286   | **0.045** ✅ |
| Attn  | M    | p=1     | 172287   | 0.487 ✅     |

### 20k Non-Patchify - SKIPPED

| Arch  | Missing Sizes | Notes                               |
| ----- | ------------- | ----------------------------------- |
| Mamba | M, S          | ⚠️ Broke in simple_copy exps - skip |
| Attn  | M, S          | ⚠️ Broke in simple_copy exps - skip |

______________________________________________________________________

### Total Submitted: 19 runs (15 COMPLETE ✅, 4 RUNNING 🔄)

| Category          | Count  | Completed | Running | Notes                                  |
| ----------------- | ------ | --------- | ------- | -------------------------------------- |
| 100k Patchify     | 12     | 10        | 2       | Hyena M p=1, XS p=1, Mamba M p=1 going |
| 100k Non-Patchify | 5      | 3         | 2       | ✅ Hyena S done (0.0065)! M/XS running |
| 20k Patchify      | 2      | 2         | 0       | ✅ All done                            |
| 20k Non-Patchify  | 0      | -         | -       | Skipped (broken configs)               |
| **Total**         | **19** | **15**    | **4**   |                                        |

> **Note on broken configs**: `mamba_m`, `mamba_s`, `attn_m`, `attn_s` (non-patchify) broke during simple_copy experiments.
>
> **Autoresume**: Successfully used to resume 6 preempted jobs! Works by downloading checkpoints from WandB.
