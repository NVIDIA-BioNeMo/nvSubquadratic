# Experiment Tracker - EMNIST Spatial Recall 2D (Color Selection)

## Experiment Overview

**Task**: EMNIST Spatial Recall 2D Regression with Color Selection

- **Input**: 64×64 canvas with 4 EMNIST digits in RGB (3 channels) with colored bounding boxes
- **Target**: 16×16 region containing the digit with the special color (e.g., red)
- **Objective**: Regress the target region for the digit indicated by its colored frame

______________________________________________________________________

## Key Differences from Mask Selection

| Aspect         | Mask Selection         | Color Selection                 |
| -------------- | ---------------------- | ------------------------------- |
| **Input**      | 2ch (grayscale + mask) | 3ch (RGB with colored frames)   |
| **Selection**  | Binary mask overlay    | Colored bounding box            |
| **Hypothesis** | Explicit spatial cue   | Must learn color discrimination |

______________________________________________________________________

## Configs Available

| Config                   | Architecture | Patchify | Size | Status  |
| ------------------------ | ------------ | -------- | ---- | ------- |
| ccnn_hyena_xs            | Hyena        | No       | XS   | Pending |
| ccnn_hyena_patchify_xs   | Hyena        | Yes      | XS   | Pending |
| ccnn_mamba_xs            | Mamba        | No       | XS   | Pending |
| ccnn_mamba_patchify_xs   | Mamba        | Yes      | XS   | Pending |
| ccnn_attn_xs             | Attention    | No       | XS   | Pending |
| ccnn_attn_patchify_xs    | Attention    | Yes      | XS   | Pending |
| ccnn_4_160_attn_patchify | Attention    | Yes      | M    | Pending |
| ccnn_4_160_attn          | Attention    | No       | M    | Pending |
| ccnn_4_160_hyena         | Hyena        | No       | M    | Pending |
| ccnn_4_160_mamba         | Mamba        | No       | M    | Pending |

______________________________________________________________________

## Results

### 🏆 TOP 5 RESULTS (50k iterations) - Updated with S/M!

| Rank | Config       | Size | Val Loss      | WandB ID |
| ---- | ------------ | ---- | ------------- | -------- |
| 🥇   | Hyena p=2    | M    | **0.0059** 🔥 | pf43ievh |
| 🥈   | Hyena p=2    | S    | **0.0062**    | yr9zl62j |
| 🥉   | Mamba p=4    | M    | **0.0087**    | akz7yi5g |
| 4    | Hyena (none) | XS   | 0.0132        | ur6zpbpu |
| 5    | Mamba p=4    | S    | 0.0135        | l01vjfp5 |

### XS Non-Patchify (50k iterations) ✅ COMPLETE

| Config        | Val Loss      | WandB Run | Notes                  |
| ------------- | ------------- | --------- | ---------------------- |
| ccnn_hyena_xs | **0.0132** 🏆 | ur6zpbpu  | Best overall!          |
| ccnn_mamba_xs | 0.0980        | dop4g17e  | Works                  |
| ccnn_attn_xs  | 0.6413        | chl2f1u3  | ❌ Fails (no learning) |

### XS Patchify by Patch Size (50k iterations) ✅ COMPLETE

#### Val Loss Summary Table

| Architecture  | p=2      | p=4      | p=8  | p=16 | p=32 |
| ------------- | -------- | -------- | ---- | ---- | ---- |
| **Hyena**     | **0.02** | **0.03** | 0.13 | 0.51 | 0.56 |
| **Mamba**     | 0.34     | **0.04** | 0.17 | 0.39 | 0.52 |
| **Attention** | 0.63     | 0.57     | 0.50 | 0.37 | 0.55 |

#### Hyena + Patchify XS

| Patch Size | Val Loss      | WandB Run | Notes     |
| ---------- | ------------- | --------- | --------- |
| p=2        | **0.0199** 🥈 | pf8yhnbp  | Excellent |
| p=4        | **0.0340** 🥉 | p7d94p3e  | Good      |
| p=8        | 0.1279        | 3ts8bnta  | Degraded  |
| p=16       | 0.5058        | qjbzf84o  | ❌ Fails  |
| p=32       | 0.5616        | m4rhytlk  | ❌ Fails  |

#### Mamba + Patchify XS

| Patch Size | Val Loss   | WandB Run | Notes       |
| ---------- | ---------- | --------- | ----------- |
| p=2        | 0.3416     | qmfu73ai  | Poor        |
| p=4        | **0.0354** | 2hfypq1r  | Best Mamba! |
| p=8        | 0.1669     | 7idl1gn1  | Degraded    |
| p=16       | 0.3934     | 5wffu5x7  | ❌ Fails    |
| p=32       | 0.5188     | s7dqfn24  | ❌ Fails    |

#### Attention + Patchify XS

| Patch Size | Val Loss | WandB Run | Notes                |
| ---------- | -------- | --------- | -------------------- |
| p=2        | 0.6347   | pybpc6ka  | ❌ Fails             |
| p=4        | 0.5727   | 2w78r3xw  | ❌ Fails             |
| p=8        | 0.4974   | s9fge1m5  | ❌ Fails             |
| p=16       | 0.3664   | y4kwfvsu  | ❌ Fails (best attn) |
| p=32       | 0.5482   | qc06voak  | ❌ Fails             |

______________________________________________________________________

## S/M Size Experiments (50k iterations) 🚀 Running Full Sweep

### Val Loss by Patch Size - S Size ✅ COMPLETE

| Architecture | none  | p=2       | p=4       | p=8   | p=16 | p=32 |
| ------------ | ----- | --------- | --------- | ----- | ---- | ---- |
| **Hyena S**  | 0.063 | **0.006** | 0.017     | 0.071 | 0.35 | 0.51 |
| **Mamba S**  | -     | 0.047     | **0.014** | 0.065 | 0.23 | 0.46 |
| **Attn S**   | -     | 0.63      | 0.52      | 0.25  | 0.27 | 0.51 |

### Val Loss by Patch Size - M Size ✅ COMPLETE

| Architecture | none         | p=2          | p=4       | p=8       | p=16 | p=32 |
| ------------ | ------------ | ------------ | --------- | --------- | ---- | ---- |
| **Hyena M**  | **0.010** 🔥 | **0.006** 🏆 | 0.018     | 0.063     | 0.24 | 0.45 |
| **Mamba M**  | -            | 0.015        | **0.009** | **0.036** | 0.23 | 0.40 |
| **Attn M**   | -            | 0.63         | 0.52      | 0.15      | 0.19 | 0.48 |

### WandB IDs - S Size

| Arch  | none     | p=2      | p=4      | p=8      | p=16     | p=32     |
| ----- | -------- | -------- | -------- | -------- | -------- | -------- |
| Hyena | ccoo1800 | yr9zl62j | fb3owks9 | fifq5jhg | csh9526z | 7zeps95e |
| Mamba | -        | xo6wul5p | l01vjfp5 | lx3pvw85 | lo17njh2 | 0mc53joz |
| Attn  | -        | pjmo9zdr | 3qlqo0r3 | 58h7mh0l | 279b4gdp | u28gubvh |

### WandB IDs - M Size

| Arch  | none     | p=2      | p=4      | p=8      | p=16     | p=32     |
| ----- | -------- | -------- | -------- | -------- | -------- | -------- |
| Hyena | m1yn6ney | pf43ievh | bmks8yjg | 8ha2l9k5 | 2976aioe | fonzck2s |
| Mamba | -        | hjb8v4p7 | akz7yi5g | p252uo9s | wwzz0h6g | a95t8bjq |
| Attn  | -        | y5gx28qv | iknfbrd1 | o6x8u3ix | ht2jot6p | b4qnwk0b |

### Completed S/M Results (Sorted by Val Loss)

| Config       | Size | Val Loss      | WandB ID | Notes             |
| ------------ | ---- | ------------- | -------- | ----------------- |
| Hyena p=2    | M    | **0.0059** 🏆 | pf43ievh | BEST!             |
| Hyena p=2    | S    | **0.0062**    | yr9zl62j | Excellent         |
| Mamba p=4    | M    | **0.0087**    | akz7yi5g | Great             |
| Mamba p=4    | S    | 0.0135        | l01vjfp5 | Good              |
| Mamba p=2    | M    | 0.0146        | hjb8v4p7 | Good              |
| Hyena p=4    | S    | 0.0169        | fb3owks9 | Good              |
| Hyena p=4    | M    | 0.0183        | bmks8yjg | Good              |
| Mamba p=8    | M    | 0.0360        | p252uo9s | Good              |
| Mamba p=2    | S    | 0.0468        | xo6wul5p | OK                |
| Hyena p=8    | M    | 0.0629        | 8ha2l9k5 | OK                |
| Mamba p=8    | S    | 0.0647        | lx3pvw85 | OK                |
| Hyena p=8    | S    | 0.0711        | fifq5jhg | OK                |
| Hyena (none) | M    | **0.0098** 🔥 | m1yn6ney | ✅ Resumed & done |
| Attn p=8     | M    | 0.1505        | o6x8u3ix | Best Attn         |
| Hyena (none) | S    | 0.0633        | ccoo1800 | ✅ Resumed & done |
| Attn p=16    | M    | 0.1896        | ht2jot6p | ❌ Fails          |
| Mamba p=16   | S    | 0.2265        | lo17njh2 | ❌ Fails          |
| Mamba p=16   | M    | 0.2326        | wwzz0h6g | ❌ Fails          |
| Hyena p=16   | M    | 0.2370        | 2976aioe | ❌ Fails          |
| Attn p=8     | S    | 0.2454        | 58h7mh0l | ❌ Fails          |
| Attn p=16    | S    | 0.2702        | 279b4gdp | ❌ Fails          |
| Hyena p=16   | S    | 0.3544        | csh9526z | ❌ Fails          |
| Mamba p=32   | M    | 0.3989        | a95t8bjq | ❌ Fails          |
| Hyena p=32   | M    | 0.4506        | fonzck2s | ❌ Fails          |
| Mamba p=32   | S    | 0.4572        | 0mc53joz | ❌ Fails          |
| Attn p=32    | M    | 0.4778        | b4qnwk0b | ❌ Fails          |
| Attn p=32    | S    | 0.5075        | u28gubvh | ❌ Fails          |
| Hyena p=32   | S    | 0.5119        | 7zeps95e | ❌ Fails          |
| Attn p=4     | M    | 0.5196        | iknfbrd1 | ❌ Fails          |
| Attn p=4     | S    | 0.5223        | 3qlqo0r3 | ❌ Fails          |
| Attn p=2     | M    | 0.6279        | y5gx28qv | ❌ Fails          |
| Attn p=2     | S    | 0.6319        | pjmo9zdr | ❌ Fails          |

### Comparison: XS vs S vs M

| Config       | XS     | S      | M             | Trend |
| ------------ | ------ | ------ | ------------- | ----- |
| Hyena p=2    | 0.0199 | 0.0062 | **0.0059** 🏆 | 3x ⬆️ |
| Hyena p=4    | 0.0340 | 0.0169 | 0.0183        | 2x ⬆️ |
| Hyena p=8    | 0.1279 | 0.0711 | 0.0629        | 2x ⬆️ |
| Mamba p=4    | 0.0354 | 0.0135 | **0.0087**    | 4x ⬆️ |
| Mamba p=8    | 0.1669 | 0.0868 | 0.0562        | 3x ⬆️ |
| Hyena (none) | 0.0132 | 0.1722 | 0.0843        | Mixed |
| Attn p=8     | 0.4974 | 0.2454 | 0.1505        | 3x ⬆️ |

______________________________________________________________________

## Experiment Log

### XS Experiments - ✅ COMPLETE

- **Date**: 2026-01-21
- **SLURM IDs**: 173576-173600
- **Iterations**: 50k
- **Status**: ✅ All 18 Complete

### S/M Experiments - Full Sweep ✅

- **Date**: 2026-01-21
- **Initial SLURM IDs**: 173715-173729 (14 runs - selective)
- **Full Sweep SLURM IDs**: 173751-173768 (18 additional runs)
- **Iterations**: 50k
- **Total S/M runs**: 32 (all patch sizes for all architectures)
- **Status**: 20 Complete ✅, 10 Running 🏃, 2 Crashed ❌
- **Best Result**: Hyena M p=2 → **0.0059** 🏆

______________________________________________________________________

## Key Findings

1. **Larger models significantly improve results** 🚀

   - Hyena p=2: 0.020 (XS) → 0.006 (S) → **0.006** (M) = **3x better!**
   - Mamba p=4: 0.035 (XS) → 0.014 (S) → **0.009** (M) = **4x better!**

1. **Hyena M p=2 is the SOTA** 🏆

   - Achieves **0.0059** val_loss - best across all sizes
   - Patchify with p=2 outperforms non-patchify for Hyena

1. **Optimal patch sizes are architecture-dependent**:

   - **Hyena**: p=2 best (can handle longer sequences)
   - **Mamba**: p=4 best (needs shorter sequences than Hyena)
   - **Attention**: p=8 best but still fails

1. **Large patches (p=16, p=32) fail universally** ❌

   - All architectures degrade severely with p≥16
   - Sequence too short to capture spatial relationships

1. **Attention completely fails** ❌

   - Best: M size p=8 at 0.15 (not learning)
   - Color selection remains too hard for attention
   - Even with larger models (S/M), attention can't learn this task

1. **Non-patchify Hyena works well at larger sizes** ✅

   - XS: 0.0132, S: 0.0633, M: **0.0098** 🔥
   - Hyena M (none) is competitive with best patchified results!

______________________________________________________________________

## Notes

- Input: 3 channels (RGB with colored frames), Output: 1 channel (grayscale)
- `num_items=4`, `placement="random"`, `use_colored_frames=True`, `with_mask=False`
- Callback shows: `[canvas | prediction | label]`

______________________________________________________________________

## Experiment Log

- **2026-01-21**: Initial XS sweep (18 experiments) ✅ Complete
- **2026-01-21**: S/M sweep (32 experiments, 20k iter) ✅ Complete
  - Hyena S (ccoo1800) crashed → Resumed via autoresume → **0.0633** ✅
  - Hyena M (m1yn6ney) crashed → Resumed via autoresume → **0.0098** 🔥 ✅
- **2026-01-21**: S/M continuation (32 experiments, +100k iter) ⏳ Running
  - SLURM 173941, 173947, 173963-173992 (32 jobs)
  - Loading weights from completed 20k runs via `start_from_checkpoint`

### 100k Continuation Results (S/M) - Val Loss ✅ COMPLETE

| Architecture | Size | Patch | WandB ID | Val Loss      | Status |
| ------------ | ---- | ----- | -------- | ------------- | ------ |
| Attention    | M    | p=2   | rsjadnut | 0.5797        | ✅     |
| Attention    | M    | p=4   | 4yrbyydr | 0.0324        | ✅     |
| Attention    | M    | p=8   | chzkhmvx | **0.0224**    | ✅     |
| Attention    | M    | p=16  | spxlduvx | 0.0765        | ✅     |
| Attention    | M    | p=32  | bb3d9idp | 0.3166        | ✅     |
| Attention    | S    | p=2   | 7044xq9i | 0.6024        | ✅     |
| Attention    | S    | p=4   | 3x1ogqkq | 0.1544        | ✅     |
| Attention    | S    | p=8   | 9q3qxglq | **0.0428**    | ✅     |
| Attention    | S    | p=16  | 1wl9tht9 | 0.1245        | ✅     |
| Attention    | S    | p=32  | pkxgon7n | 0.3749        | ✅     |
| Hyena        | M    | none  | p5rgfkw0 | **0.0005** 🥈 | ✅     |
| Hyena        | M    | p=2   | oeu8fdxj | **0.0007**    | ✅     |
| Hyena        | M    | p=4   | o8xc9idy | 0.0035        | ✅     |
| Hyena        | M    | p=8   | mystxt1s | 0.0178        | ✅     |
| Hyena        | M    | p=16  | 28dw083a | 0.0609        | ✅     |
| Hyena        | M    | p=32  | gfan6x33 | 0.2340        | ✅     |
| Hyena        | S    | none  | 2wvdpgpl | **0.0049**    | ✅     |
| Hyena        | S    | p=2   | q3va47uf | **0.0006** 🥉 | ✅     |
| Hyena        | S    | p=4   | kuhxcjud | 0.0051        | ✅     |
| Hyena        | S    | p=8   | j0yamzsh | 0.0206        | ✅     |
| Hyena        | S    | p=16  | b7d8zchy | 0.1085        | ✅     |
| Hyena        | S    | p=32  | nmxq9opj | 0.3145        | ✅     |
| Mamba        | M    | p=2   | b7kx8v6j | **0.0012**    | ✅     |
| Mamba        | M    | p=4   | lyr7drz1 | **0.0004** 🏆 | ✅     |
| Mamba        | M    | p=8   | nr0adh64 | 0.0085        | ✅     |
| Mamba        | M    | p=16  | tu8duucg | 0.0666        | ✅     |
| Mamba        | M    | p=32  | wuiny4mo | 0.2283        | ✅     |
| Mamba        | S    | p=2   | 9dzcogjo | 0.0078        | ✅     |
| Mamba        | S    | p=4   | ydndsl3j | **0.0017**    | ✅     |
| Mamba        | S    | p=8   | f8rxovv4 | 0.0158        | ✅     |
| Mamba        | S    | p=16  | h7rrw8e3 | 0.0962        | ✅     |
| Mamba        | S    | p=32  | fs64vju5 | 0.2980        | ✅     |

### 🏆 Top 10 Results (100k) - FINAL

1. **Mamba M p=4: 0.000415** (lyr7drz1) 🏆 NEW BEST!
1. **Hyena M none: 0.000517** (p5rgfkw0) 🥈 Non-patchify works!
1. **Hyena S p=2: 0.000588** (q3va47uf) 🥉
1. **Hyena M p=2: 0.000731** (oeu8fdxj)
1. **Mamba M p=2: 0.001166** (b7kx8v6j)
1. **Mamba S p=4: 0.001677** (ydndsl3j)
1. **Hyena M p=4: 0.003526** (o8xc9idy)
1. **Hyena S none: 0.004910** (2wvdpgpl)
1. **Hyena S p=4: 0.005062** (kuhxcjud)
1. **Mamba S p=2: 0.007780** (9dzcogjo)

______________________________________________________________________

**Last Updated**: 2026-01-22
**Status**: ✅ ALL 100k RUNS COMPLETE! 32/32 finished
