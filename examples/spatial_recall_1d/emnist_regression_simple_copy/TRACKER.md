# Spatial Recall 1D - EMNIST Simple Copy - Experiment Tracker

## Task Description

1D version of spatial recall where:

- Images are flattened FIRST (16×16 → 256 elements)
- Flattened image placed as contiguous segment in 1D canvas (4096 elements)
- Model must recall the flattened image from a **causal** perspective

Key difference from 2D: Models are **causal** (can only see past, not future).

## Model Configurations

### XS (Extra-Small) Models

| Model     | Hidden Dim | Heads/Headdim          | Params | Notes                                |
| --------- | ---------- | ---------------------- | ------ | ------------------------------------ |
| Attention | 160        | 8 heads (head_dim=20)  | ~719K  | Causal attention with RoPE           |
| Mamba     | 128        | headdim=32, expand=2   | ~738K  | Unidirectional (bidirectional=False) |
| Hyena     | 160        | SIREN kernel, 3 layers | ~757K  | Causal CKConvND + CausalConv1D       |

**Note**: Mamba hidden_dim must be multiple of 16 for Mamba2 compatibility.

## Dataset Configuration

- **Target size**: 16×16 (flattened to 256)
- **Canvas size**: 64×64 (flattened to 4096)
- **Placement**: fixed
- **Num items**: 1
- **With mask**: False

## Experiments

### Spatial Recall (Regression) Experiments

| Job ID | Model        | readout_value | Status    | Step | W&B ID     | Notes                      |
| ------ | ------------ | ------------- | --------- | ---- | ---------- | -------------------------- |
| 172326 | Attention XS | 0.0           | Running   | 94k  | -          | 100k iters                 |
| 172328 | Attention XS | -1.0          | Running   | 94k  | -          | 100k iters                 |
| 172337 | Mamba XS     | 0.0           | Finished  | 100k | `hyjeexlp` | 100k iters                 |
| 172338 | Mamba XS     | -1.0          | Preempted | 62k  | `tjzfkevb` | Crashed → 172466           |
| 172466 | Mamba XS     | -1.0          | Running   | 62k+ | `tjzfkevb` | Autoresume (low partition) |
| 172413 | Hyena XS     | 0.0           | Running   | 52k  | -          | 100k iters                 |
| 172414 | Hyena XS     | -1.0          | Running   | 52k  | -          | 100k iters                 |

### L_cache Ablation (Hyena with L_cache=64 instead of 4096)

| Job ID | Model                 | readout_value | Status  | Step | W&B ID     | Val Loss        | Notes             |
| ------ | --------------------- | ------------- | ------- | ---- | ---------- | --------------- | ----------------- |
| 172440 | Hyena XS (L_cache=64) | 0.0           | Stopped | 22k  | `g5d21isc` | **0.000056** 🏆 | Converged early!  |
| 172441 | Hyena XS (L_cache=64) | -1.0          | Stopped | 22k  | `ixgb2eg6` | **0.000047** 🏆 | Best result ever! |

### Attention rope_base Sweep (testing frequency prior effect)

| Job ID | Model        | rope_base | Status   | Step | W&B ID     | Val Loss   | Notes                      |
| ------ | ------------ | --------- | -------- | ---- | ---------- | ---------- | -------------------------- |
| 172833 | Attention XS | 0.01      | Finished | 20k  | `8216lyby` | 0.2096     | Too fast decay             |
| 172832 | Attention XS | 0.1       | Finished | 20k  | `dc8tmo2s` | 0.2069     | Too fast decay             |
| 172831 | Attention XS | 1         | Finished | 20k  | `d6z5ject` | 0.8276     | Way too fast decay         |
| 172473 | Attention XS | 10        | Finished | 20k  | `mcgrp0om` | **0.0900** | 🥇 Best!                   |
| 172474 | Attention XS | 64        | Finished | 20k  | `jmfgbuox` | 0.1249     | Surprisingly worse than 10 |
| 172475 | Attention XS | 100       | Finished | 20k  | `9xg244pz` | 0.0989     | 🥈 Second best             |
| 172476 | Attention XS | 1000      | Finished | 20k  | `3lfz523c` | 0.3146     | Poor                       |
| 172477 | Attention XS | 10000     | Finished | 20k  | `bru97vti` | 0.1653     | Default baseline           |
| 172478 | Attention XS | 100000    | Finished | 20k  | `cythhwqs` | 0.3156     | Worst - too slow decay     |

**Sweep conclusion**:

- **rope_base=10 is optimal** (0.0900), with rope_base=100 close second (0.0989)
- **Too low** (≤1): Position encodings decay too fast → loss ~0.2-0.8
- **Too high** (≥1000): Position encodings change too slowly → loss ~0.3
- Still **~1900x worse** than Hyena L_cache=64 (0.000047)!

### Mamba Frequency/Memory Parameter Sweep

Testing whether Mamba's analogous frequency parameters can improve performance:

- **A_init_range**: Controls A matrix eigenvalues (decay rate). Smaller = slower decay = longer memory.
- **dt_min/dt_max**: Controls discretization step. Smaller = finer resolution = slower effective decay.
- **Learning rate**: Standard hyperparameter sweep.

Config: `ccnn_mamba_causal_xs_long_memory.py` (modify constants for each run)

**Sweep Design** (12 runs):

| Run | A_init_range | dt_min | dt_max | lr   | Description                    | Job ID | Status   | Val Loss      |
| --- | ------------ | ------ | ------ | ---- | ------------------------------ | ------ | -------- | ------------- |
| 1   | (0.1, 1)     | 0.0001 | 0.01   | 1e-4 | Very slow decay + fine dt      | 172562 | Finished | 0.7692        |
| 2   | (0.5, 4)     | 0.0001 | 0.01   | 1e-4 | Slow decay + fine dt           | 172563 | Finished | 0.7681        |
| 3   | (1, 16)      | 0.0001 | 0.01   | 1e-4 | Default A + fine dt            | 172564 | Finished | 0.7982        |
| 4   | (0.1, 1)     | 0.0001 | 0.001  | 1e-4 | Very slow decay + very fine dt | 172565 | Finished | 0.7693        |
| 5   | (0.5, 4)     | 0.0001 | 0.001  | 1e-4 | Slow decay + very fine dt      | 172566 | Finished | **0.7440** 🥇 |
| 6   | (1, 16)      | 0.001  | 0.1    | 1e-4 | Default (baseline)             | 172567 | Finished | 0.8841        |
| 7   | (0.5, 4)     | 0.0001 | 0.01   | 1e-3 | Long memory + high lr          | 172568 | Finished | 0.7828        |
| 8   | (0.5, 4)     | 0.0001 | 0.01   | 1e-5 | Long memory + low lr           | 172569 | Finished | 0.7695        |
| 9   | (1, 16)      | 0.001  | 0.1    | 1e-3 | Default + high lr              | 172570 | Finished | 0.7983        |
| 10  | (1, 16)      | 0.001  | 0.1    | 1e-5 | Default + low lr               | 172571 | Finished | 0.8842        |
| 11  | (0.1, 1)     | 0.001  | 0.1    | 1e-4 | Very slow decay + default dt   | 172572 | Finished | 0.8131        |
| 12  | (0.5, 4)     | 0.001  | 0.1    | 1e-4 | Slow decay + default dt        | 172573 | Finished | 0.7946        |

**Baseline comparison**: Mamba XS default (100k iters) = 0.719 val loss
**Target to beat**: Hyena L_cache=64 (22k iters) = 0.000047 val loss (~15,000x gap!)

**Hypothesis**: If Mamba's frequency parameters are analogous to Hyena's L_cache, we should see significant improvement with longer memory settings.

#### Sweep with readout_value=-1.0

Same sweep as above but with `readout_value=-1.0` (explicit readout marker):

| Run | A_init_range | dt_min | dt_max | lr   | Description                    | Job ID | Status   | Val Loss      |
| --- | ------------ | ------ | ------ | ---- | ------------------------------ | ------ | -------- | ------------- |
| 1   | (0.1, 1)     | 0.0001 | 0.01   | 1e-4 | Very slow decay + fine dt      | 172574 | Finished | 0.3582        |
| 2   | (0.5, 4)     | 0.0001 | 0.01   | 1e-4 | Slow decay + fine dt           | 172575 | Finished | 0.3453        |
| 3   | (1, 16)      | 0.0001 | 0.01   | 1e-4 | Default A + fine dt            | 172576 | Finished | 0.3138        |
| 4   | (0.1, 1)     | 0.0001 | 0.001  | 1e-4 | Very slow decay + very fine dt | 172577 | Finished | 0.3863        |
| 5   | (0.5, 4)     | 0.0001 | 0.001  | 1e-4 | Slow decay + very fine dt      | 172578 | Finished | 0.4096        |
| 6   | (1, 16)      | 0.001  | 0.1    | 1e-4 | Default (baseline)             | 172579 | Finished | 0.6873        |
| 7   | (0.5, 4)     | 0.0001 | 0.01   | 1e-3 | Long memory + high lr          | 172580 | Finished | **0.1612** 🏆 |
| 8   | (0.5, 4)     | 0.0001 | 0.01   | 1e-5 | Long memory + low lr           | 172581 | Finished | 0.6944        |
| 9   | (1, 16)      | 0.001  | 0.1    | 1e-3 | Default + high lr              | 172582 | Finished | **0.2122** 🥈 |
| 10  | (1, 16)      | 0.001  | 0.1    | 1e-5 | Default + low lr               | 172583 | Finished | 0.7958        |
| 11  | (0.1, 1)     | 0.001  | 0.1    | 1e-4 | Very slow decay + default dt   | 172584 | Finished | 0.3410        |
| 12  | (0.5, 4)     | 0.001  | 0.1    | 1e-4 | Slow decay + default dt        | 172585 | Finished | 0.3475        |

**Sweep Conclusions:**

- 🏆 **Best (readout=-1.0)**: A_init=(0.5,4), dt=(0.0001,0.01), **lr=1e-3** → **0.1612**
- 🥈 **2nd best**: A_init=(1,16), dt=(0.001,0.1), **lr=1e-3** → **0.2122**
- 🥉 **Best (readout=0.0)**: A_init=(0.5,4), dt=(0.0001,0.001), lr=1e-4 → **0.7440**
- **Higher lr=1e-3 is critical** for readout=-1.0 setting!
- **readout=-1.0 + higher lr gives 4.6x improvement** (0.1612 vs 0.7440)
- **Fine dt_max=0.01 helps** but lr matters more with readout=-1.0
- ⚠️ **Still ~3,400x worse than Hyena L_cache=64 (0.000047)**
- **Conclusion**: Mamba frequency params + higher lr help significantly but still DON'T close the gap to Hyena!

#### Extended Training (100k) - Best Config

| Job ID | Config                                           | Status   | Step | Val Loss   | Notes                                    |
| ------ | ------------------------------------------------ | -------- | ---- | ---------- | ---------------------------------------- |
| 172596 | A=(0.5,4), dt=(0.0001,0.01), lr=1e-3, readout=-1 | Finished | 100k | **0.0145** | Improved from 0.1612 (20k) to 0.0145! 🎉 |

### Patchification Sweep (patch_size vs sequence length tradeoff)

Testing how patch size affects model performance. With patchification:

- Canvas: 4096 elements → `4096 / patch_size` tokens
- Each patch captures local structure, attention/convolution operates on patch embeddings

**Configs:**

- `ccnn_mamba_causal_xs_patchify.py`
- `ccnn_attn_causal_xs_patchify.py`
- `ccnn_hyena_causal_xs_lcache64_patchify.py` (L_cache = 4096 / patch_size)

#### Mamba Patchify Sweep

| Job ID | patch_size | num_tokens | Status   | Val Loss   | Notes         |
| ------ | ---------- | ---------- | -------- | ---------- | ------------- |
| 172628 | 2          | 2048       | Finished | 0.801      | Poor          |
| 172629 | 4          | 1024       | Finished | 0.789      | Poor          |
| 172630 | 8          | 512        | Finished | 0.617      | Bad           |
| 172631 | 16         | 256        | Finished | 0.238      | Moderate      |
| 172632 | 32         | 128        | Finished | 0.034      | Good          |
| 172633 | 64         | 64         | Finished | 0.098      | Worse than 32 |
| 172634 | 128        | 32         | Finished | 0.019      | Very good     |
| 172635 | 256        | 16         | Finished | 0.020      | Very good     |
| 172636 | 512        | 8          | Finished | 0.021      | Very good     |
| 172637 | 1024       | 4          | Finished | 0.015      |               |
| 172756 | 2048       | 2          | Finished | 0.0155     |               |
| 172757 | 4096       | 1          | Finished | **0.0120** | 🥇 Best Mamba |

#### Attention Patchify Sweep

| Job ID | patch_size | num_tokens | Status   | Val Loss   | Notes             |
| ------ | ---------- | ---------- | -------- | ---------- | ----------------- |
| 172638 | 2          | 2048       | Finished | 0.343      | Moderate          |
| 172639 | 4          | 1024       | Finished | 0.310      | Moderate          |
| 172640 | 8          | 512        | Finished | 0.094      | Good              |
| 172641 | 16         | 256        | Finished | 0.034      | Very good         |
| 172642 | 32         | 128        | Finished | 0.057      | Worse than 16     |
| 172643 | 64         | 64         | Finished | 0.024      | Very good         |
| 172644 | 128        | 32         | Finished | 0.011      | Excellent         |
| 172645 | 256        | 16         | Finished | 0.0076     | Excellent         |
| 172646 | 512        | 8          | Finished | 0.0070     | Excellent         |
| 172647 | 1024       | 4          | Finished | 0.0064     |                   |
| 172758 | 2048       | 2          | Finished | **0.0060** | 🥇 Best Attention |
| 172759 | 4096       | 1          | Finished | 0.0073     | Worse than 2048   |

#### Hyena Patchify Sweep (L_cache = 64 constant)

| Job ID | patch_size | num_tokens | L_cache | Status   | Val Loss     | Notes                     |
| ------ | ---------- | ---------- | ------- | -------- | ------------ | ------------------------- |
| 172658 | 2          | 2048       | 64      | Finished | 8.56e-05     | 🏆 Excellent!             |
| 172659 | 4          | 1024       | 64      | Finished | **7.13e-05** | 🏆🏆🏆 BEST OVERALL!      |
| 172660 | 8          | 512        | 64      | Finished | 7.81e-05     | 🏆 Excellent!             |
| 172661 | 16         | 256        | 64      | Finished | 1.14e-04     | 🥈 Very good              |
| 172662 | 32         | 128        | 64      | Finished | 2.73e-04     | Good                      |
| 172663 | 64         | 64         | 64      | Finished | 3.78e-04     | Good (seq = L_cache)      |
| 172664 | 128        | 32         | 64      | Finished | 0.00297      | Moderate (seq \< L)       |
| 172665 | 256        | 16         | 64      | Finished | 0.00908      | Poor (seq \<\< L)         |
| 172666 | 512        | 8          | 64      | Finished | 0.00798      | Poor (seq \<\< L)         |
| 172667 | 1024       | 4          | 64      | Finished | 0.00731      | Poor (seq \<\< L)         |
| 172760 | 2048       | 2          | 64      | Finished | 0.0080       | seq \<\< L                |
| 172761 | 4096       | 1          | 64      | Finished | 0.0033       | seq \<\< L (1 token only) |

**Sweep Conclusions (all 36 runs completed ✅):**

- 🏆 **Hyena patch=4 (1024 tokens) = 7.13e-05** - BEST OVERALL by ~85x!
- Hyena works GREAT even when seq >> L_cache (2048 tokens with L_cache=64)!
- **Longer sequences (more tokens) = BETTER** for Hyena!
- Performance degrades when seq \<\< L_cache (too few tokens)
- **Attention**: Best at patch=2048 (2 tokens) → 0.0060
- **Mamba**: Best at patch=4096 (1 token) → 0.0120
- Attention/Mamba improve with larger patches (fewer tokens) - essentially learning patch→output mapping
- **L_cache=64 is sufficient for sequences up to 2048 tokens!**

**Final Rankings:**

| Rank | Model     | Best Config           | Val Loss     |
| ---- | --------- | --------------------- | ------------ |
| 🥇   | Hyena     | patch=4 (1024 tokens) | **7.13e-05** |
| 🥈   | Attention | patch=2048 (2 tokens) | 0.0060       |
| 🥉   | Mamba     | patch=4096 (1 token)  | 0.0120       |

### Autoregressive Pretraining Experiments

| Job ID | Model                 | readout_value | Status   | Step | W&B Run    | Notes     |
| ------ | --------------------- | ------------- | -------- | ---- | ---------- | --------- |
| 172343 | Attention XS Pretrain | 0.0           | Finished | 20k  | `iefl9ab8` | 20k iters |
| 172351 | Mamba XS Pretrain     | 0.0           | Finished | 20k  | `q1wklbij` | 20k iters |
| 172352 | Attention XS Pretrain | -1.0          | Finished | 20k  | `n328nxa7` | 20k iters |
| 172353 | Mamba XS Pretrain     | -1.0          | Finished | 20k  | `aepgk6og` | 20k iters |
| 172415 | Hyena XS Pretrain     | 0.0           | Finished | 20k  | `hm8s0n2l` | 20k iters |
| 172419 | Hyena XS Pretrain     | -1.0          | Finished | 20k  | `sqab1n22` | 20k iters |

### Fine-tuning from Pretrained Checkpoints

| Job ID | Model        | readout_value | Pretrain Run | Status    | Step | W&B ID     | Notes                      |
| ------ | ------------ | ------------- | ------------ | --------- | ---- | ---------- | -------------------------- |
| 172373 | Mamba XS     | 0.0           | `q1wklbij`   | Preempted | 69k  | `dhgdwj6e` | Crashed → 172462           |
| 172374 | Attention XS | 0.0           | `iefl9ab8`   | Preempted | 46k  | `041vtk2s` | Crashed → 172463           |
| 172377 | Mamba XS     | -1.0          | `aepgk6og`   | Preempted | 69k  | `dthsb3u4` | Crashed → 172464           |
| 172378 | Attention XS | -1.0          | `n328nxa7`   | Preempted | 44k  | `aiy0e5xa` | Crashed → 172465           |
| 172462 | Mamba XS     | 0.0           | `q1wklbij`   | Running   | 70k+ | `dhgdwj6e` | Autoresume (low partition) |
| 172463 | Attention XS | 0.0           | `iefl9ab8`   | Running   | 46k+ | `041vtk2s` | Autoresume (low partition) |
| 172464 | Mamba XS     | -1.0          | `aepgk6og`   | Running   | 69k+ | `dthsb3u4` | Autoresume (low partition) |
| 172465 | Attention XS | -1.0          | `n328nxa7`   | Running   | 44k+ | `aiy0e5xa` | Autoresume (low partition) |

### Experiment Variants

1. **readout_value=0.0** (default): Readout region filled with zeros
1. **readout_value=-1.0**: Readout region explicitly marked with -1, so model knows where to output
1. **Fine-tuning from pretrain**: Start from AR-pretrained weights and fine-tune on recall task (80k iters)

## Results

### Spatial Recall Results (in progress)

| Model                     | readout_value | Val Loss        | Step | Notes                              |
| ------------------------- | ------------- | --------------- | ---- | ---------------------------------- |
| **Hyena XS (L_cache=64)** | -1.0          | **0.000047** 🏆 | 22k  | BEST! ~1900x better than Attn!     |
| **Hyena XS (L_cache=64)** | 0.0           | **0.000056** 🏆 | 22k  | Also incredible!                   |
| Attention XS              | 0.0           | 0.061           | 100k | Default rope_base=10000            |
| Attention XS              | -1.0          | 0.068           | 100k | Default rope_base=10000            |
| Attention XS (rope=10)    | 0.0           | 0.090           | 20k  | Best rope_base, still ~1900x worse |
| Hyena XS                  | 0.0           | 0.110           | 52k  | Still training (L_cache=4096)      |
| Hyena XS                  | -1.0          | 0.154           | 52k  | Still training (L_cache=4096)      |
| Mamba XS                  | 0.0           | 0.719           | 100k | Finished, poor                     |
| Mamba XS                  | -1.0          | 0.193           | 62k  | Crashed, readout helped            |

**Observations:**

- 🏆🏆🏆 **Hyena L_cache=64 DOMINATES** with val loss ~0.00005 vs Attention's 0.061 (~1000x better!)
- L_cache=64 means SIREN uses coarser frequency grid (step_size=1/63 vs 1/4095)
- **Attention rope_base sweep**: Lower values help (10→0.090 best), but still ~1900x worse than Hyena
- **Key insight**: The L_cache effect in Hyena is NOT equivalent to rope_base in Attention!
  - Hyena L_cache affects SIREN kernel's positional grid resolution + output initialization
  - Attention rope_base only affects RoPE frequency decay rate
  - These are fundamentally different mechanisms

### Autoregressive Pretraining Results

| Model                 | readout_value | Val Loss       | Step | Notes                |
| --------------------- | ------------- | -------------- | ---- | -------------------- |
| **Hyena XS Pretrain** | 0.0           | **0.00324** 🏆 | 20k  | Best AR performance! |
| Mamba XS Pretrain     | 0.0           | 0.00335        | 20k  | Second best          |
| Mamba XS Pretrain     | -1.0          | 0.00361        | 20k  | Also excellent       |
| Hyena XS Pretrain     | -1.0          | 0.00370        | 20k  | Very good            |
| Attention XS Pretrain | 0.0           | 0.00644        | 20k  | Good AR modeling     |
| Attention XS Pretrain | -1.0          | 0.00692        | 20k  | Slightly worse       |

**Observations:**

- 🏆 **Hyena beats Mamba** on AR pretraining! (0.00324 vs 0.00335)
- All long-conv models (Hyena, Mamba) outperform Attention on AR
- Interesting contrast: Mamba is good at AR but struggles with recall

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
