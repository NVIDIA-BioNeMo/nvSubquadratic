# Spatial Recall v2 — Experiment Tracker

W&B project: [nvsubquadratic](https://wandb.ai/implicit-long-convs/nvsubquadratic)

> **Status**: All 1D/2D/3D runs complete for Hyena, Attention, and Mamba. Hyena Gaussian mask 2D ablation complete. Only remaining: 3D Attention color_cond no-patch rerun (compile=False).

______________________________________________________________________

## Overview

v2 modernisations (relative to v1):

- `torch.compile` with `max-autotune-no-cudagraphs` (Hyena/Attention) or disabled (Mamba)
- bf16-mixed precision
- RMSNorm / RMSNormChannelFirst (replaces LayerNorm)
- Hyena v2 architecture: SiLU+Sigmoid gates, L2 QK-norm, output norm
- Attention: precomputed RoPE buffers (compile-compatible)
- AdamW optimizer, cosine schedule with 5% warmup

## Model Configurations

All models use 4 blocks. Patched variants use `Patchify`/`Unpatchify` (Conv/ConvTranspose) as in/out projections.

| Model             | Hidden Dim | Params | Notes                                                   |
| ----------------- | ---------- | ------ | ------------------------------------------------------- |
| Hyena             | 256        | ~1.89M | QKV wrapper, SIREN kernel, `compile_compatible_fftconv` |
| Attention (1D/2D) | 256        | ~1.84M | 8 heads, head_dim=32, RoPE                              |
| Attention (3D)    | 240        | ~1.66M | 8 heads, head_dim=30 (30%6==0 for 3D RoPE)              |
| Mamba (unidir)    | 208        | ~1.80M | headdim=32, expand=2; causal 1D                         |
| Mamba (bidir)     | 160        | ~1.90M | headdim=32, expand=2; bidirectional 1D/2D/3D            |

> **Note**: `hidden_dim` must be a multiple of 16 for Mamba2 (`d_inner = expand * d_model` must be divisible by `headdim=32`). Original `hidden_dim=216` was invalid. Bidirectional doubles Mamba2 layers, so `hidden_dim` is reduced from 208 → 160 to match ~1.9M.

### Known `torch.compile` issues

- **Patchify stride bug (fixed)**: non-contiguous tensors from `rearrange` caused `convolution_backward` assertion in inductor. Fixed by adding `.contiguous()` in `Patchify`/`Unpatchify` forward methods.
- **3D Attention + RoPE**: head_dim must be divisible by 6. Fixed by using hidden_dim=240 (head_dim=30) for 3D Attention.
- **3D Attention color_cond OOM**: Non-patched 3D Attention (32768 tokens) OOM during `torch.compile` — inductor tried to allocate 512 GiB for padded matrix multiplication. Needs `compile=False` or patching to reduce sequence length.

______________________________________________________________________

## Results Matrix

Legend: **bold** = best per task/row, 🔄 = running, 🚫 = not yet run

**Random baselines** (MSE of predicting the dataset-mean label):

| Task                      | Baseline MSE | Notes                                                                            |
| ------------------------- | ------------ | -------------------------------------------------------------------------------- |
| simple_copy (1-ch)        | **0.549**    | Var(EMNIST 16×16 normalized digits)                                              |
| color_conditioning (3-ch) | **0.284**    | Var(digit × random palette color); lower because 2/3 channels are zero per color |

### 1D — No patches (50k iters, sequence length = 4096)

| Task        | Causality  | Hyena                                                                                     | Attention                                                                            | Mamba                                                                                     |
| ----------- | ---------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| simple_copy | causal     | **1.32e-5** [eugrgout](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/eugrgout) | 0.1281 [xpueooyc](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xpueooyc) | 0.739 [a31yf66b](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/a31yf66b)       |
| simple_copy | non-causal | **5.20e-5** [olvighai](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/olvighai) | 0.1504 [zzffh2hu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/zzffh2hu) | 0.0854 [mk8fhvgx](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/mk8fhvgx)      |
| color_cond  | causal     | **0.0629** [3kagx67x](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/3kagx67x)  | 0.2697 [ice2p8h9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ice2p8h9) | 0.205 [qvt7qysw](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/qvt7qysw)       |
| color_cond  | non-causal | **0.0062** [jszs2xyz](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jszs2xyz)  | 0.2743 [0upsax9u](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/0upsax9u) | **0.00649** [b1xfl1h8](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/b1xfl1h8) |

> **Mamba causal**: hidden_dim=208 (unidir), LR=5e-4, GC=1.0. simple_copy causal (0.739) is above random baseline — causal Mamba cannot solve this task.
> **Mamba non-causal**: hidden_dim=160 (bidir), LR=1e-3, GC=10.0. Bidir Mamba dramatically improves over causal. Color_cond non-causal (0.00649) rivals Hyena (0.0062).

### 1D — Patched, causal (p=64, 50k iters, 64 tokens)

| Task        | Hyena                                                                                     | Attention                                                                             |
| ----------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| simple_copy | **3.85e-5** [dz8j5e3a](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dz8j5e3a) | 1.31e-3 [x9hhpmtn](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/x9hhpmtn) |
| color_cond  | **0.0630** [q3c7i8tv](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/q3c7i8tv)  | 0.2604 [me9o7fgs](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/me9o7fgs)  |

### 1D — Hyena Patch-Size Ablation (non-causal, 50k iters)

| Task        | p=2 (2048 tok)                                                                        | p=4 (1024 tok)                                                                        | p=8 (512 tok)                                                                         | p=16 (256 tok)                                                                           | p=32 (128 tok)                                                                            | p=64 (64 tok)                                                                         | p=128 (32 tok)                                                                        | p=256 (16 tok)                                                                        | No patch (4096 tok) |
| ----------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------- |
| simple_copy | 1.12e-4 [xw1680ay](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xw1680ay) | 1.20e-4 [e03hltl3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/e03hltl3) | 9.56e-5 [4yqz3vij](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/4yqz3vij) | 4.47e-5 [5u0ubeyr](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5u0ubeyr)    | **3.34e-5** [bcu5u0pv](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/bcu5u0pv) | 3.42e-5 [gp7oozbp](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/gp7oozbp) | 3.73e-5 [000bu9w3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/000bu9w3) | 5.70e-5 [lbagpjqe](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lbagpjqe) | 5.20e-5             |
| color_cond  | 0.0445 [pmfefi4i](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/pmfefi4i)  | 0.0278 [vp0l8tuy](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vp0l8tuy)  | 0.0078 [z904ee6a](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/z904ee6a)  | **0.0050** [35tg1h76](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/35tg1h76) | 0.0079 [wbgmuhcv](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/wbgmuhcv)      | 0.0235 [otugybji](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/otugybji)  | 0.142 [u6els4i0](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/u6els4i0)   | 0.246 [vjlfkki1](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vjlfkki1)   | 0.0062              |

### 1D — Attention Patch-Size Ablation (non-causal, 50k iters)

| Task        | p=2 (2048 tok)                                                                      | p=4 (1024 tok)                                                                      | p=8 (512 tok)                                                                       | p=16 (256 tok)                                                                        | p=32 (128 tok)                                                                        | p=64 (64 tok)                                                                         | p=128 (32 tok)                                                                            | p=256 (16 tok)                                                                        | No patch (4096 tok) |
| ----------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------- |
| simple_copy | 0.197 [k0suwc1l](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/k0suwc1l) | 0.041 [c0q9r3ca](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/c0q9r3ca) | 0.009 [vxee4b3a](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vxee4b3a) | 5.38e-3 [pwqvu4oq](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/pwqvu4oq) | 9.96e-4 [n18xjkbn](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/n18xjkbn) | 1.51e-4 [x51h48q0](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/x51h48q0) | **1.20e-4** [5jlmqp67](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5jlmqp67) | 1.76e-4 [63d7egbu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/63d7egbu) | 0.1504              |
| color_cond  | 0.275 [xg0ze4a3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xg0ze4a3) | 0.274 [e6vcfme1](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/e6vcfme1) | 0.269 [xaulovml](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xaulovml) | 0.266 [q5e6ajhe](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/q5e6ajhe)   | 0.264 [9a6cbwyl](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/9a6cbwyl)   | 0.261 [thz0o6zj](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/thz0o6zj)   | **0.256** [g9n5jkoi](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/g9n5jkoi)   | 0.256 [3x4p8ms0](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/3x4p8ms0)   | 0.274               |

### 1D — Mamba Bidir Patch-Size Ablation (non-causal, 50k iters, hidden_dim=160)

| Task        | p=2 (2048 tok)                                                                       | p=4 (1024 tok)                                                                        | p=8 (512 tok)                                                                             | p=16 (256 tok)                                                                        | p=32 (128 tok)                                                                        | p=64 (64 tok)                                                                             | p=128 (32 tok)                                                                        | p=256 (16 tok)                                                                        | No patch (4096 tok) |
| ----------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------- |
| simple_copy | 0.0153 [1cttxrim](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/1cttxrim) | 2.57e-4 [serrcezb](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/serrcezb) | 1.40e-3 [n8r9oke9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/n8r9oke9)     | 1.27e-4 [34j6pb01](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/34j6pb01) | 5.27e-5 [lpxgte87](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lpxgte87) | **4.66e-5** [ikj2isdw](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ikj2isdw) | 7.45e-5 [wqwuxx8w](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/wqwuxx8w) | 1.86e-3 [z21r7xj5](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/z21r7xj5) | 0.0854              |
| color_cond  | 0.0121 [6tlfi6nu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6tlfi6nu) | 7.88e-3 [vb9uwnbg](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vb9uwnbg) | **2.38e-4** [0l3b0umg](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/0l3b0umg) | 8.51e-4 [6i61tyhm](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6i61tyhm) | 2.48e-3 [yz3kpfu8](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/yz3kpfu8) | 7.57e-3 [5p0qx8s7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5p0qx8s7)     | 0.0348 [etra3ldn](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/etra3ldn)  | 0.113 [etjphdjd](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/etjphdjd)   | 0.00649             |

> Mamba benefits enormously from patchification on simple_copy: p=64 (4.66e-5) matches Hyena's best (3.34e-5). Color_cond best at p=8 (2.38e-4), dramatically better than non-patched (0.00649).

### 2D — No patches (50k iters, sequence length = 4096)

| Task        | Hyena                                                                                     | Hyena (Gaussian mask)                                                                     | Attention                                                                            | Mamba (bidir)                                                                         |
| ----------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| simple_copy | 0.000208 [dgskbet7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dgskbet7)    | **1.31e-4** [21dirnk3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/21dirnk3) | 0.0923 [5wvmyxbq](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5wvmyxbq) | 0.349 [1mcywrtu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/1mcywrtu)   |
| color_cond  | **0.00254** [2m7xg9ue](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/2m7xg9ue) | 3.18e-3 [fmvyec34](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/fmvyec34)     | 0.2528 [6y4ed04f](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6y4ed04f) | 0.00834 [zz1b4k0g](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/zz1b4k0g) |

### 2D — Hyena Patch-Size Ablation (50k iters)

| Task        | Kernel      | p=2 (1024 tok)                                                                            | p=4 (256 tok)                                                                             | p=8 (64 tok)                                                                              | p=16 (16 tok)                                                                         | No patch (4096 tok)                                                                   |
| ----------- | ----------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| simple_copy | standard    | 3.27e-4 [r53a2nzb](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/r53a2nzb)     | 1.33e-4 [0zegy0v5](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/0zegy0v5)     | **6.53e-5** [yelms3ly](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/yelms3ly) | 8.52e-5 [voxd8eam](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/voxd8eam) | 2.08e-4                                                                               |
| simple_copy | Gaussian    | 2.98e-4 [5nqau16k](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5nqau16k)     | 1.01e-4 [3wxjwodq](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/3wxjwodq)     | 1.03e-4 [gk0yo0ee](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/gk0yo0ee)     | 1.44e-4 [9hq6ya92](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/9hq6ya92) | 1.31e-4                                                                               |
| simple_copy | Gauss ω₀∝L  | 1.10e-3 [hbbzslkw](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/hbbzslkw)     | 3.54e-4 [aldt0jvr](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/aldt0jvr)     | 5.50e-4 [ephwnvs4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ephwnvs4)     | 4.50e-4 [rydv6ods](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/rydv6ods) | —                                                                                     |
| color_cond  | standard    | **0.00209** [onyp87gv](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/onyp87gv) | 0.00561 [sfkgm6uj](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/sfkgm6uj)     | 0.01694 [pwex00e4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/pwex00e4)     | 0.06380 [ipnwqpux](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ipnwqpux) | 0.00254                                                                               |
| color_cond  | Gaussian    | 4.07e-3 [dfktztc8](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dfktztc8)     | 7.81e-3 [eyd16ed5](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/eyd16ed5)     | 0.0227 [o5tc1crd](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o5tc1crd)      | 0.0616 [nmc5xg8u](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/nmc5xg8u)  | 3.18e-3                                                                               |
| color_cond  | Gauss ω₀∝L  | **3.79e-3** [c7zd7dn9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/c7zd7dn9) | **5.46e-3** [dgtbl402](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dgtbl402) | **0.0163** [r6z5hhf0](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/r6z5hhf0)  | 0.0663 [3jwacl84](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/3jwacl84)  | —                                                                                     |
| color_cond  | Gauss ω₀=30 | 5.86e-3 [1osqhdrt](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/1osqhdrt)     | 6.63e-3 [dmub18y7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dmub18y7)     | 0.0220 [52lco0il](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/52lco0il)      | 0.0752 [h66fritc](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/h66fritc)  | 4.53e-3 [khpkb1s2](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/khpkb1s2) |

> **ω₀ ablation summary** (Gaussian mask, color_cond):
>
> - **ω₀∝L** (scaled down: 0.625–5.0): best at p=2/4/8 — up to 1.4x better than ω₀=10. Lower ω₀ avoids wasting capacity on frequencies the coarse grid can't resolve.
> - **ω₀=30** (scaled up): mostly worse than ω₀=10. Slight help at p=4/8 but hurts at no-patch, p=2, p=16.
> - **Conclusion**: color_cond benefits from *lower* ω₀ on coarse grids. Higher ω₀ does not help.
> - **simple_copy** (ω₀∝L only): scaling *hurts* 3-5x — high frequencies needed for pixel-precise recall regardless of grid resolution.
> - Gaussian mask vs standard Hyena: comparable on simple_copy (Gaussian slightly better at p=4: 1.01e-4 vs 1.33e-4), slightly worse on color_cond across all patch sizes.

### 2D — Attention Patch-Size Ablation (50k iters)

| Task        | p=2 (1024 tok)                                                                       | p=4 (256 tok)                                                                         | p=8 (64 tok)                                                                             | p=16 (16 tok)                                                                             | No patch (4096 tok) |
| ----------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------- |
| simple_copy | 0.0632 [fwfp2asa](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/fwfp2asa) | 4.30e-3 [g9x5adia](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/g9x5adia) | 2.56e-4 [uw3sxxmn](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/uw3sxxmn)    | **1.96e-4** [75juy3i0](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/75juy3i0) | 0.0923              |
| color_cond  | 0.2705 [wti4iogh](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/wti4iogh) | 0.2155 [nn1abk9m](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/nn1abk9m)  | **0.1266** [8c2n11pj](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/8c2n11pj) | 0.1311 [dk0fzwe3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dk0fzwe3)      | 0.2528              |

### 2D — Mamba Bidir Patch-Size Ablation (50k iters, hidden_dim=160)

| Task        | p=2 (1024 tok)                                                                        | p=4 (256 tok)                                                                             | p=8 (64 tok)                                                                              | p=16 (16 tok)                                                                         | No patch (4096 tok) |
| ----------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------------- |
| simple_copy | 5.25e-3 [316hh9ct](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/316hh9ct) | 2.92e-4 [nwvirwwk](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/nwvirwwk)     | **5.70e-5** [lgz82sx7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lgz82sx7) | 1.87e-3 [jw1syq0r](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jw1syq0r) | 0.349               |
| color_cond  | 8.61e-4 [j7jd3uxu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/j7jd3uxu) | **4.06e-4** [zjdefavd](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/zjdefavd) | 0.0106 [z1nv5mws](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/z1nv5mws)      | 0.0550 [fcpox1fw](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/fcpox1fw)  | 0.00834             |

> Mamba 2D simple_copy: p=8 (5.70e-5) matches Hyena's best (6.53e-5). Color_cond: p=4 (4.06e-4) dramatically beats Hyena's best (2.09e-3) by 5x.

### 3D — simple_copy (50k iters, volume = 8×64×64 = 32768 voxels)

| Variant              | Hyena (256)                                                                               | Attention w/ RoPE (240)                                                                   | Mamba bidir (160)                                                                         | Attention no RoPE (256, old)                                                        |
| -------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| No patch (32768 tok) | **3.77e-4** [7afq7yvc](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/7afq7yvc) | —                                                                                         | 0.568 [t6f3pv4b](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/t6f3pv4b)       | 0.884 [uegzkdjj](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/uegzkdjj) |
| p=2 (4096 tok)       | 1.06e-4 [p9i45cak](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/p9i45cak)     | 0.172 [2veluz00](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/2veluz00)       | 0.107 [ucvphue0](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ucvphue0)       | 0.884                                                                               |
| p=4 (512 tok)        | **3.30e-5** [yetswbzb](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/yetswbzb) | 0.0165 [fm0eddsz](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/fm0eddsz)      | 7.99e-3 [zg16fzgx](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/zg16fzgx)     | 0.862                                                                               |
| p=8 (64 tok)         | 4.15e-5 [pfvody2v](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/pfvody2v)     | **3.26e-4** [6uxb0enu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6uxb0enu) | **9.79e-4** [y0hbwudb](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/y0hbwudb) | 0.712                                                                               |

> 3D Attention uses hidden_dim=240 (head_dim=30) for 3D RoPE compatibility. Old runs used hidden_dim=256 with `use_rope=False`.
> 3D Mamba patched: p=8 (9.79e-4) beats Attention p=8 (3.26e-4 is still better) but far behind Hyena (4.15e-5). Mamba improves dramatically with patches (0.568 → 9.79e-4 at p=8).

### 3D — color_conditioning (50k iters, volume = 8×64×64 = 32768 voxels, 4 items)

| Variant              | Hyena (256)                                                                         | Attention w/ RoPE (240)                                                             | Mamba bidir (160)                                                                        |
| -------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| No patch (32768 tok) | 0.225 [qst2u9c4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/qst2u9c4) | OOM [q3xh6x0b](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/q3xh6x0b)   | **0.0150** [ju98o3fr](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ju98o3fr) |
| p=2 (4096 tok)       | 0.212 [en7ewvgd](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/en7ewvgd) | 0.276 [mm88vhum](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/mm88vhum) | 0.0463 [4qhh04sa](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/4qhh04sa)     |
| p=4 (512 tok)        | 0.209 [gu3h5ir2](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/gu3h5ir2) | 0.274 [99m03564](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/99m03564) | 0.0563 [n2lq93ea](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/n2lq93ea)     |
| p=8 (64 tok)         | 0.150 [e9mzph0v](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/e9mzph0v) | 0.267 [orwlkpt3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/orwlkpt3) | 0.128 [bdnlpjou](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/bdnlpjou)      |

> 3D Mamba non-patched color_cond (**0.0150**) crushes Hyena (0.225) and Hyena patched p=8 (0.150) by 10-15x. Mamba patched also strong: p=2 (0.0463) already beats Hyena best (0.150). Mamba p=8 (0.128) is worse than non-patched — patching hurts Mamba on 3D color_cond.

______________________________________________________________________

## Phase 1 — Hyperparameter Ablation (1D Simple Copy, 20k iters)

### Hyena 1D Ablation

Base config: LR=5e-4, WD=1e-3, GC=1.0 (chosen defaults after this ablation).

| ID         | Config                    | test/loss    | Notes     | SLURM |
| ---------- | ------------------------- | ------------ | --------- | ----- |
| H1-default | LR=5e-4, WD=1e-3, GC=10.0 | 0.000215     | baseline  | 40348 |
| H1-lr5e5   | LR=5e-5                   | 0.000345     | too slow  | 40349 |
| H1-wd0     | WD=0                      | **0.000178** | best loss | 40351 |
| H1-gc100   | GC=100                    | 0.000246     | —         | 40354 |
| H1-wd1e2   | WD=1e-2                   | 0.000250     | —         | 40352 |

**Chosen defaults**: LR=5e-4, WD=1e-3, GC=1.0 (good balance of stability and performance).

### Hyena 2D Color Conditioning Validation

Confirms the Hyena defaults (LR=5e-4, GC=1.0) transfer well to 2D.

| ID             | Config           | test/loss | Notes           | SLURM |
| -------------- | ---------------- | --------- | --------------- | ----- |
| H-cc2d-best    | LR=5e-4, GC=1.0  | 0.01404   | **3.3x better** | 40365 |
| H-cc2d-default | LR=1e-4, GC=10.0 | 0.04624   | old defaults    | 40366 |

### Attention 1D Ablation

Base config: LR=1e-4, WD=1e-3, GC=10.0.

| ID           | LR       | GC       | val/loss   | test/loss  | SLURM |
| ------------ | -------- | -------- | ---------- | ---------- | ----- |
| A1-lr1e4     | 1e-4     | 10.0     | 0.2000     | 0.2002     | 40377 |
| A1-lr5e4     | 5e-4     | 10.0     | 0.1984     | 0.1987     | 40378 |
| **A1-lr1e3** | **1e-3** | **10.0** | **0.1656** | **0.1661** | 40379 |
| A1-lr5e4-gc1 | 5e-4     | 1.0      | 0.1703     | 0.1709     | 40380 |
| A1-lr1e3-gc1 | 1e-3     | 1.0      | 0.2108     | 0.2115     | 40381 |

**Chosen defaults**: LR=1e-3, WD=1e-3, GC=10.0 (17% improvement over baseline).

> Unlike Hyena, tighter grad clipping (GC=1.0) hurts Attention at high LR.

### Mamba 1D Ablation — Causal (unidirectional, hidden_dim=208, ~1.80M params)

Base config: LR=1e-4, WD=1e-3, GC=10.0. Throughput: ~10.7 it/s on H100.

| ID               | LR       | GC      | test/loss | SLURM |
| ---------------- | -------- | ------- | --------- | ----- |
| M1-base          | 1e-4     | 10.0    | 0.884     | 40639 |
| **M1-lr5e4-gc1** | **5e-4** | **1.0** | **0.759** | 40642 |
| M1-lr5e4         | 5e-4     | 10.0    | 0.802     | 40640 |
| M1-lr1e3         | 1e-3     | 10.0    | 0.820     | 40641 |
| M1-lr1e3-gc1     | 1e-3     | 1.0     | 0.909     | 40643 |

**Chosen causal defaults**: LR=5e-4, GC=1.0 (best of a bad lot — all above random baseline 0.549).

> Causal Mamba struggles heavily on 1D simple_copy. Even the best config (0.759) is far worse than random (0.549). This suggests unidirectional Mamba cannot solve the spatial recall task in causal mode.

### Mamba 1D Ablation — Bidirectional (non-causal, hidden_dim=160, ~1.90M params)

| ID            | LR       | GC       | test/loss | SLURM |
| ------------- | -------- | -------- | --------- | ----- |
| MB1-base      | 1e-4     | 10.0     | 0.663     | 40644 |
| MB1-lr5e4     | 5e-4     | 10.0     | 0.231     | 40645 |
| **MB1-lr1e3** | **1e-3** | **10.0** | **0.155** | 40646 |
| MB1-lr5e4-gc1 | 5e-4     | 1.0      | 0.661     | 40647 |
| MB1-lr1e3-gc1 | 1e-3     | 1.0      | 0.174     | 40648 |

**Chosen bidirectional defaults**: LR=1e-3, GC=10.0 (comparable to Attention at 0.150).

> Bidirectional Mamba (0.155) vastly outperforms causal (0.759) and is comparable to Attention (0.150), but still far behind Hyena (5.2e-5).

______________________________________________________________________

## Key Observations

1. **Hyena dominates across all dimensions and tasks**: orders of magnitude better than Attention on non-patched tasks (1D, 2D) and most patched configurations.
1. **Patchification massively helps Attention on simple_copy**:
   - 1D: **0.150 → 1.20e-4** at p=128 (1250x improvement!). Monotonically improves with larger patches.
   - 2D: **0.0923 → 1.96e-4** at p=16 (470x improvement!).
1. **Hyena simple_copy — non-monotonic patch sweet spot**:
   - 1D: best at p=32 (3.34e-5), comparable across p=16–128. Degrades at extremes (p=2: 1.12e-4, p=256: 5.70e-5).
   - 2D: best at p=8 (6.53e-5), worse at both p=2 and p=16.
1. **Color conditioning — fine spatial detail matters**:
   - Hyena 1D: best at p=16 (0.0050), but large patches collapse (p=128: 0.142, p=256: 0.246). Non-patched (0.0062) is competitive.
   - Hyena 2D: monotonically better with smaller patches; p=2 (0.002) beats no-patch (0.003).
   - Attention: hopeless on color_cond across all 1D variants (~0.256–0.275). Patching gives marginal improvement at best.
1. **3D RoPE is critical for Attention**:
   - Without RoPE (head_dim=32): 0.71–0.88 across all patch sizes — completely broken.
   - With RoPE (head_dim=30, hidden_dim=240): p=8 achieves **3.26e-4** (2000x improvement over no-RoPE).
   - Attention still lags Hyena on 3D simple_copy (3.26e-4 vs 4.15e-5 at p=8).
1. **3D color_conditioning**:
   - Hyena best at p=8 (0.150), comparable to 2D results. Smaller patches (p=2,4) plateau around 0.21.
   - Attention ~0.27 across all patch sizes — similar to 1D/2D color_cond failure mode.
1. **3D very slow without patches**: ~1 it/s for non-patched 3D (ETA ~13h for 50k iters).
1. **Patchify `.contiguous()` fix**: resolved `torch.compile` + Patchify stride bug for all channel counts.
1. **Mamba — causal is broken, bidir is competitive**:
   - Causal Mamba (unidir) cannot solve 1D simple_copy (0.739, worse than random 0.549). Causal color_cond (0.205) also poor.
   - Bidir Mamba (non-causal) at 4096 tokens: simple_copy 0.0854 (beats Attention 0.150), color_cond **0.00649** (rivals Hyena 0.0062!).
   - With patches, Mamba matches Hyena on simple_copy: p=64 gives 4.66e-5 vs Hyena's 3.34e-5.
   - Color_cond with patches: Mamba best at p=8 (2.38e-4), much better than non-patched (6.49e-3) and Hyena best (5.0e-3).
1. **Mamba compiled at ~10–30 it/s**: non-patched ~10.7 it/s, patched p=32 ~30 it/s (compiled with `max-autotune-no-cudagraphs`). 3D at ~4.5 it/s (vs Hyena ~1.1 it/s).
1. **Mamba 2D patched**: simple_copy p=8 (5.70e-5) matches Hyena (6.53e-5). Color_cond p=4 (**4.06e-4**) beats Hyena (2.09e-3) by 5x — Mamba is the new best on 2D color_conditioning with patches.
1. **Mamba 3D color_conditioning**: non-patched **0.0150** crushes Hyena no-patch (0.225, 15x) and Hyena best patched (0.150, 10x). Mamba dominates color_conditioning across all dimensions. Patching *hurts* Mamba on 3D cc: p=2 (0.0463), p=4 (0.0563), p=8 (0.128) — all worse than non-patched (0.0150).
1. **Hyena Gaussian mask (2D)**: Gaussian mask slightly improves simple_copy non-patched (1.31e-4 vs 2.08e-4 standard) but is comparable/slightly worse on color_cond (3.18e-3 vs 2.54e-3). Patched results similar to standard Hyena — no dramatic improvement from Gaussian masking.
1. **omega_0 scaling (2D Gaussian mask)**: task-dependent. Scaling ω₀ down (∝L_cache) *hurts* simple_copy (3-5x worse) but *helps* color_cond at p=2/4/8 (up to 1.4x). Scaling ω₀ *up* to 30 mostly hurts color_cond. Conclusion: simple_copy wants max frequency content; color_cond prefers frequency-matched kernels (lower ω₀ for coarser grids).

______________________________________________________________________

## TODO

- [x] ~~Install `mamba_ssm` in `nv-subq` conda environment~~ — v2.3.1 built from source (`~/resources/mamba/`)
- [x] ~~Reduce Mamba 2D/3D `hidden_dim` to ~160~~ — unidir=208 (1.80M), bidir=160 (1.90M); must be multiple of 16
- [x] ~~Run Mamba 1D ablation (LR, GC sweep)~~ — causal: LR=5e-4, GC=1.0; bidir: LR=1e-3, GC=10.0
- [x] ~~Complete 3D experiments (Hyena/Attention)~~ — all done (except 3D Attention color_cond no-patch: OOM)
- [x] ~~Launch 1D Mamba production runs (causal + non-causal, 50k iters)~~ — 40670–40673
- [x] ~~Launch 1D Mamba patch-size ablation (bidir, 50k iters)~~ — 40654–40669
- [x] ~~Launch 2D Mamba production runs (simple_copy + color_conditioning)~~ — 40750, 40751
- [x] ~~Launch 2D Mamba patch-size ablation (bidir, 50k iters)~~ — 40809–40817
- [x] ~~Launch 3D Mamba production runs (simple_copy + color_conditioning)~~ — 40782, 40783
- [x] ~~Launch 3D Mamba patch-size ablation (bidir, 50k iters)~~ — 40835–40840, all complete
- [x] ~~Launch 2D Hyena Gaussian mask runs (non-patched + patched ablation)~~ — 40869–40878, all complete
- [x] ~~Fix `torch.compile` + Patchify stride bug~~ — fixed with `.contiguous()` in Patchify/Unpatchify
- [x] ~~3D Attention RoPE~~ — fixed with hidden_dim=240 (head_dim=30, 30%6==0)
- [x] ~~3D color_conditioning~~ — implemented and initial patched runs complete
- [x] ~~omega_0 ∝ L scaling sweep (2D Gaussian mask, 50k iters)~~ — 40925–40932, all complete
- [x] ~~omega_0=30 ablation on color_cond 2D Gaussian mask (no-patch + patch sizes)~~ — 40962–40966, all complete
- [ ] Rerun 3D Attention color_cond no-patch with `compile=False` (OOM with inductor)

______________________________________________________________________

**Last Updated**: 2026-04-14
