# Spatial Recall 1D - EMNIST Mask Selection - Experiment Tracker

## Task Description

1D version of mask selection (similar to 2D mask selection) where:

- Images are flattened FIRST (16×16 → 256 elements)
- 4 flattened images placed as contiguous segments in 1D canvas (4096 elements)
- Binary mask channel indicates which digit to recall
- Model must regress the target region for the masked digit from a **causal** perspective

Key difference from simple_copy: 4 items on canvas with mask indicating target (vs 1 item with fixed position).

## Dataset Configuration

- **Input**: 2 channels (grayscale canvas + binary mask)
- **Output**: 1 channel (16×16 flattened = 256 elements)
- **Canvas size**: 64×64 flattened → 4096 elements
- **Target size**: 16×16 flattened → 256 elements
- **Num items**: 4 (1 target + 3 distractors)
- **Placement**: random
- **Mask**: Binary channel indicating target digit location

## Model Configurations

### XS (Extra-Small) Models

| Model     | Hidden Dim | Heads/Headdim          | Params | Notes                                |
| --------- | ---------- | ---------------------- | ------ | ------------------------------------ |
| Attention | 160        | 8 heads (head_dim=20)  | ~719K  | Causal attention with RoPE           |
| Mamba     | 128        | headdim=32, expand=2   | ~738K  | Unidirectional (bidirectional=False) |
| Hyena     | 160        | SIREN kernel, 3 layers | ~757K  | Causal CKConvND + CausalConv1D       |

______________________________________________________________________

## Available Configs

### Non-Patchify (full 4096 sequence length)

| Config                        | Size | Hidden Dim | Notes                                      |
| ----------------------------- | ---- | ---------- | ------------------------------------------ |
| ccnn_hyena_causal_xs_lcache64 | XS   | 160        | L_cache=64 (key finding from simple_copy!) |
| ccnn_hyena_causal_s_lcache64  | S    | 256        | 🆕 L_cache=64                              |
| ccnn_hyena_causal_m_lcache64  | M    | 416        | 🆕 L_cache=64                              |
| ccnn_mamba_causal_xs          | XS   | 128        | Unidirectional Mamba                       |
| ccnn_mamba_causal_s           | S    | 224        | 🆕 Unidirectional Mamba                    |
| ccnn_mamba_causal_m           | M    | 352        | 🆕 Unidirectional Mamba                    |
| ccnn_attn_causal_xs           | XS   | 160        | Causal attention with RoPE                 |
| ccnn_attn_causal_s            | S    | 256        | 🆕 Causal attention with RoPE              |
| ccnn_attn_causal_m            | M    | 384        | 🆕 Causal attention with RoPE              |

> **Note**: Only L_cache=64 Hyena configs included (L_cache=4096 variant omitted based on simple_copy findings)

### Patchify (64 tokens with patch_size=64)

| Config                        | Size | Notes          |
| ----------------------------- | ---- | -------------- |
| ccnn_hyena_causal_xs_patchify | XS   | L_cache=64     |
| ccnn_mamba_causal_xs_patchify | XS   | Unidirectional |
| ccnn_attn_causal_xs_patchify  | XS   | Causal + RoPE  |

______________________________________________________________________

## Experiments

### Initial Baseline Experiments (20k iterations)

**Non-Patchify Models (readout_value=0.0)**

| W&B ID                                                                        | Config                        | Status   | Val Loss   | Notes                 |
| ----------------------------------------------------------------------------- | ----------------------------- | -------- | ---------- | --------------------- |
| [j9tixl8h](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/j9tixl8h) | ccnn_hyena_causal_xs_lcache64 | Finished | **0.0789** | 🏆 Best non-patchify! |
| [9sr6pmwd](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/9sr6pmwd) | ccnn_mamba_causal_xs          | Finished | 0.7648     | Poor                  |
| [p47kqjb4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/p47kqjb4) | ccnn_attn_causal_xs           | Finished | 0.5866     |                       |

**Non-Patchify Models (readout_value=-1.0)**

| W&B ID                                                                        | Config                        | Status   | Val Loss   | Notes                    |
| ----------------------------------------------------------------------------- | ----------------------------- | -------- | ---------- | ------------------------ |
| [i3fcp3ep](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/i3fcp3ep) | ccnn_hyena_causal_xs_lcache64 | Finished | **0.1050** | Worse than readout=0.0   |
| [x7nv1xn3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/x7nv1xn3) | ccnn_mamba_causal_xs          | Finished | 0.4451     | 1.7x better with marker! |
| [0u7t5rsn](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/0u7t5rsn) | ccnn_attn_causal_xs           | Finished | 0.5435     | Slight improvement       |

**Patchify Models XS (readout_value=0.0)**

| W&B ID                                                                        | Config                        | Patch Size | Status   | Val Loss | Notes |
| ----------------------------------------------------------------------------- | ----------------------------- | ---------- | -------- | -------- | ----- |
| [tkt469hb](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/tkt469hb) | ccnn_hyena_causal_xs_patchify | 64         | Finished | 0.2007   |       |
| [vpyjeess](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vpyjeess) | ccnn_mamba_causal_xs_patchify | 64         | Finished | 0.3423   |       |
| [d0ele5xq](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/d0ele5xq) | ccnn_attn_causal_xs_patchify  | 64         | Crashed  | -        |       |

**Patchify Models XS (readout_value=-1.0)**

| W&B ID                                                                        | Config                        | Patch Size | Status   | Val Loss   | Notes |
| ----------------------------------------------------------------------------- | ----------------------------- | ---------- | -------- | ---------- | ----- |
| [5fsh98ms](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5fsh98ms) | ccnn_hyena_causal_xs_patchify | 64         | Finished | **0.1389** |       |
| [e8yhful7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/e8yhful7) | ccnn_mamba_causal_xs_patchify | 64         | Crashed  | -          |       |
| [oz3e9w6m](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/oz3e9w6m) | ccnn_attn_causal_xs_patchify  | 64         | Finished | 0.3838     |       |

______________________________________________________________________

## Expected Findings (Based on simple_copy and 2D mask_selection)

### From 1D simple_copy:

1. **Hyena L_cache=64** dramatically outperformed all other configs (val loss ~0.00005)
1. Patchification with large patch_size (1024) worked best for Attention/Mamba
1. For Hyena, smaller patch sizes (4-8) with longer sequences performed better

### From 2D mask_selection:

1. All architectures improved significantly with more training (5x → 15-17x improvement)
1. Hyena achieved best results overall (0.00129 at 100k iterations)
1. Mamba + Patchify worked well (0.00295 at 100k iterations)
1. Optimal patch sizes varied: Hyena p=2, Mamba p=4, Attention p=8

### Hypotheses for 1D mask_selection:

1. ✅ **Hyena L_cache=64** dominates (testing SIREN frequency grid effect)
1. ✅ Mask selection is harder than simple_copy (4 items vs 1, random placement)
1. ❌ Patchification hurts all models at XS size (unlike 2D where it helped)
1. 🔄 Longer training may help close the gap

______________________________________________________________________

## WandB

- **Group (XS non-patchify)**: `spatial_recall_1d_emnist_mask_selection_xs`
- **Group (XS patchify)**: `spatial_recall_1d_emnist_mask_selection_xs_patchify`
- **Project**: `nvsubquadratic`
- **Entity**: `implicit-long-convs`

______________________________________________________________________

## Notes

- Models are **causal** (unidirectional) - can only see past, not future
- Binary mask in second channel indicates which of 4 digits to recall
- This tests the model's ability to:
  1. Process the mask to identify target location
  1. Recall information from past context (causal constraint)
  1. Output the correct target in the readout region

______________________________________________________________________

### Patch Size Sweep (20k iterations)

Sweeping patch sizes: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024

- Canvas size: 4096 → Sequence lengths: 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4
- **Partition**: LOW (may be preempted)
- **Autoresume**: Enabled (will auto-resume from checkpoint if preempted)

**Hyena XS Patchify**

| Patch | Seq Len | Val Loss   | W&B ID                                                                        | Notes   |
| ----- | ------- | ---------- | ----------------------------------------------------------------------------- | ------- |
| 2     | 2048    | 0.1211     | [rt9xkeud](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/rt9xkeud) |         |
| 4     | 1024    | 0.0868     | [78m2kvgx](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/78m2kvgx) |         |
| 8     | 512     | **0.0775** | [cjo5sk28](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/cjo5sk28) | ⭐ BEST |
| 16    | 256     | 0.0905     | [v3n1ec5f](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/v3n1ec5f) |         |
| 32    | 128     | 0.1747     | [udzmy1h8](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/udzmy1h8) |         |
| 64    | 64      | 0.2008     | [vp3t401l](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vp3t401l) |         |
| 128   | 32      | 0.2163     | [a4z14mj2](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/a4z14mj2) |         |
| 256   | 16      | 0.3098     | [wpel3q1a](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/wpel3q1a) |         |
| 512   | 8       | 0.3852     | [vu2uf057](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vu2uf057) |         |
| 1024  | 4       | 0.5108     | [n67o2rzu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/n67o2rzu) |         |

**Mamba XS Patchify**

| Patch | Seq Len | Val Loss | W&B ID                                                                        | Notes      |
| ----- | ------- | -------- | ----------------------------------------------------------------------------- | ---------- |
| 2     | 2048    | 0.8810   | [5o84w0hr](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5o84w0hr) | Poor       |
| 4     | 1024    | 0.8735   | [rnnny5n9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/rnnny5n9) | Poor       |
| 8     | 512     | 0.8317   | [jx1alihs](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jx1alihs) | Poor       |
| 16    | 256     | 0.2966   | [jkicz203](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jkicz203) |            |
| 32    | 128     | 0.2868   | [ufy1fgnr](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ufy1fgnr) |            |
| 64    | 64      | 0.3484   | [tdqr0b4b](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/tdqr0b4b) |            |
| 128   | 32      | 0.3614   | [ewwf2cej](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ewwf2cej) |            |
| 256   | 16      | 0.2773   | [rcxla6x6](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/rcxla6x6) | Best Mamba |
| 512   | 8       | 0.4495   | [v4ng6qrd](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/v4ng6qrd) |            |
| 1024  | 4       | 0.4674   | [7fr8npjs](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/7fr8npjs) |            |

**Attention XS Patchify**

| Patch | Seq Len | Val Loss | W&B ID                                                                        | Notes     |
| ----- | ------- | -------- | ----------------------------------------------------------------------------- | --------- |
| 2     | 2048    | 0.5617   | [6weiqxt6](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6weiqxt6) |           |
| 4     | 1024    | 0.5631   | [flbcrw4b](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/flbcrw4b) |           |
| 8     | 512     | 0.5163   | [jx4ezd8h](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jx4ezd8h) |           |
| 16    | 256     | 0.4332   | [jtkzuvzg](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jtkzuvzg) |           |
| 32    | 128     | 0.3873   | [y5ms9uve](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/y5ms9uve) |           |
| 64    | 64      | 0.4244   | [xfurxc9k](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xfurxc9k) |           |
| 128   | 32      | 0.3381   | [glso2fc7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/glso2fc7) | Best Attn |
| 256   | 16      | 0.3517   | [4il24qvg](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/4il24qvg) |           |
| 512   | 8       | 0.4732   | [rkbbydeq](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/rkbbydeq) |           |
| 1024  | 4       | 0.4977   | [a91yeovi](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/a91yeovi) |           |

______________________________________________________________________

### Patch Size Sweep S Models (20k iterations) ✅ COMPLETED

**Hyena S Patchify**

| Patch Size | Seq Len | Val Loss   | W&B ID                                                                        | Notes           |
| ---------- | ------- | ---------- | ----------------------------------------------------------------------------- | --------------- |
| 2          | 2048    | 0.2616     | [xiuth0je](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xiuth0je) |                 |
| 4          | 1024    | **0.0476** | [5ihugxl9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5ihugxl9) | ⭐ Best Hyena S |
| 8          | 512     | 0.0716     | [day93ikf](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/day93ikf) |                 |
| 16         | 256     | 0.0791     | [wmiopyez](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/wmiopyez) |                 |
| 32         | 128     | 0.1036     | [7f4k7mpu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/7f4k7mpu) |                 |
| 64         | 64      | 0.1026     | [4pzeyiwf](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/4pzeyiwf) |                 |
| 128        | 32      | 0.1787     | [mn0adhov](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/mn0adhov) |                 |
| 256        | 16      | 0.2200     | [t3p6w8es](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/t3p6w8es) |                 |
| 512        | 8       | 0.3306     | [zjtbp65k](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/zjtbp65k) |                 |
| 1024       | 4       | 0.4805     | [2i6ywocn](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/2i6ywocn) |                 |

**Mamba S Patchify (readout=0.0)**

| Patch Size | Seq Len | Val Loss   | W&B ID                                                                        | Notes           |
| ---------- | ------- | ---------- | ----------------------------------------------------------------------------- | --------------- |
| 2          | 2048    | 0.8809     | [bj8ptbca](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/bj8ptbca) | Poor            |
| 4          | 1024    | 0.8735     | [xcly9acy](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xcly9acy) | Poor (long seq) |
| 8          | 512     | 0.8299     | [veqnb5i4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/veqnb5i4) | Poor            |
| 16         | 256     | 0.2342     | [mo2pfz0f](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/mo2pfz0f) |                 |
| 32         | 128     | 0.2067     | [fezot7wi](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/fezot7wi) |                 |
| 64         | 64      | **0.1775** | [jl631qd9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jl631qd9) | Best w/o marker |
| 128        | 32      | 0.1812     | [8pxha1eb](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/8pxha1eb) |                 |
| 256        | 16      | 0.3145     | [1u0vcjpy](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/1u0vcjpy) |                 |
| 512        | 8       | 0.3222     | [vb8er11w](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vb8er11w) |                 |
| 1024       | 4       | 0.3997     | [vwzmc4ce](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vwzmc4ce) |                 |

**Mamba S Patchify (readout=-1.0)** 🔥

| Patch Size | Seq Len | Val Loss   | W&B ID                                                                        | Notes            |
| ---------- | ------- | ---------- | ----------------------------------------------------------------------------- | ---------------- |
| 2          | 2048    | 0.5079     | [srg6a8l3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/srg6a8l3) |                  |
| 4          | 1024    | 0.3032     | [6z2nwr3y](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6z2nwr3y) | 2.9x better!     |
| 8          | 512     | 0.0269     | [tymxx56y](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/tymxx56y) | **30x better!**  |
| 16         | 256     | **0.0134** | [16w4ajj6](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/16w4ajj6) | ⭐⭐ **BEST S!** |
| 32         | 128     | 0.0413     | [19j4nm9b](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/19j4nm9b) |                  |
| 64         | 64      | 0.0813     | [fyobjnel](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/fyobjnel) |                  |
| 128        | 32      | 0.1200     | [ojuqh1et](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ojuqh1et) |                  |
| 256        | 16      | 0.2033     | [q4byvsqz](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/q4byvsqz) |                  |
| 512        | 8       | 0.2831     | [zjt4cnai](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/zjt4cnai) |                  |
| 1024       | 4       | 0.3899     | [h2k16rut](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/h2k16rut) |                  |

**Attention S Patchify**

| Patch Size | Seq Len | Val Loss   | W&B ID                                                                        | Notes       |
| ---------- | ------- | ---------- | ----------------------------------------------------------------------------- | ----------- |
| 2          | 2048    | 0.5704     | [25e4olzt](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/25e4olzt) |             |
| 4          | 1024    | 0.5298     | [dtk5uati](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dtk5uati) |             |
| 8          | 512     | 0.4727     | [a5pkoo8f](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/a5pkoo8f) |             |
| 16         | 256     | 0.3411     | [xyz01ea7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/xyz01ea7) |             |
| 32         | 128     | **0.2307** | [gimdwky4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/gimdwky4) | Best Attn S |
| 64         | 64      | 0.2833     | [jbblmj3g](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jbblmj3g) |             |
| 128        | 32      | 0.3148     | [cbgfkjqk](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/cbgfkjqk) |             |
| 256        | 16      | 0.3200     | [2nq2pr5d](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/2nq2pr5d) |             |
| 512        | 8       | 0.3744     | [iyt6wjwv](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/iyt6wjwv) |             |
| 1024       | 4       | 0.4946     | [cpa96tou](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/cpa96tou) |             |

______________________________________________________________________

### Patch Size Sweep M Models (20k iterations) ✅ MOSTLY COMPLETED

**Hyena M Patchify** 🏆

| Patch Size | Seq Len | Val Loss   | W&B ID                                                                        | Notes                    |
| ---------- | ------- | ---------- | ----------------------------------------------------------------------------- | ------------------------ |
| 2          | 2048    | 0.0624     | [g8zqfxq9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/g8zqfxq9) |                          |
| 4          | 1024    | Crashed    | [4nndw5a0](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/4nndw5a0) |                          |
| 8          | 512     | 0.0461     | [04k24oug](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/04k24oug) |                          |
| 16         | 256     | 0.0407     | [0faxid3v](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/0faxid3v) |                          |
| 32         | 128     | **0.0202** | [o597arba](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o597arba) | ⭐⭐⭐ **BEST OVERALL!** |
| 64         | 64      | 0.0540     | [2thp4eq3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/2thp4eq3) |                          |
| 128        | 32      | 0.1146     | [fhfew7rn](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/fhfew7rn) |                          |
| 256        | 16      | 0.1834     | [n2pvdikz](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/n2pvdikz) |                          |
| 512        | 8       | 0.2690     | [wt9mtbpd](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/wt9mtbpd) |                          |
| 1024       | 4       | 0.4603     | [tnecoqti](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/tnecoqti) |                          |

**Mamba M Patchify (readout=0.0)**

| Patch Size | Seq Len | Val Loss   | W&B ID                                                                        | Notes           |
| ---------- | ------- | ---------- | ----------------------------------------------------------------------------- | --------------- |
| 2          | 2048    | 0.8809     | [374ub0jv](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/374ub0jv) | Poor            |
| 4          | 1024    | Crashed    | [58y7sebl](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/58y7sebl) |                 |
| 8          | 512     | 0.5512     | [gsovh73c](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/gsovh73c) | Poor (long seq) |
| 16         | 256     | 0.3038     | [5chqzwt9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5chqzwt9) |                 |
| 32         | 128     | **0.1028** | [4czcin5k](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/4czcin5k) | Best w/o marker |
| 64         | 64      | 0.1176     | [dq473664](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/dq473664) |                 |
| 128        | 32      | 0.1915     | [n8ood8em](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/n8ood8em) |                 |
| 256        | 16      | 0.1537     | [lsdokgb1](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lsdokgb1) |                 |
| 512        | 8       | 0.2512     | [7zvi4h43](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/7zvi4h43) |                 |
| 1024       | 4       | 0.4754     | [beczz4pg](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/beczz4pg) |                 |

**Mamba M Patchify (readout=-1.0)** 🔥

| Patch Size | Seq Len | Val Loss   | W&B ID                                                                        | Notes           |
| ---------- | ------- | ---------- | ----------------------------------------------------------------------------- | --------------- |
| 2          | 2048    | 0.4374     | [k1uhcbx7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/k1uhcbx7) |                 |
| 4          | 1024    | 0.1451     | [fgz9k5t4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/fgz9k5t4) |                 |
| 8          | 512     | 0.0289     | [77gzembz](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/77gzembz) | **18x better!** |
| 16         | 256     | **0.0237** | [2sf33ki6](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/2sf33ki6) | ⭐ Best Mamba M |
| 32         | 128     | 0.0366     | [okbjufgc](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/okbjufgc) |                 |
| 64         | 64      | 0.0494     | [enabiv1t](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/enabiv1t) |                 |
| 128        | 32      | 0.0947     | [sumrnzc5](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/sumrnzc5) |                 |
| 256        | 16      | 0.1598     | [hny74nf9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/hny74nf9) |                 |
| 512        | 8       | 0.2163     | [wiwdbaxu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/wiwdbaxu) |                 |
| 1024       | 4       | 0.3676     | [4iykvg4d](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/4iykvg4d) |                 |

**Attention M Patchify**

| Patch Size | Seq Len | Val Loss   | W&B ID                                                                        | Notes          |
| ---------- | ------- | ---------- | ----------------------------------------------------------------------------- | -------------- |
| 2          | 2048    | 0.5183     | [gvp55qyz](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/gvp55qyz) |                |
| 4          | 1024    | 0.5033     | [todbpof2](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/todbpof2) |                |
| 8          | 512     | 0.4269     | [hyjcbz13](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/hyjcbz13) |                |
| 16         | 256     | 0.2232     | [s4p66j1s](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/s4p66j1s) |                |
| 32         | 128     | **0.1582** | [uc84ic0r](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/uc84ic0r) | ⭐ Best Attn M |
| 64         | 64      | 0.1632     | [h5opbstq](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/h5opbstq) |                |
| 128        | 32      | 0.1839     | [28r80s4b](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/28r80s4b) |                |
| 256        | 16      | 0.2327     | [w69wmklt](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/w69wmklt) |                |
| 512        | 8       | 0.3469     | [34h644wg](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/34h644wg) |                |
| 1024       | 4       | 0.4774     | [lugtcmhb](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lugtcmhb) |                |

______________________________________________________________________

### Summary: Best Results by Model (All Sizes) 🏆

| Model | Size | Best Patch | Best Val Loss | readout | W&B ID                                                                        | Notes                   |
| ----- | ---- | ---------- | ------------- | ------- | ----------------------------------------------------------------------------- | ----------------------- |
| Hyena | M    | 32         | **0.0202** 🏆 | 0.0     | [o597arba](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o597arba) | **BEST OVERALL!**       |
| Mamba | S    | 16         | **0.0134** 🥈 | -1.0    | [16w4ajj6](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/16w4ajj6) | Incredible with marker! |
| Mamba | M    | 16         | **0.0237** 🥉 | -1.0    | [2sf33ki6](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/2sf33ki6) | Excellent with marker   |
| Hyena | M    | 16         | 0.0407        | 0.0     | [0faxid3v](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/0faxid3v) |                         |
| Hyena | S    | 4          | 0.0476        | 0.0     | [5ihugxl9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/5ihugxl9) |                         |
| Hyena | XS   | 8          | 0.0775        | 0.0     | [cjo5sk28](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/cjo5sk28) |                         |
| Mamba | XS   | 32         | 0.0828        | -1.0    | [t1154gzp](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/t1154gzp) |                         |
| Mamba | M    | 32         | 0.1028        | 0.0     | [4czcin5k](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/4czcin5k) | Best M w/o marker       |
| Attn  | M    | 32         | 0.1582        | 0.0     | [uc84ic0r](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/uc84ic0r) | Best Attention          |
| Mamba | S    | 64         | 0.1775        | 0.0     | [jl631qd9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/jl631qd9) | Best S w/o marker       |
| Attn  | S    | 32         | 0.2307        | 0.0     | [gimdwky4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/gimdwky4) |                         |

### Baseline Reference (for interpreting loss values)

Based on EMNIST **byclass** (62 classes) resized to **16x16**, with **normalize_input=True**:

- Normalized pixel mean: ~0, variance: ~0.79
- Visualization: `_tmp/emnist_byclass_mean_digit.png`

| Baseline                 | MSE Loss  | Interpretation                            |
| ------------------------ | --------- | ----------------------------------------- |
| Random N(0,1)            | **~1.79** | Worse than this = completely broken       |
| Predict zeros/mean       | ~0.79     | Just predicting background                |
| **Predict "mean digit"** | **~0.55** | Avg of all 62 classes (no discrimination) |
| \< 0.55                  | -         | Learning something                        |
| **\< 0.27**              | -         | **Good** (2x better than mean digit)      |
| \< 0.10                  | -         | **Very good**                             |
| \< 0.05                  | -         | **Excellent**                             |
| \< 0.02                  | -         | **Outstanding**                           |

### Key Observations (UPDATED with S/M results!)

1. 🏆 **Hyena M dominates!** Best: **0.0202** at patch=32 → **27x better** than mean digit!

   - Significantly better than XS (0.0784) - larger model helps a lot
   - Optimal patch size shifted from 8 (XS) to 32 (M)

1. 🔥 **Mamba S with readout=-1.0 is incredible!** Best: **0.0134** at patch=16

   - Better than Hyena XS (0.0784)!
   - 30x improvement over readout=0 at patch=8 (0.83 → 0.027)
   - readout marker is absolutely critical

1. **Mamba M with readout=-1.0**: **0.0238** at patch=16

   - 18x improvement at patch=8 (0.55 → 0.031)
   - Scales well with model size

1. **Scale matters for all models**:

   - Hyena: XS=0.078 → S=0.048 → M=0.020 (4x improvement XS→M)
   - Mamba (w/ marker): XS=0.083 → S=0.013 → M=0.024 (S best!)
   - Attention: XS=0.34 → S=0.23 → M=0.16 (2x improvement)

1. **Optimal patch sizes**:

   - Hyena: XS=8, S=4, M=32
   - Mamba (w/ marker): XS=32, S=16, M=16
   - Attention: all prefer 32-128

### Mamba XS Patchify with readout_value=-1.0 (testing if explicit marker helps)

| Patch | Seq Len | Val Loss   | W&B ID                                                                        | vs ro=0 | Notes                 |
| ----- | ------- | ---------- | ----------------------------------------------------------------------------- | ------- | --------------------- |
| 2     | 2048    | 0.5482     | [o70l7674](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/o70l7674) | 0.88    | 1.6x better           |
| 4     | 1024    | 0.4552     | [742uet0f](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/742uet0f) | 0.87    | 1.9x better           |
| 8     | 512     | 0.1820     | [vyt8qm3o](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/vyt8qm3o) | 0.83    | **4.5x better**       |
| 16    | 256     | 0.1063     | [txdu7xpg](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/txdu7xpg) | 0.30    | 2.8x better           |
| 32    | 128     | **0.0828** | [t1154gzp](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/t1154gzp) | 0.29    | ⭐ **BEST Mamba XS!** |
| 64    | 64      | 0.1426     | [i5g5w8is](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/i5g5w8is) | 0.35    | 2.4x better           |
| 128   | 32      | 0.1987     | [j8349tu9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/j8349tu9) | 0.36    | 1.8x better           |
| 256   | 16      | 0.2808     | [4dckyhx9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/4dckyhx9) | 0.28    | ~same                 |
| 512   | 8       | 0.4622     | [z95tbsr9](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/z95tbsr9) | 0.45    | ~same                 |
| 1024  | 4       | 0.5095     | [96tajuxu](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/96tajuxu) | 0.47    | ~same                 |

**Key Insight**: readout=-1.0 marker **dramatically helps Mamba at longer sequences** (4.5x improvement at patch=8!)

- Optimal patch shifted from 256 → 32
- Best Mamba: **0.0828** (now competitive with Hyena's 0.0784!)

______________________________________________________________________

**Last Updated**: 2026-01-21
**Status**: 🚀 Running S/M experiments

______________________________________________________________________

### Larger Model Experiments (S and M, Non-Patchify)

Testing larger non-patchified models on full 4096 sequence length.

**Model Sizes:**

| Size | Hidden Dim | Params |
| ---- | ---------- | ------ |
| S    | 224-256    | ~2M    |
| M    | 352-416    | ~5M    |

**S Models (Non-Patchify) - 20k iterations**

| W&B ID                                                                        | Config                       | readout | Val Loss   | Notes      |
| ----------------------------------------------------------------------------- | ---------------------------- | ------- | ---------- | ---------- |
| [1mzab4ug](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/1mzab4ug) | ccnn_hyena_causal_s_lcache64 | 0.0     | **0.0430** | 🏆 Best S! |
| [d71p5zhm](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/d71p5zhm) | ccnn_mamba_causal_s          | -1.0    | 0.3359     |            |
| [008ti1wp](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/008ti1wp) | ccnn_attn_causal_s           | 0.0     | 0.5119     |            |
| [kxfz41ko](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/kxfz41ko) | ccnn_mamba_causal_s          | 0.0     | 0.7590     |            |

**M Models (Non-Patchify) - 20k iterations**

| W&B ID                                                                        | Config                       | readout | Val Loss   | Notes                |
| ----------------------------------------------------------------------------- | ---------------------------- | ------- | ---------- | -------------------- |
| [tes3tvr5](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/tes3tvr5) | ccnn_mamba_causal_m          | -1.0    | **0.2941** | Best M               |
| [6sn6wajq](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6sn6wajq) | ccnn_attn_causal_m           | 0.0     | 0.4821     |                      |
| [6aqccxk7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6aqccxk7) | ccnn_hyena_causal_m_lcache64 | 0.0     | 0.5235     | ⚠️ Unstable training |
| [ks95lg4g](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ks95lg4g) | ccnn_mamba_causal_m          | 0.0     | 0.8806     |                      |

> **Note**: Hyena M (6aqccxk7) showed unstable training - loss diverged after reaching 0.037. Restart submitted (SLURM 173573).

______________________________________________________________________

## Non-Patchified Extended Training (+100k from 20k checkpoint)

Training non-patchified models for +100k iterations to study convergence on full 4096 sequence length.

**Status:** XS ✅ ALL DONE, S preempted, M cancelled pending Hyena M investigation (2026-01-21)

| SLURM ID      | Size | Model | readout | Source Run                                                                    | Result Run                                                                    | Status       | Val Loss      |
| ------------- | ---- | ----- | ------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ------------ | ------------- |
| 173557        | XS   | Attn  | 0.0     | [p47kqjb4](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/p47kqjb4) | [lf332z1r](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/lf332z1r) | ✅ Finished  | 0.1986        |
| 173558        | XS   | Attn  | -1.0    | [0u7t5rsn](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/0u7t5rsn) | [1lqi17yf](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/1lqi17yf) | ✅ Finished  | 0.1666        |
| 173559        | XS   | Hyena | 0.0     | [j9tixl8h](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/j9tixl8h) | [nleryir2](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/nleryir2) | ✅ Finished  | **0.0121** 🏆 |
| 173560        | XS   | Hyena | -1.0    | [i3fcp3ep](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/i3fcp3ep) | [16zjed1r](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/16zjed1r) | ✅ Finished  | 0.0175        |
| 173561        | XS   | Mamba | 0.0     | [9sr6pmwd](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/9sr6pmwd) | [rq6f2kqs](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/rq6f2kqs) | ✅ Finished  | 0.6189        |
| 173562→173730 | XS   | Mamba | -1.0    | [x7nv1xn3](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/x7nv1xn3) | [6lcr7aiw](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6lcr7aiw) | ✅ Finished  | 0.1898        |
| 173563        | S    | Attn  | 0.0     | [008ti1wp](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/008ti1wp) | -                                                                             | ⏸️ Preempted | -             |
| 173564        | S    | Hyena | 0.0     | [1mzab4ug](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/1mzab4ug) | -                                                                             | ⏸️ Preempted | -             |
| 173565        | S    | Mamba | 0.0     | [kxfz41ko](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/kxfz41ko) | -                                                                             | ⏸️ Preempted | -             |
| 173566        | S    | Mamba | -1.0    | [d71p5zhm](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/d71p5zhm) | -                                                                             | ⏸️ Preempted | -             |
| 173567        | M    | Attn  | 0.0     | [6sn6wajq](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6sn6wajq) | -                                                                             | ⏸️ Cancelled | -             |
| 173568        | M    | Hyena | 0.0     | [6aqccxk7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6aqccxk7) | -                                                                             | ⏸️ Cancelled | -             |
| 173569        | M    | Mamba | 0.0     | [ks95lg4g](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/ks95lg4g) | -                                                                             | ⏸️ Cancelled | -             |
| 173570        | M    | Mamba | -1.0    | [tes3tvr5](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/tes3tvr5) | -                                                                             | ⏸️ Cancelled | -             |

### Hyena M Investigation

**Restart test (SLURM 173573):** [95kt8tin](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/95kt8tin) - val=0.5235 @ 20k

⚠️ **CONFIRMED:** Hyena M restart got **identical** val_loss (0.5235) as original run [6aqccxk7](https://wandb.ai/implicit-long-convs/nvsubquadratic/runs/6aqccxk7). This is NOT random instability - Hyena M systematically struggles on full 4096 sequence without patchification. **Recommendation:** Try lower learning rate (1e-5) or use patchification for M models.

TODO(@dwromero): Fix Hyena M instability and submit new runs.

______________________________________________________________________

## Results Summary (XS Only)

| Model | Patchify    | Best Test Loss | Best readout | Notes                       |
| ----- | ----------- | -------------- | ------------ | --------------------------- |
| Hyena | Yes (p=8)   | **0.0784** 🏆  | 0.0          | Best overall                |
| Hyena | No          | 0.0796         | 0.0          | Very close second           |
| Mamba | Yes (p=32)  | **0.0828** 🥈  | -1.0         | NEW! Competitive with Hyena |
| Hyena | Yes (p=64)  | 0.1403         | -1.0         |                             |
| Mamba | Yes (p=16)  | 0.1071         | -1.0         |                             |
| Mamba | Yes (p=256) | 0.2794         | 0.0          | Best without marker         |
| Attn  | Yes (p=128) | 0.3403         | 0.0          | Best Attention              |
| Attn  | No          | 0.5442         | -1.0         |                             |
| Mamba | No          | 0.7648         | 0.0          | Broken at long seq          |

**Key Findings:**

1. 🏆 **Hyena dominates** with test loss 0.0784 (patchify p=8) and 0.0796 (no-patchify)
1. 🥈 **Mamba with readout=-1.0 now competitive!** Best: 0.0828 (p=32) - only 5.6% worse than Hyena
1. **readout=-1.0 is critical for Mamba** at longer sequences (up to 4.5x improvement)
1. **Optimal patch sizes**: Hyena=8, Mamba=32 (with marker), Attention=128
1. **Attention struggles** regardless of configuration (best: 0.3403)

______________________________________________________________________

## Extended Training (+50k iterations from 20k checkpoint)

To check if models converge better with more training, we resume M models from their 20k checkpoints for +50k more iterations using `start_from_checkpoint`.

**Status: 32/35 completed, 3 still running** (2026-01-21)

### 20k vs 20k+50k Extended Training Comparison (Val Loss)

| Model     | Patch | Readout | 20k Val Loss | 20k+50k Val Loss | Improvement | % Better        |
| --------- | ----- | ------- | ------------ | ---------------- | ----------- | --------------- |
| **Attn**  | 4     | 0.0     | 0.5033       | 0.3096\*         | +0.1937     | +38.5%          |
| Attn      | 8     | 0.0     | 0.4269       | 0.0594           | +0.3675     | **+86.1%** 🔥   |
| Attn      | 16    | 0.0     | 0.2232       | 0.0352           | +0.1880     | **+84.2%** 🔥   |
| Attn      | 32    | 0.0     | 0.1582       | **0.0332**       | +0.1249     | +79.0% ⬆️       |
| Attn      | 64    | 0.0     | 0.1632       | 0.0367           | +0.1264     | +77.5% ⬆️       |
| Attn      | 128   | 0.0     | 0.1839       | 0.0547           | +0.1292     | +70.3% ⬆️       |
| Attn      | 256   | 0.0     | 0.2327       | 0.1105           | +0.1222     | +52.5% ⬆️       |
| Attn      | 512   | 0.0     | 0.3469       | 0.2056           | +0.1413     | +40.7%          |
| Attn      | 1024  | 0.0     | 0.4774       | 0.4074           | +0.0700     | +14.7%          |
| **Hyena** | 8     | 0.0     | 0.0461       | 0.0104\*         | +0.0357     | +77.5% ⬆️       |
| Hyena     | 16    | 0.0     | 0.0407       | 0.0038           | +0.0370     | **+90.7%** 🔥   |
| Hyena     | 32    | 0.0     | 0.0202       | **0.0029** 🏆    | +0.0173     | **+85.6%** 🔥   |
| Hyena     | 64    | 0.0     | 0.0540       | 0.0140           | +0.0401     | +74.1% ⬆️       |
| Hyena     | 128   | 0.0     | 0.1146       | 0.0342           | +0.0804     | +70.1% ⬆️       |
| Hyena     | 256   | 0.0     | 0.1834       | 0.0727           | +0.1106     | +60.3% ⬆️       |
| Hyena     | 512   | 0.0     | 0.2690       | 0.1288           | +0.1402     | +52.1% ⬆️       |
| Hyena     | 1024  | 0.0     | 0.4603       | 0.2789           | +0.1814     | +39.4%          |
| **Mamba** | 4     | -1.0    | 0.1451       | 0.0238\*         | +0.1214     | **+83.6%** 🔥   |
| Mamba     | 8     | -1.0    | 0.0289       | **0.0014** 🥇    | +0.0275     | **+95.1%** 🔥🔥 |
| Mamba     | 8     | 0.0     | 0.5512       | 0.3860           | +0.1652     | +30.0%          |
| Mamba     | 16    | -1.0    | 0.0238       | 0.0033           | +0.0205     | **+86.1%** 🔥   |
| Mamba     | 16    | 0.0     | 0.3038       | 0.0757           | +0.2282     | +75.1% ⬆️       |
| Mamba     | 32    | -1.0    | 0.0366       | 0.0083           | +0.0283     | +77.4% ⬆️       |
| Mamba     | 32    | 0.0     | 0.1028       | 0.0282           | +0.0746     | +72.5% ⬆️       |
| Mamba     | 64    | -1.0    | 0.0494       | 0.0161           | +0.0333     | +67.5% ⬆️       |
| Mamba     | 64    | 0.0     | 0.1176       | 0.0344           | +0.0833     | +70.8% ⬆️       |
| Mamba     | 128   | -1.0    | 0.0947       | 0.0338           | +0.0609     | +64.3% ⬆️       |
| Mamba     | 128   | 0.0     | 0.1915       | 0.0522           | +0.1393     | +72.7% ⬆️       |
| Mamba     | 256   | -1.0    | 0.1598       | 0.0647           | +0.0951     | +59.5% ⬆️       |
| Mamba     | 256   | 0.0     | 0.1537       | 0.0615           | +0.0922     | +60.0% ⬆️       |
| Mamba     | 512   | -1.0    | 0.2163       | 0.1124           | +0.1039     | +48.0%          |
| Mamba     | 512   | 0.0     | 0.2512       | 0.1382           | +0.1131     | +45.0%          |
| Mamba     | 1024  | -1.0    | 0.3676       | 0.2200           | +0.1476     | +40.1%          |
| Mamba     | 1024  | 0.0     | 0.4754       | 0.2766           | +0.1987     | +41.8%          |

**Legend:** 🔥 = >80% improvement, ⬆️ = >50% improvement, * = still running

### Key Findings from +50k Extended Training

1. 🥇 **Mamba (ro=-1, p=8) is now BEST overall!** Val loss **0.0014** - 95% improvement from 20k!
1. 🏆 **Hyena (p=32) second best** with val loss **0.0029** - 86% improvement
1. **Extended training (+50k) dramatically helps** - most models improved 50-90%
1. **Attention best at p=32** with val loss 0.0332 - 79% improvement
1. **readout=-1.0 still critical for Mamba** - consistently better than readout=0

### Best Models After 20k+50k Training

| Rank | Model | Patch | Readout | Val Loss   | Improvement |
| ---- | ----- | ----- | ------- | ---------- | ----------- |
| 🥇 1 | Mamba | 8     | -1.0    | **0.0014** | 95.1%       |
| 🥈 2 | Hyena | 32    | 0.0     | **0.0029** | 85.6%       |
| 🥉 3 | Mamba | 16    | -1.0    | 0.0033     | 86.1%       |
| 4    | Hyena | 16    | 0.0     | 0.0038     | 90.7%       |
| 5    | Mamba | 32    | -1.0    | 0.0083     | 77.4%       |
| 6    | Attn  | 32    | 0.0     | 0.0332     | 79.0%       |

**Conclusion:** Extended training (+50k iterations from 20k checkpoint) reveals that **Mamba with readout=-1.0 and patch=8 is the best model**, achieving val loss of 0.0014 - significantly better than Hyena!

______________________________________________________________________

## Further Extended Training (20k+100k total)

To study convergence behavior, we continue training from the 20k+50k checkpoints for another +50k iterations.

**Status: ✅ All 33 jobs completed** (2026-01-21)

### Full Training Progression: 20k → 20k+50k → 20k+100k (Val Loss)

| Model     | Patch | Readout | 20k    | 20k+50k | 20k+100k      | Total Improvement |
| --------- | ----- | ------- | ------ | ------- | ------------- | ----------------- |
| **Attn**  | 4     | 0.0     | 0.5033 | 0.2926  | 0.0854        | +83.0%            |
| Attn      | 8     | 0.0     | 0.4269 | 0.0594  | **0.0234**    | +94.5%            |
| Attn      | 16    | 0.0     | 0.2232 | 0.0352  | **0.0166**    | +92.6%            |
| Attn      | 32    | 0.0     | 0.1582 | 0.0332  | 0.0173        | +89.1%            |
| Attn      | 64    | 0.0     | 0.1632 | 0.0367  | **0.0166**    | +89.8%            |
| Attn      | 128   | 0.0     | 0.1839 | 0.0547  | 0.0346        | +81.2%            |
| Attn      | 256   | 0.0     | 0.2327 | 0.1105  | 0.0665        | +71.4%            |
| Attn      | 512   | 0.0     | 0.3469 | 0.2056  | 0.1651        | +52.4%            |
| Attn      | 1024  | 0.0     | 0.4774 | 0.4074  | 0.3094        | +35.2%            |
| **Hyena** | 8     | 0.0     | 0.0461 | 0.0097  | 0.0040        | +91.3%            |
| Hyena     | 16    | 0.0     | 0.0407 | 0.0038  | **0.0014** 🥈 | +96.5%            |
| Hyena     | 32    | 0.0     | 0.0202 | 0.0029  | **0.0015** 🥉 | +92.6%            |
| Hyena     | 64    | 0.0     | 0.0540 | 0.0140  | 0.0081        | +85.1%            |
| Hyena     | 128   | 0.0     | 0.1146 | 0.0342  | 0.0209        | +81.7%            |
| Hyena     | 256   | 0.0     | 0.1834 | 0.0727  | 0.0458        | +75.0%            |
| Hyena     | 512   | 0.0     | 0.2690 | 0.1288  | 0.0742        | +72.4%            |
| Hyena     | 1024  | 0.0     | 0.4603 | 0.2789  | 0.1868        | +59.4%            |
| **Mamba** | 8     | -1.0    | 0.0289 | 0.0014  | **0.0004** 🥇 | **+98.6%**        |
| Mamba     | 16    | -1.0    | 0.0237 | 0.0033  | **0.0016**    | +93.1%            |
| Mamba     | 32    | -1.0    | 0.0366 | 0.0083  | 0.0035        | +90.4%            |
| Mamba     | 64    | -1.0    | 0.0494 | 0.0161  | 0.0089        | +82.0%            |
| Mamba     | 128   | -1.0    | 0.0947 | 0.0338  | 0.0223        | +76.4%            |
| Mamba     | 256   | -1.0    | 0.1598 | 0.0647  | 0.0424        | +73.5%            |
| Mamba     | 512   | -1.0    | 0.2163 | 0.1124  | 0.0723        | +66.6%            |
| Mamba     | 1024  | -1.0    | 0.3676 | 0.2200  | 0.1786        | +51.4%            |
| Mamba     | 8     | 0.0     | 0.5512 | 0.3860  | 0.3280        | +40.5%            |
| Mamba     | 16    | 0.0     | 0.3038 | 0.0757  | 0.0468        | +84.6%            |
| Mamba     | 32    | 0.0     | 0.1028 | 0.0282  | 0.0132        | +87.2%            |
| Mamba     | 64    | 0.0     | 0.1176 | 0.0344  | 0.0219        | +81.3%            |
| Mamba     | 128   | 0.0     | 0.1915 | 0.0522  | 0.0312        | +83.7%            |
| Mamba     | 256   | 0.0     | 0.1537 | 0.0615  | 0.0411        | +73.2%            |
| Mamba     | 512   | 0.0     | 0.2512 | 0.1382  | 0.0969        | +61.4%            |
| Mamba     | 1024  | 0.0     | 0.4754 | 0.2766  | 0.1931        | +59.4%            |

### Best Models After 20k+100k Training

| Rank | Model     | Patch | Readout | Val Loss   | Total Improvement |
| ---- | --------- | ----- | ------- | ---------- | ----------------- |
| 🥇 1 | **Mamba** | 8     | -1.0    | **0.0004** | 98.6%             |
| 🥈 2 | Hyena     | 16    | 0.0     | **0.0014** | 96.5%             |
| 🥉 3 | Hyena     | 32    | 0.0     | **0.0015** | 92.6%             |
| 4    | Mamba     | 16    | -1.0    | 0.0016     | 93.1%             |
| 5    | Mamba     | 32    | -1.0    | 0.0035     | 90.4%             |
| 6    | Hyena     | 8     | 0.0     | 0.0040     | 91.3%             |
| 7    | Mamba     | 64    | -1.0    | 0.0089     | 82.0%             |
| 8    | Mamba     | 32    | 0.0     | 0.0132     | 87.2%             |
| 9    | Attn      | 16/64 | 0.0     | 0.0166     | ~90%              |

### Convergence Analysis (Improvement from 20k+50k → 20k+100k)

| Model | Patch | Readout | 20k+50k→100k Improvement  | Status                 |
| ----- | ----- | ------- | ------------------------- | ---------------------- |
| Mamba | 8     | -1.0    | 0.0014 → 0.0004 = **71%** | ✅ Still learning fast |
| Hyena | 16    | 0.0     | 0.0038 → 0.0014 = **63%** | ✅ Still learning fast |
| Hyena | 32    | 0.0     | 0.0029 → 0.0015 = **48%** | ✅ Still learning      |
| Mamba | 16    | -1.0    | 0.0033 → 0.0016 = **52%** | ✅ Still learning      |
| Attn  | 8     | 0.0     | 0.0594 → 0.0234 = **61%** | ✅ Still learning fast |
| Mamba | 8     | 0.0     | 0.3860 → 0.3280 = **15%** | ⚠️ Slowing down        |
| Attn  | 1024  | 0.0     | 0.4074 → 0.3094 = **24%** | ⚠️ Slow convergence    |

### Key Findings

1. 🥇 **Mamba (ro=-1, p=8) achieves val_loss 0.0004** - the best by far! 98.6% total improvement
1. 🥈 **Hyena p=16 catches up** to 0.0014 - now matches Mamba's 20k+50k result
1. ✅ **Best models still improving 50-70%** from 20k+50k to 20k+100k - no stagnation!
1. **Attention dramatically improved** - best now at 0.0166 (90%+ improvement)
1. ⚠️ **Mamba ro=0 at p=8 stagnating** - readout=-1.0 is crucial
1. **Optimal patch size = 8-32** for all architectures

### Conclusion

**Models are still learning significantly at 20k+100k iterations!** The best performers (Mamba ro=-1, Hyena) show 50-70% improvement in the last +50k steps. Training could potentially continue to improve further.
