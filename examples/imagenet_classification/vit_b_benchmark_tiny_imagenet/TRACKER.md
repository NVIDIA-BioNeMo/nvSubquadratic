# Hyena Vision Research Tracker

**Objective**: Benchmark Hyena against ViT on vision tasks, focusing on **patchification scaling** and **operator diagnostics**.

## 🔬 Research Phases

### Phase 1: Hyena Diagnostics (Imagenette 160px)
**Goal**: Optimize the Hyena operator for 2D vision signals.

| Experiment | ID | WandB | Config | Status | Val Acc | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Hyena)** | `134927` | `3i7rino1` | `imagenette_hyena_patchify.py` | ⏱️ Timeout | 94.4% | Epoch 417/~680, killed at 12h time limit |
| **Baseline (Attn)** | `134889` | `ym5m6fmv` | `imagenette_attention_patchify.py` | ❌ OOM | 90.1% | OOM'd at Epoch 252 |
| Split WD | `134924` | `0vrk` | `imagenette_hyena_split_wd.py` | ❌ OOM | 92.6% | OOM'd at Epoch 312 |
| Hi-Freq | `134925` | `uh92` | `imagenette_hyena_omega_60.py` | ❌ OOM | 93.6% | OOM'd at Epoch 347, ω₀=60 |
| Deep Filter | `134906` | TBD | `imagenette_hyena_deep_filter.py` | ❌ Failed | - | MLP Depth=5 |
| **Baseline (Attn) v2** | `136444` | TBD | `imagenette_attention_patchify.py` | 🔄 Running | - | Re-run with OOM fix (W&B upload disabled) |
| Split WD v2 | `136442` | TBD | `imagenette_hyena_split_wd.py` | 🔄 Running | - | Re-run with OOM fix (W&B upload disabled) |
| Hi-Freq v2 | `136443` | TBD | `imagenette_hyena_omega_60.py` | 🔄 Running | - | Re-run with OOM fix (W&B upload disabled) |

### Phase 2: Pixel vs Patch Wars (Imagenette 160px)
**Goal**: Push patch size to 1x1 (pixels) to demonstrate Hyena's long-context advantage.

| Patch Size | Seq Len | Hyena Job | Attn Job | Hyena Acc | Attn Acc | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **16x16** (Std) | 100 | `135521` | `135520` | 91.7% | 88.8% | - |
| **10x10** (Current) | 256 | `134888` | `134889` | - | 90.1% | Hyena failed, Attn OOM'd |
| **4x4** | 1,600 | - | - | - | - | - |
| **2x2** | 6,400 | - | - | - | - | - |
| **1x1** | 25,600 | - | ❌ OOM | - | - | - |

### Phase 3: Generalization (Tiny-ImageNet 64px)
**Goal**: Verify "Gold" config on 200 classes.

| Model | Config | Status | Val Acc | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Hyena (Pixel) | `hyena_pixel.py` | 📅 Planned | - | 64x64 resolution (Seq 4096) |
| ViT (Patch 4) | `attention_patchify.py` | 📅 Planned | - | Comparison point |

---

## 📊 Observations & Insights

*   **2026-02-17**: Re-submitted 3 OOM'd runs (Attn=136441, Split WD=136442, Hi-Freq=136443) with W&B checkpoint upload disabled.
*   **2026-02-17**: Phase 2 (16x16) completed! Hyena 91.7% vs Attn 88.8%. Hyena wins at low resolution.
*   **2026-02-17**: Phase 1 results finalized. Hyena Baseline 94.4% (timeout), Hi-Freq 93.6%, Split WD 92.6% (both OOM).
*   **2026-02-16**: Launched baselines (Jobs 134888, 134889) on Imagenette after fixing dataset path issues.

---

## 🛠️ Model Configurations (ViT-B Scale)

| Parameter | Value |
| :--- | :--- |
| Hidden Dim | 768 |
| Layers | 12 |
| Heads | 12 (Attendance only) |
| Expansion | 4.0 (GELU) |
| Precision | bf16-mixed |

---

## 📂 Quick Links
*   **WandB Project**: `nvsubquadratic`
*   **Entity**: `implicit-long-convs`
