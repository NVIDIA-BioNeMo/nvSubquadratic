# ViT-5-Small ImageNet-1k — v5 Patch-Size Ablation

## Goal

Ablate patch size (2, 4, 8, 16) for **Attention** and **Hyena + Gaussian mask**
on ImageNet-1k pretraining. All runs share the same training recipe (800 epochs,
LAMB lr=4e-3, wd=0.05, cosine schedule, 3-Augment, Mixup/CutMix, EMA 0.99996).

The effective batch size is held constant at **2048** across all patch sizes.
Per-GPU batch size is 256; if a configuration OOMs at batch 256, gradient
accumulation is used to maintain the effective batch size (e.g. micro-batch 128
with 2 accumulation steps per GPU, or 64 with 4 steps).

## Configs

| File                             | Model                       | Compile                    | FFT backend |
| -------------------------------- | --------------------------- | -------------------------- | ----------- |
| `attention_pretrain.py`          | ViT5Attention (CLS, 4 regs) | max-autotune               | —           |
| `hyena_gap_gaussian_pretrain.py` | Hyena + Gaussian mask (GAP) | max-autotune-no-cudagraphs | subq_ops    |

## Patch-size overrides

`num_patches_h`, `num_patches_w` (Attention) and `grid_w`, `L_cache` (Hyena) are
computed dynamically via `${eval:'${net.image_size} // ${net.patch_size}'}` interpolators.
Only `net.patch_size` needs to be overridden:

```
net.patch_size=P
```

If OOM at batch 256, add: `dataset.batch_size=<micro_batch>  trainer.accumulate_grad_batches=<accum_steps>`
(keep `micro_batch * accum_steps * num_gpus = 2048`).

## Results — Attention

| Patch | Tokens/img | Batch/GPU | Grad Accum | GPUs | WandB Run | val/acc_ema | it/s (1 GPU) |
| ----- | ---------- | --------- | ---------- | ---- | --------- | ----------- | ------------ |
| 16    | 196        | 256       | 1          | 8    |           |             | 8.5          |
| 8     | 784        | 256       | 1          | 8    |           |             | 5.2          |
| 4     | 3,136      | 64        | 4          | 8    |           |             | —            |
| 2     | 12,544     | 16        | 16         | 8    |           |             | —            |

## Results — Hyena + Gaussian Mask (subq_ops)

| Patch | Tokens/img | Batch/GPU | Grad Accum | GPUs | WandB Run | val/acc_ema | it/s (1 GPU) |
| ----- | ---------- | --------- | ---------- | ---- | --------- | ----------- | ------------ |
| 16    | 196        | 256       | 1          | 8    |           |             | 7.3          |
| 8     | 784        | 256       | 1          | 8    |           |             | 5.1          |
| 4     | 3,136      | 64        | 4          | 8    |           |             | —            |
| 2     | 12,544     | 16        | 16         | 8    |           |             | —            |

## Memory Probing Notes (1×H100 80GB, with torch.compile)

Attention uses `max-autotune` (CUDA graphs); Hyena+G uses `max-autotune-no-cudagraphs` + `subq_ops`.
Both models OOM at the same batch thresholds despite different compile modes.

| Patch | Batch 256         | Batch 128  | Batch 64      | Batch 32   | Batch 16      |
| ----- | ----------------- | ---------- | ------------- | ---------- | ------------- |
| 16    | Attn OK, Hyena OK | —          | —             | —          | —             |
| 8     | Attn OK, Hyena OK | —          | —             | —          | —             |
| 4     | OOM (both)        | OOM (both) | **OK (both)** | —          | —             |
| 2     | OOM (both)        | —          | —             | OOM (both) | **OK (both)** |

Grad accumulation set so `batch * accum * 8 GPUs = 2048`.
