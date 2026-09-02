# Next Steps — Realistic Pretrained Encoder Experiment

## Why the toy is insufficient

The toy used a ViT trained from scratch on a controlled task. The real target is
fine-tuning **Pillar-0 / Atlas** (a pretrained 3D radiology encoder) on a new domain
(e.g. pathology) while preserving its existing representations. Three things the toy
cannot test:

1. **Forgetting safety** — no valuable pretrained representation exists to protect.
1. **Global mixing necessity** — 196 tokens is too small; local conv is already global.
1. **Domain shift** — the pretrain and fine-tune tasks were structurally identical.

______________________________________________________________________

## Proposed next experiment: pretrained ViT + domain shift

### Setup

**Backbone**: a publicly available pretrained vision encoder, e.g.:

- `DINOv2-small` (ViT-S/14, ImageNet) — easy to load, well-understood
- `UNI` or `CONCH` (pathology foundation models) if pathology is the target domain
- A ViT-B/16 pretrained on natural images from `timm`

**Image resolution**: 224×224 with patch size 14 → **16×16 = 256 tokens**, or
224×224 with patch size 8 → **28×28 = 784 tokens**. At 784 tokens, a local 3×3 conv
covers only 0.4% of the sequence in one layer — global routing is no longer free.

**Fine-tune dataset**: a domain-shifted classification task, e.g.:

- PCam (PatchCamelyon) — binary tumour classification from H&E patches, 96×96
- NCT-CRC-HE — 9-class colorectal cancer tissue classification
- Or any pathology slide dataset if available internally

**Joint pretrain for forgetting test**: before fine-tuning, hold out a fraction of
ImageNet-val classes. After fine-tuning on pathology, measure accuracy drop on those
held-out classes. This is the forgetting proxy.

### Arms

| Arm             | Description                                              |
| --------------- | -------------------------------------------------------- |
| Head only       | Frozen backbone, train linear head                       |
| Full fine-tune  | All params unfrozen                                      |
| C2 adapter      | Zero-init parallel Hyena2D, frozen backbone              |
| C1 adapter      | Zero-init local conv, frozen backbone (ablation)         |
| LoRA (optional) | Standard LoRA on Q/V for comparison with PEFT literature |

### Primary metrics

1. **New task accuracy** (pathology) — does the adapter match or beat full fine-tune?
1. **Forgetting** (ImageNet accuracy drop) — zero-init arms should preserve far more.
1. **Displacement-dependence** — if using a spatial task, bucket by displacement.

### Why this test is decisive

If C2 > C1 at 784 tokens but C2 ≈ C1 at 196 tokens (consistent with our current
result), the mechanism story is confirmed: global mixing only helps when the task
genuinely requires routing across the full sequence, not when local chains suffice.

If C2 ≈ C1 at 784 tokens too, the null result is stronger: Hyena2D adds no benefit
over a much cheaper local conv in the fine-tuning regime, regardless of scale. That
is also a publication-worthy finding.

### Engineering prerequisites

- Adapter attachment: `attach_parallel_hyena_adapter(model)` — walk the pretrained
  model's attention blocks and wrap each `sequence_mixer` (or equivalent) with
  `ViT5ParallelHyenaSequenceMixer`. Requires matching the block interface of the
  chosen backbone.
- For non-ViT5 backbones (e.g. HuggingFace `ViTModel`): write a thin wrapper that
  maps the HF attention module to the same `[B, T, C] → [B, T, C]` interface, then
  slot in the adapter.
- Grid size for Hyena: `grid_w = image_size // patch_size`. At 224/14 = 16 and
  224/8 = 28. `L_cache` in `SIRENKernelND` should be set to `grid_w`.

### Suggested first run

Start with DINOv2-small on PCam (binary, small dataset, fast to run):

- 2 GPUs, 3 seeds per arm, ~1 hour per arm
- Measures: PCam accuracy + ImageNet-val accuracy (on the ~100 held-out classes)

This is the minimal experiment that tests forgetting safety and domain-shift
generalisation simultaneously.
