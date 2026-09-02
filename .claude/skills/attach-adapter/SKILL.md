---
name: attach-adapter
description: Attach a zero-init parallel Hyena2D/conv adapter to a pretrained ViT-style vision encoder (timm, HuggingFace, DINOv2) for parameter-efficient fine-tuning. Use when asked to "add an adapter to <encoder>", "adapt DINOv2/UNI/a ViT with Hyena", "set up a PEFT fine-tune with the parallel branch", or to run C0/C1/C2 ablation arms on an external backbone.
---

# Attach a zero-init parallel adapter to a pretrained encoder

Wraps every attention module of a frozen ViT-style encoder with

```
out = Attn(x) + W_up( inner_mixer( W_down(x_patches) ) )
```

`W_up` is zero-initialised, so the encoder's function is **bit-exactly unchanged** at
step 0 and its pretrained representation is not disturbed. Training moves `W_up` off
zero and the function departs continuously.

Library: `nvsubquadratic/modules/encoder_adapter.py`.

**Do not modify** `Hyena`, `QKVSequenceMixer`, `CKConvND`, `SIRENKernelND`, or the
encoder's own modules. This is purely additive.

______________________________________________________________________

## Step 1 — Inspect the encoder

Never guess the token layout. Run this and read the output:

```python
import torch
from nvsubquadratic.modules.encoder_adapter import find_attention_modules

targets = find_attention_modules(model)
print(f"attention modules: {len(targets)}")
for name, mod in targets[:3]:
    print(f"  {name}: {type(mod).__name__}")

# Probe the actual token count rather than trusting attributes
with torch.no_grad():
    feats = model.forward_features(sample_x)  # timm
print("token sequence:", feats.shape)  # [B, T, C]
```

Determine and write down four things:

| Quantity                | How to get it                                                |
| ----------------------- | ------------------------------------------------------------ |
| `hidden_dim` (C)        | `model.embed_dim`, or the last dim of the token sequence     |
| `num_prefix_tokens` (P) | `model.num_prefix_tokens` (timm), else `T - grid_h*grid_w`   |
| `prefix_position`       | `"first"` for timm / HF / DINOv2; `"last"` for in-repo ViT-5 |

Sanity check: **`T - P` must equal `grid_h * grid_w`.** If it does not, the layout
assumption is wrong — stop and re-derive it. Do not proceed on a guess.

## Step 2 — Report and confirm

Report the four quantities plus the proposed rank and inner mixer to the user, and
confirm before mutating the model. Silent mis-attachment produces a model that trains
happily and is wrong; the cost of one confirmation is much lower than the cost of a
misattributed result.

## Step 3 — Choose the inner mixer

The three arms share an identical scaffold — same zero-init, same rank, same frozen
backbone — and differ only in what sits between `W_down` and `W_up`:

| Arm | `inner_mixer_factory`                                | Tests                                     |
| --- | ---------------------------------------------------- | ----------------------------------------- |
| C0  | `None` → `nn.Identity`                               | Capacity: does the bottleneck alone help? |
| C1  | a 2-D depthwise conv over `[B, H, W, r]`             | Local prior                               |
| C2  | `lambda: instantiate(<QKVSequenceMixer(Hyena) cfg>)` | Global implicit filter                    |

`inner_mixer_factory` is called **once per attention module**, so each block gets
independent weights. Passing a shared module instance instead would tie all blocks
together — a silent and hard-to-spot bug.

For C2, build the Hyena config with `fft_padding="zero"` + `grid_type="double"`
(images are not periodic), `data_dim=2`, `is_causal=False`, `hidden_dim=rank`, and
`L_cache=<a typical grid side>` (a cache hint only — the kernel extends at runtime). `make_hyena_inner_cfg()` in `examples/hyena2d_parallel_adapter/_base_lg.py` or `make_hyena2d_inner_mixer_cfg()` in
`examples/hyena2d_parallel_adapter/_base.py` is a working reference.

Do not run C2 alone. C0 and C1 are what make a C2 result interpretable.

**Resolution is handled for you.** The adapter reshapes patch tokens to the 2-D grid
with dimensions derived *per forward pass*, so one instance serves every resolution
an any-resolution encoder emits — verified on C-RADIOv4's set (224/256/384/512/768/
1024 px) with a single instance. The operators below are already resolution-agnostic:
a Hyena kernel is regenerated at `2L-1` per axis on each call. Pass the inner mixer
directly (it consumes `[B, H, W, r]`); do **not** wrap it in `ViT5HyenaAdapter`, whose
`grid_w` is fixed at construction and would pin you to one resolution.

For non-square inputs call `adapter.set_grid_hw((H, W))` before the forward pass. A
non-square token count with no explicit grid raises rather than guessing — silently
mixing over a wrong layout is the worse failure.

## Step 4 — Attach

```python
from nvsubquadratic.modules.encoder_adapter import attach_parallel_adapters

paths = attach_parallel_adapters(
    model,
    hidden_dim=C,
    rank=32,
    inner_mixer_factory=factory,
    num_prefix_tokens=P,
    prefix_position="first",
)
```

Prefix tokens (CLS, registers) are **bypassed**: the adapter reads the patch tokens
only and writes exactly zero at prefix positions. CLS keeps updating through the
attention branch. This is both correct — CLS has no grid coordinate — and necessary,
since `T` generally does not factor into a rectangle (DINOv2-with-registers at 224/14
gives `T = 261`, and `261 % 16 = 5`).

## Step 5 — Verify (do not skip)

```python
from nvsubquadratic.modules.encoder_adapter import (
    verify_zero_init_identity,
    check_gradient_unlock,
)

ok, report = verify_zero_init_identity(fresh_model, sample_x, attach_fn)
assert ok, report  # torch.equal — bit-exact, not allclose

model.train()
# NOT `o.sum()` -- see the warning below.
ok, report = check_gradient_unlock(model, sample_x, lambda o: (o**2).sum())
assert ok, report
```

**Never use `o.sum()` as the probe loss on a feature extractor.** Most ViTs end in
LayerNorm, whose output is zero-mean along the channel axis, so `sum()` is
identically ~0 for *any* input and its gradient is **exactly zero**. That looks
exactly like a dead adapter and will send you debugging working code. Use
`(o**2).sum()`, a random linear head, or the real task loss.

`verify_zero_init_identity` uses `torch.equal`, not `allclose`, deliberately: an
approximate match hides real wiring bugs. It runs in eval mode because dropout and
stochastic depth break determinism on their own.

Expected gradient pattern at step 0: `W_up.grad != 0`, `W_down.grad == 0` exactly.
That is correct LoRA-style behaviour — `W_up` moves first and the rest unlocks after.
A branch that stays dead forever is indistinguishable from "the adapter didn't help"
in a loss curve, so confirm `W_down.grad != 0` after one optimiser step too.

**If the identity check fails, stop and report it.** It means the approach needs
rethinking, not debugging around. Likely causes, in order:

1. `prefix_position` or `num_prefix_tokens` wrong → patch grid misaligned
1. A `W_up` with `bias=True` or non-zero weights
1. Model not in eval mode
1. Encoder attention that takes a mask or returns extra state the shim drops

## Step 6 — Freeze and report

```python
from nvsubquadratic.modules.encoder_adapter import freeze_except_adapters

trainable, total = freeze_except_adapters(model, also_train=("head",))
print(f"trainable {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

Report the trainable count per arm. C0/C1/C2 differ in parameter count by design
(C2's SIREN kernel is the bulk); the ablation isolates *structure at fixed rank*, not
matched parameters. Say which you are testing so the result is not misread.

______________________________________________________________________

## Backbone reference

| Backbone                                      | Attention signature           | Shim            | Prefix                                         |
| --------------------------------------------- | ----------------------------- | --------------- | ---------------------------------------------- |
| timm ViT / DINOv2                             | `forward(x) -> Tensor`        | passthrough     | `model.num_prefix_tokens`                      |
| HF `ViTModel`, `Dinov2Model`                  | returns `(ctx, attn_weights)` | take `[0]`      | 1 (CLS), +4 with registers                     |
| in-repo ViT-5                                 | `forward(x) -> Tensor`        | passthrough     | `prefix_position="last"`                       |
| **C-RADIOv4** (`nvidia/C-RADIOv4-{H,SO400M}`) | timm ViT under `model.model`  | passthrough     | **10** = 3 CLS (one per teacher) + 7 registers |
| MONAI SwinUNETR / C-RADIO **ViTDet mode**     | windowed                      | **unsupported** | —                                              |

`AttentionShim` handles the tuple and `ModelOutput` cases automatically. Windowed
attention (Swin) needs genuine design work — the grid is per-window, not global —
so flag it rather than forcing it.

**C-RADIOv4 specifics.** `vit_huge_patch16_224` (H: 1280-dim, 32 blocks) or
`vit_so400m_patch16_224` (SO400M: 1152-dim, 27 blocks), patch 16 both. Prefix count
comes from `cls_token.py`: `num_registers = register_multiple - (num_tokens % register_multiple)` = `10 - 3 = 7`, so `num_skip = 10`. Prefer reading
`model.num_summary_tokens` at runtime over hard-coding it. Its `patch_embed` is
replaced by a `ViTPatchGenerator` (CPE), so `model.patch_embed.grid_size` does not
exist — probe the token count instead.

**Do not enable ViTDet mode when adapting.** `ViTDetHook` registers a forward
pre-hook that reshapes patches into windows (`DEFAULT_NUM_WINDOWED = 5`, so every
6th block is global and the rest windowed at 6x6-32x32 tokens). A 2-D global filter
over a windowed sequence is not well-defined.

## FiLM conditioning — required for C2 to be HyenaND

**A Hyena branch without `film_cfg` is a static implicit long convolution, not
HyenaND.** Wessels et al. (arXiv:2607.19378) define the operator as

```
K_i(x) = w(c_i) ⊙ f_θ(c_i; z(x))
```

with the dependence on control variable `z(x)` implemented by FiLM over the SIREN
hidden layers. Input dependence is constitutive, not an add-on. If you build C2 with
`film_cfg=None`, label it a static-kernel arm and do not report it as HyenaND.

`conditioning_source` selects where `z(x)` comes from:

| Value                    | `z(x)`                      | When                                                                                                                                       |
| ------------------------ | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `"patch_mean"` (default) | mean of patch tokens        | **Frozen encoders.** Depends on nothing the base model had to learn for you.                                                               |
| `"prefix"`               | mean of CLS/register tokens | Mirrors `RegisterPooling` in the ViT-5 ImageNet configs. Sound when registers were trained as part of a Hyena model; weaker off-the-shelf. |
| `"none"`                 | —                           | Static-kernel ablation, or any mixer that rejects the kwarg (`nn.Identity`, depthwise conv).                                               |

Two properties keep this cheap and safe: `z(x)` is pooled once per instance so the
inner mixer still issues one ND FFT call, and FiLM initialises to `(γ, β) = (1, 0)`
— unmodulated at init, so the zero-init `W_up` bit-exactness guarantee still holds.

Build the kernel with a `film_cfg` sized for `cond_dim=hidden_dim`. Pair `nn.Identity`
and depthwise conv with `"none"` — they will raise on the kwarg, and conditioning is
meaningless for them.

Caveat: neither the paper nor this repo has a clean FiLM-on vs FiLM-off ablation at
matched settings. If that comparison matters, run `"none"` as an explicit arm.

## Known results worth carrying in

**Paper (ImageNet-1K, ViT-5-Small 22M, patch 16):** attention 81.8%, pure HyenaND
81.5%, `(HA)×6` **82.1%**, `(HHHA)×3` 82.0%. Hybrids are **layer-interleaved**; the
paper contains no parallel in-block variant, which is the gap this adapter explores.

**In-repo toy (196 tokens, 56×56 canvas, 14×14 grid):** C1 (local conv) beat C2 on
every displacement bucket. Two confounds, both worth avoiding on the next run:

1. That C2 had **no `film_cfg`** — it was a static kernel, not HyenaND.
1. At 196 tokens a stack of 3×3 convs already spans the grid within a few layers, so
   global mixing has nothing to buy.

Expect a fair test only at **≥784 tokens** and **with conditioning enabled**. See
`reports/hyena2d_parallel_adapter/REPORT.md`.
