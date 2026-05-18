---
name: hyenand-retrofit
description: Replace attention in a PyTorch model with HyenaND from the nvSubquadratic library, with paper-grounded default hyperparameters (SIREN ω₀, register count, per-axis Gaussian mask, FFT padding, hybrid layer pattern). Use this skill whenever the user wants to swap attention for a subquadratic operator, port a ViT / U-Net / diffusion / genomics LM to HyenaND, convert a `nn.MultiheadAttention` or `F.scaled_dot_product_attention` site to a Hyena mixer, build a striped Hyena LM, set up a HyenaND experiment config, or asks "how do I use nvSubquadratic with my model." Trigger even when the user does not explicitly name HyenaND — phrases like "make my ViT subquadratic," "subquadratic alternative to attention for my 3D U-Net," "Hyena layer for my transformer," "swap attention with FFT convolution," or "long-context model with O(L log L) scaling" should all activate this skill.
---

# hyenand-retrofit

Replace attention in a user's model with HyenaND from the nvSubquadratic library. The output is a runnable sibling file alongside the user's original — the original is not modified.

## When to use

- User has an attention-based model (ViT, U-Net, DiT, causal LM, hierarchical encoder) and wants a subquadratic alternative
- User explicitly mentions HyenaND, Hyena, nSubQ, nvSubquadratic, or "subquadratic attention"
- User is inside the `nvSubquadratic-private` repo and wants to add a new attention/Hyena variant
- User asks about scaling to long contexts (long genomes, high-resolution images, 3D volumes, PDE grids)

If the user only wants conceptual explanation (no code), answer in chat. This skill is for producing a working file.

## The two paths

Decide which one applies *before* writing anything:

1. **Native path** — the user's file already uses `nvsubquadratic` (`LazyConfig`, `build_attention_net`, `ViT5Attention`, `ViT5ClassificationNet`). The repo has matched `build_attention_net` / `build_hyena_net` / `build_hybrid_net` builders, so the swap is mechanical: replace the builder call, flip `compile_compatible_fftconv`, optionally pick a layer pattern.

2. **Foreign path** — the user's file uses generic PyTorch (`nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `timm`, `transformers`, `diffusers`). You must construct a full Hyena module from scratch and wire it in as a drop-in attention replacement.

Look at the user's file. If you see `from nvsubquadratic` imports or `LazyConfig(ViT5Attention)`, take the native path. Otherwise foreign.

## Native path workflow

The repo factors attention vs Hyena into builder functions. The user's entry file is almost always just a thin wrapper around one of:

- `build_attention_net(patch_size)` — pure attention
- `build_hyena_net(patch_size)` — pure Hyena
- `build_hybrid_net(layer_pattern="...", patch_size=...)` — mixed, where pattern is a string of `H`/`A` characters (one per block)

### Step 1: ask the user which target

Use AskUserQuestion if not already specified:

- **Pure Hyena** — strongest when geometry + global structure dominate (PDE fields, high-resolution vision, long genomes)
- **Hybrid** — strongest when selectivity still matters (ImageNet classification, medical segmentation). Paper winners:
  - 2D ImageNet ViT-Small: `(HA)×6` (best) or `(HHHA)×3`
  - 3D medical SwinUNETR-style: `HHAA` (hierarchical — Hyena in early high-resolution stages, attention later)
  - 1D genomics (1B striped LM, Evo2): one A per 4 blocks (`HHHA` repeat, H₂ mixing) was best in §5.2

### Step 2: write the sibling file

Copy the entry-file structure verbatim, then change:

- Import: replace `build_attention_net` with `build_hyena_net` (pure) or `build_hybrid_net` (hybrid). Drop the unused builder.
- Builder call inside `get_config()`: same swap.
- For hybrid: add a `LAYER_PATTERN = "..." * (NUM_BLOCKS // len_repeat)` line at module scope, pass it as `layer_pattern=LAYER_PATTERN`.
- Remove `config.compile_compatible_fftconv = False` if present (attention sets this False explicitly; the default is True, and HyenaND needs True).
- Update the module docstring and any inline comments to reflect the new grid math (Hyena adds registers — see `hyena_patch16.py` for the canonical comment style).
- Keep filename convention: `attention_patch16.py` → `hyena_patch16.py`, `full_attention.py` → `full_hyena.py` or `hybrid_hhha.py`.

### Step 3: smoke-test stub

Append a `__main__` block that constructs the config and prints `repr(config.net)`. The user can run this on CPU to verify imports resolve and the LazyConfig graph is well-formed before committing GPU time.

```python
if __name__ == "__main__":
    cfg = get_config()
    print(cfg.net)
```

## Foreign path workflow

The user has, e.g., a `timm` ViT, a HF transformers model, or a hand-written PyTorch transformer. You need to construct a Hyena module yourself.

### Step 1: identify the attention sites

Look for any of:

- `nn.MultiheadAttention`
- `F.scaled_dot_product_attention`
- `flash_attn.*` calls
- Custom attention modules (look for `softmax(Q @ K.T / sqrt(d))` or `attn_drop`/`proj_drop` member names)
- timm `Attention` class, HF `*Attention` classes

### Step 2: build the Hyena replacement

For each attention site, emit a Hyena module config. The minimum spec:

```python
from nvsubquadratic.lazy_config import LazyConfig
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.modules.rms_norm import RMSNorm
from nvsubquadratic.utils.qk_norm import L2Norm
import torch
```

The full Hyena config has many knobs — see `references/defaults.md` for the canonical table. The short version of the defaults by modality:

| Modality | data_dim | mask | fft_padding | use_rope | gate_nonlinear | typical ω₀ |
|----------|----------|------|-------------|----------|----------------|------------|
| Vision (image classification, diffusion) | 2 | identity or per-axis Gaussian | circular or zero | False | SiLU | 10 |
| Medical 3D segmentation | 3 | per-axis Gaussian | zero | False | SiLU | 10 |
| Genomics / causal LM | 1 | exponential decay with causal zeroing | causal | True | SiLU | 10 |
| PDE fields | 2 or 3 | per-axis Gaussian | circular | False | SiLU | 10 |

Read `references/defaults.md` for full parameter lists, init schemes, and the reasoning behind each choice.

### Step 3: wire it in

In the sibling file, subclass or monkey-patch the user's model to replace each attention module with the Hyena module. Prefer subclassing — it's easier to read and rollback. Example skeleton:

```python
# my_model_hyenand.py — generated by hyenand-retrofit

import torch
from nvsubquadratic.lazy_config import instantiate, LazyConfig
from nvsubquadratic.modules.hyena_nd import Hyena
# ... other imports per defaults.md ...

from my_model import MyViT  # user's original

def build_hyena_mixer(hidden_dim: int, grid_h: int, grid_w: int) -> torch.nn.Module:
    """Drop-in replacement for nn.MultiheadAttention(hidden_dim, num_heads)."""
    cfg = LazyConfig(Hyena)(
        global_conv_cfg=LazyConfig(CKConvND)(
            data_dim=2,
            hidden_dim=hidden_dim,
            kernel_cfg=LazyConfig(SIRENKernelND)(
                data_dim=2, out_dim=hidden_dim, mlp_hidden_dim=32,
                num_layers=3, embedding_dim=32, omega_0=10.0,
                L_cache=max(grid_h, grid_w), use_bias=True, hidden_omega_0=1.0,
            ),
            # ... see defaults.md ...
        ),
        # ... see defaults.md ...
    )
    return instantiate(cfg)

class MyViTHyenaND(MyViT):
    def __init__(self, *args, grid_h: int, grid_w: int, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            block.attn = build_hyena_mixer(self.embed_dim, grid_h, grid_w)
```

### Step 4: smoke-test stub

Append a `__main__` block that constructs the model and runs one forward pass on synthetic input of the user's stated shape. This catches shape mismatches before the user invests in training.

## Filename and location convention

- Sibling file, same directory as the user's original
- Name: replace `attention` with `hyena` (or `hybrid_<pattern>` for hybrid), keep all other tokens
- If the user's file has no `attention` token in the name, append `_hyenand` before the extension
- Do not edit the user's original — keep the diff trivial

## Verification

After writing the file:

1. Read it back and confirm the changes you intended actually landed
2. Confirm the only difference from the user's file is the attention→Hyena swap plus any required toggles (`compile_compatible_fftconv`, layer pattern)
3. Confirm imports are syntactically correct (the user can `python -c "from <file> import get_config"` to verify)
4. Confirm the smoke-test stub is present

## What not to do

- Do not modify the user's original file
- Do not invent new builders if the matched `build_*_net` pair already exists in `_base_config.py`
- Do not skip the smoke-test stub — silent shape mismatches at training start are the #1 retrofit failure
- Do not pick a hybrid pattern at random; either ask, or take the paper default for the modality (HHHA×3 for 2D vision, HHAA for 3D hierarchical, HHHA repeat for 1D genomics)
- Do not omit `use_rope=True` for 1D autoregressive — without RoPE, positional recall collapses
- Do not omit the per-axis Gaussian mask for ND≥2 unless the user explicitly opts out — it was the difference between Hyena and bidirectional Mamba in the §5.1 color_cond probes

## References

- `references/defaults.md` — full per-modality default tables, init schemes, and rationale
