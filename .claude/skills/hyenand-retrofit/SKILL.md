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

1. **Native path** — the user's file already uses `nvsubquadratic` (`LazyConfig`, `build_attention_net`, `ViT5Attention`, `ViT5ClassificationNet`). The repo has matched builders:

   - `build_attention_net` + `build_hyena_net` live in `examples/vit5_imagenet/v5_patch/_base_config.py` (pure variants, fixed pattern).
   - `build_hybrid_net` lives in a separate file: `examples/vit5_imagenet/vit5_hybrid/_base_config.py` (pattern-driven). Switching from a pure-attention v5_patch entry to a hybrid means changing both the import path and the entry directory, not just the function name.

   With the right builder picked, the swap is mechanical: replace the builder call, flip `compile_compatible_fftconv`, optionally pick a layer pattern.

1. **Foreign path** — the user's file uses generic PyTorch (`nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `timm`, `transformers`, `diffusers`). You must construct a full Hyena module from scratch and wire it in as a drop-in attention replacement.

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

- Import: replace `build_attention_net` with `build_hyena_net` (pure). For hybrid, import `build_hybrid_net` from `vit5_hybrid._base_config` (different module). Drop the unused builder.
- Builder call inside `get_config()`: same swap.
- For hybrid: add a `LAYER_PATTERN = "..." * (NUM_BLOCKS // len_repeat)` line at module scope, pass it as `layer_pattern=LAYER_PATTERN`.
- Remove `config.compile_compatible_fftconv = False` if present (attention sets this False explicitly; the default is True, and HyenaND needs True).
- Update the module docstring and any inline comments to reflect the new grid math (Hyena adds registers — see `hyena_patch16.py` for the canonical comment style).
- Keep filename convention: `attention_patch16.py` → `hyena_patch16.py`, `full_attention.py` → `full_hyena.py` or `hybrid_hhha.py`.

Native sibling files in this repo do **not** carry an inline `__main__` smoke block — the canonical `hyena_patch16.py` and `hybrid_hhha.py` are tiny config shims. Don't add one. If the user wants to validate the LazyConfig graph, point them at the existing `examples/vit5_imagenet/v5_patch/_smoke_test.py` pattern (separate file, not inline).

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

The full Hyena config has many knobs. Note that only `use_rope` and `gate_nonlinear` are direct `Hyena(...)` kwargs — the rest (`data_dim`, `mask_cfg`, `fft_padding`) belong to the `CKConvND` config that you pass in as `global_conv_cfg`. See `references/defaults.md` for the canonical table and parameter ownership. Short version:

| Modality                                 | data_dim (CKConvND) | mask (CKConvND)                       | fft_padding (CKConvND) | use_rope (Hyena) | gate_nonlinear (Hyena) | ω₀ (SIREN kernel) |
| ---------------------------------------- | ------------------- | ------------------------------------- | ---------------------- | ---------------- | ---------------------- | ----------------- |
| Vision (image classification, diffusion) | 2                   | identity or per-axis Gaussian         | circular or zero       | False            | SiLU                   | 10                |
| Medical 3D segmentation                  | 3                   | per-axis Gaussian                     | zero                   | False            | SiLU                   | 10                |
| Genomics / causal LM                     | 1                   | exponential decay with causal zeroing | causal                 | True             | SiLU                   | 100               |
| PDE fields                               | 2 or 3              | per-axis Gaussian                     | circular               | False            | SiLU                   | 10                |

Read `references/defaults.md` for full parameter lists, init schemes, and the reasoning behind each choice.

### Step 3: wire it in via an adapter (do not assign Hyena directly)

`nn.MultiheadAttention` and `Hyena` have **three incompatible interfaces** that bite you on the first forward pass if you do `block.attn = Hyena(...)` naively:

1. **Return shape.** `nn.MultiheadAttention(..., need_weights=False)` returns a `(out, weights)` tuple; callers commonly unpack `h, _ = self.attn(...)`. `Hyena.forward` returns a single tensor.
1. **Kwargs.** `nn.MultiheadAttention` accepts `need_weights=`, `key_padding_mask=`, `attn_mask=`. `Hyena.forward(query, key, value, cp_group=None, **mixer_kwargs)` does not — extra kwargs flow into the global conv and may crash.
1. **Input layout.** `nn.MultiheadAttention(batch_first=True)` takes `[B, N, C]` (flat token sequence). `Hyena.forward` requires `[B, *spatial, C]` channel-last and reshapes internally to `[B, C, H, W]` (or `[B, C, T]`, `[B, C, D, H, W]`). For ViT, this means undoing the patch flatten — and **handling the CLS token separately**, because `1 + grid_h*grid_w` is never a clean rectangular grid.

You need an adapter wrapper. Skeleton (vision, 2D, with CLS):

```python
# my_model_hyenand.py — generated by hyenand-retrofit

import torch
import torch.nn as nn
from nvsubquadratic.lazy_config import instantiate, LazyConfig
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.utils.qk_norm import L2Norm

from my_model import MyViT  # user's original


def build_hyena_mixer(hidden_dim: int, grid_h: int, grid_w: int) -> nn.Module:
    """Instantiate a 2D HyenaND mixer for a `grid_h × grid_w` patch grid.

    See references/defaults.md for the full per-modality knob list — this is
    the minimum that passes a forward pass; tune mask/short_conv/qk_norm
    against the canonical vision config in
    examples/vit5_imagenet/v5_patch/_base_config.py.
    """
    cfg = LazyConfig(Hyena)(
        global_conv_cfg=LazyConfig(CKConvND)(
            data_dim=2,
            hidden_dim=hidden_dim,
            kernel_cfg=LazyConfig(SIRENKernelND)(
                data_dim=2,
                out_dim=hidden_dim,
                mlp_hidden_dim=32,
                num_layers=3,
                embedding_dim=32,
                omega_0=10.0,
                L_cache=max(grid_h, grid_w),
                use_bias=True,
                hidden_omega_0=1.0,
            ),
            mask_cfg=LazyConfig(GaussianModulationND)(
                data_dim=2,
                num_channels=hidden_dim,
                min_attenuation_at_step=0.1,
                max_attenuation_at_limit=0.95,
                init_extent=1.0,
                parametrization="direct",
            ),
            fft_padding="circular",  # see defaults.md for "zero" alternative
        ),
        short_conv_cfg=LazyConfig(nn.Conv2d)(
            in_channels=3 * hidden_dim,
            out_channels=3 * hidden_dim,
            kernel_size=3,
            groups=3 * hidden_dim,
            padding=1,
            bias=False,
        ),
        gate_nonlinear_cfg=LazyConfig(nn.SiLU)(),
        gate_nonlinear_2_cfg=LazyConfig(nn.Sigmoid)(),
        pixelhyena_norm_cfg=LazyConfig(nn.GroupNorm)(
            num_groups=1, num_channels=hidden_dim
        ),
        qk_norm_cfg=LazyConfig(L2Norm)(),
        use_rope=False,
    )
    return instantiate(cfg)


class HyenaAttnAdapter(nn.Module):
    """Drop-in replacement for ``nn.MultiheadAttention(dim, heads, batch_first=True)``.

    Bridges the three interface gaps:
      - reshapes ``[B, N, C]`` token sequence to ``[B, H, W, C]`` channel-last
        before calling Hyena, then flattens back
      - peels CLS off the front (and any other prefix tokens) so the
        remaining ``H * W`` tokens form a clean spatial grid
      - returns ``(out, None)`` so existing ``h, _ = self.attn(...)`` callers
        keep working
      - swallows ``need_weights``/``attn_mask`` kwargs that Hyena doesn't take
    """

    def __init__(
        self,
        hyena_mixer: nn.Module,
        grid_h: int,
        grid_w: int,
        num_prefix_tokens: int = 1,
    ):
        super().__init__()
        self.mixer = hyena_mixer
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_prefix_tokens = num_prefix_tokens  # e.g. 1 for CLS, 0 for none

    def forward(self, query, key=None, value=None, **_kwargs):
        # Self-attention: query == key == value. We ignore key/value.
        x = query
        prefix, patches = x[:, : self.num_prefix_tokens], x[:, self.num_prefix_tokens :]
        B, N, C = patches.shape
        assert (
            N == self.grid_h * self.grid_w
        ), f"HyenaAttnAdapter: expected {self.grid_h * self.grid_w} patch tokens, got {N}"
        patches_2d = patches.view(B, self.grid_h, self.grid_w, C)  # [B, H, W, C]
        out_2d = self.mixer(patches_2d, patches_2d, patches_2d)  # Q=K=V self-mix
        out = out_2d.view(B, N, C)
        out = torch.cat([prefix, out], dim=1)
        return out, None  # (attn_out, attn_weights) shape contract


class MyViTHyenaND(MyViT):
    def __init__(
        self, *args, grid_h: int, grid_w: int, num_prefix_tokens: int = 1, **kwargs
    ):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            mixer = build_hyena_mixer(self.embed_dim, grid_h, grid_w)
            block.attn = HyenaAttnAdapter(mixer, grid_h, grid_w, num_prefix_tokens)
```

If the host model's attention call site does *not* unpack a tuple (e.g. `x = self.attn(h, h, h)` with no `_`), drop the `(out, None)` tuple and just return `out` from the adapter. If it has no CLS/register prefix, pass `num_prefix_tokens=0`.

### Step 4: smoke-test stub

Append a `__main__` block that constructs the model and runs one forward pass on synthetic input of the user's stated shape. This catches shape mismatches before the user invests in training. Keep this stub for the foreign path — the user has no pre-existing harness, unlike the native path where the LazyConfig graph is exercised by the experiment runner.

## Filename and location convention

- Sibling file, same directory as the user's original
- Name: replace `attention` with `hyena` (or `hybrid_<pattern>` for hybrid), keep all other tokens
- If the user's file has no `attention` token in the name, append `_hyenand` before the extension
- Do not edit the user's original — keep the diff trivial

## Verification

After writing the file:

1. Read it back and confirm the changes you intended actually landed
1. Confirm the only difference from the user's file is the attention→Hyena swap plus any required toggles (`compile_compatible_fftconv`, layer pattern)
1. Confirm imports are syntactically correct (the user can `python -c "from <file> import get_config"` to verify)
1. **Foreign path only:** confirm the smoke-test stub is present *and* that the adapter wires CLS / prefix tokens and the tuple return correctly

## What not to do

- Do not modify the user's original file
- Do not invent new builders if the matched `build_*_net` pair already exists in `_base_config.py`
- Do not skip the foreign-path smoke-test stub — silent shape mismatches at training start are the #1 retrofit failure. (For the native path, omit the stub — the canonical configs don't carry one and the experiment runner exercises the graph.)
- Do not assign `Hyena(...)` directly to an `nn.MultiheadAttention` slot — always wrap with an adapter (see Foreign path Step 3) to bridge the tuple-return, kwargs, and shape mismatches
- Do not pick a hybrid pattern at random; either ask, or take the paper default for the modality (HHHA×3 for 2D vision, HHAA for 3D hierarchical, HHHA repeat for 1D genomics)
- Do not omit `use_rope=True` for 1D autoregressive — without RoPE, positional recall collapses
- Do not omit the per-axis Gaussian mask for ND≥2 unless the user explicitly opts out — it was the difference between Hyena and bidirectional Mamba in the §5.1 color_cond probes

## References

- `references/defaults.md` — full per-modality default tables, init schemes, and rationale
