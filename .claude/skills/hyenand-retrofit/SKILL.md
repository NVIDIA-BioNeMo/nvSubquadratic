---
name: hyenand-retrofit
description: Replace attention in a PyTorch model with HyenaND from the nvSubquadratic library. Covers 1D / 2D / 3D hosts (ViT, U-Net, diffusion, causal LM, hierarchical encoders). Trigger when the user wants a subquadratic alternative to attention, ports a model to HyenaND, swaps `nn.MultiheadAttention` or `F.scaled_dot_product_attention` for a Hyena mixer, builds a striped Hyena LM, or asks "how do I use nvSubquadratic with my model." Phrases like "make my ViT subquadratic," "Hyena layer for my U-Net," "swap attention with FFT convolution," "subquadratic alternative for my 3D segmentation network," or "long-context model with O(L log L) scaling" should all activate this skill.
---

# hyenand-retrofit

Replace attention in a user's model with HyenaND from the nvSubquadratic library. The output is a runnable sibling file alongside the user's original — the original is not modified, with one exception: hierarchical hosts whose natural API is a conditional swap (`use_hyena=True` flag inside the host class) edit in place.

If the user only wants conceptual explanation (no code), answer in chat. This skill is for producing a working file.

## Native path (user is already inside nvSubquadratic)

If the user's file imports nvSubquadratic builders (`build_attention_net`, `LazyConfig(ViT5Attention)`, etc.), the swap is mechanical:

- Pure Hyena: replace `build_attention_net` with `build_hyena_net`, drop `compile_compatible_fftconv = False` if present (Hyena needs the default `True`).
- Hybrid: import `build_hybrid_net` from the matching `_base_config.py`, pass `layer_pattern=...`.

Native sibling files are bare config shims — no `__main__` block; the experiment runner exercises the LazyConfig graph.

The rest of this skill covers the **foreign path** — generic PyTorch hosts using `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, timm, HF, etc.

## Decide four things up front

These four axes are orthogonal. Fix them before writing.

1. **`data_dim ∈ {1, 2, 3}`** — number of spatial axes the mixer sees. Picks `Conv1d/2d/3d` for the short conv and sets `data_dim` on `CKConvND`, `SIRENKernelND`, and `GaussianModulationND`.

1. **`causal ∈ {True, False}`** — autoregressive 1D LMs are causal; vision, segmentation, PDE are bidirectional. Causal sets `is_causal=True` on `CKConvND` (1D only) plus `use_rope=True`, mask `parametrization="exp_decay"`, `omega_0=100`. Bidirectional sets `is_causal=False`, `use_rope=False`, `parametrization="direct"`, `omega_0=10`. Boundary condition (`fft_padding`) is a separate axis-4-style choice — see the FFT-backend section.

1. **Host layout** — `tokens [B, N, C]` (most ViTs, causal LMs) or `feature_map [B, C, *spatial]` (CNNs, U-Nets, hierarchical encoders). Spatial dim count does *not* change this — a 1D, 2D, or 3D feature-map host all use `[B, C, *S] -> [B, *S, C]`.

1. **Return contract** — `(out, None)` tuple if the call site is `h, _ = self.attn(...)` (matches `nn.MultiheadAttention`); bare tensor otherwise.

## Pure or hybrid

Separate decision: replace every attention site (pure) or leave some as attention (hybrid). Hybrids are common in vision and genomics LMs because attention's selectivity complements Hyena's global mixing. If unsure, ask via AskUserQuestion.

- **Pure** — swap every site. Smallest change, cleanest comparison.
- **Hybrid** — *which* sites stay attention is itself an ablation, not a settled choice. Pick any reasonable starting point (e.g., alternate, or hold attention in the deepest stages) and treat the pattern as a knob to sweep. For hierarchical encoders, prefer a per-stage `bool` list (`[True, True, False, False]`) over a `"HHAA"` string — it maps cleanly onto the encoder's stage construction loop.

## The Hyena module

Knobs below that don't depend on the four axes are the dim-agnostic default. Substitute `DATA_DIM`, `CAUSAL`, and `HIDDEN_DIM` from the four-axis decision.

```python
from nvsubquadratic.lazy_config import LazyConfig, instantiate
from nvsubquadratic.modules.hyena_nd import Hyena
from nvsubquadratic.modules.ckconv_nd import CKConvND
from nvsubquadratic.modules.kernels_nd import SIRENKernelND
from nvsubquadratic.modules.masks_nd import GaussianModulationND
from nvsubquadratic.utils.qk_norm import L2Norm
import torch
import torch.nn as nn

kernel_cfg = LazyConfig(SIRENKernelND)(
    data_dim=DATA_DIM,
    out_dim=HIDDEN_DIM,
    mlp_hidden_dim=32,
    num_layers=3,
    embedding_dim=32,
    omega_0=100.0 if CAUSAL else 10.0,
    hidden_omega_0=1.0,
    L_cache=MAX_SPATIAL,  # see foot-gun #3
    use_bias=True,
)

mask_cfg = LazyConfig(GaussianModulationND)(
    data_dim=DATA_DIM,
    num_channels=HIDDEN_DIM,
    min_attenuation_at_step=0.1,
    max_attenuation_at_limit=0.95,
    init_extent=1.0,
    parametrization="exp_decay" if CAUSAL else "direct",
)

global_conv_cfg = LazyConfig(CKConvND)(
    data_dim=DATA_DIM,
    hidden_dim=HIDDEN_DIM,
    kernel_cfg=kernel_cfg,
    mask_cfg=mask_cfg,
    fft_padding="zero",  # "circular" for periodic BCs; per-axis list for mixed
    is_causal=CAUSAL,  # 1D-only; ignored when CAUSAL is False
    fft_backend=(
        "subq_ops" if (DATA_DIM == 2 and not CAUSAL) else "torch_fft"
    ),  # see "FFT backend selection" below
)

ConvND = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[DATA_DIM]
short_conv_cfg = LazyConfig(ConvND)(
    in_channels=3 * HIDDEN_DIM,
    out_channels=3 * HIDDEN_DIM,
    kernel_size=3,
    groups=3 * HIDDEN_DIM,
    padding=1,
    bias=False,
)

mixer = instantiate(
    LazyConfig(Hyena)(
        global_conv_cfg=global_conv_cfg,
        short_conv_cfg=short_conv_cfg,
        gate_nonlinear_cfg=LazyConfig(nn.SiLU)(),
        pixelhyena_norm_cfg=LazyConfig(nn.GroupNorm)(
            num_groups=1, num_channels=HIDDEN_DIM
        ),
        qk_norm_cfg=LazyConfig(L2Norm)(),
        use_rope=CAUSAL,
        rope_base=10000.0,
    )
)
```

**Knob ownership** (common foot-gun):

- `Hyena(...)`: `use_rope`, `rope_base`, `gate_nonlinear_cfg`, `pixelhyena_norm_cfg`, `qk_norm_cfg`, `short_conv_cfg`, `global_conv_cfg`.
- `CKConvND(...)` (passed as `global_conv_cfg`): `data_dim`, `hidden_dim`, `mask_cfg`, `fft_padding`, `is_causal`, `kernel_cfg`, `fft_backend`. Also optional `grid_type` (`"single"`/`"double"`), `use_chunked_fftconv` (memory optimization; works with zero/causal padding only — circular has no chunked variant by design), and `use_fp16_fft` (memory; requires power-of-2 spatial dims for circular padding; not allowed with `subq_ops`). `fft_padding` accepts a single mode string (`"zero"`/`"circular"`) or a per-axis list (`["circular", "zero", ...]`) — see the FFT-backend section.
- `SIRENKernelND(...)` (passed as `kernel_cfg`): `omega_0`, `hidden_omega_0`, `mlp_hidden_dim`, `num_layers`, `embedding_dim`, `L_cache`, `use_bias`, `out_dim`.
- `GaussianModulationND(...)` (passed as `mask_cfg`): `data_dim`, `num_channels`, `min_attenuation_at_step`, `max_attenuation_at_limit`, `init_extent`, `parametrization`.

## FFT backend and boundary conditions

`CKConvND` has two backends. The skeleton above picks one from the four axes; the rule:

- **`fft_backend="subq_ops"`** — optimized CUDA kernel from the optional `subquadratic_ops_torch` package (`pip install subquadratic_ops_torch`). Faster on H100/A100 for the 2D vision case. Constraints (all asserted at construction): `data_dim == 2`, `fft_padding == "zero"`, `is_causal == False`, `use_fp16_fft == False`. The canonical 2D ImageNet configs (`vit5_hybrid`, `v5/hyena_gap_pretrain.py`) use this path.
- **`fft_backend="torch_fft"`** (default) — pure `torch.fft`, supports the full matrix: `data_dim ∈ {1, 2, 3}`, `fft_padding ∈ {"zero", "circular"}` or per-axis list, `is_causal=True` (1D only), optional fp16 and chunked variants.

The skeleton's `fft_backend="subq_ops" if (DATA_DIM == 2 and not CAUSAL) else "torch_fft"` picks the fast path when eligible and falls back otherwise. If the user has not installed `subquadratic_ops_torch`, the import fails at first forward — either swap to `"torch_fft"` or tell them to install it.

**Per-axis mixed boundary conditions.** `fft_padding` accepts a list of modes — one per spatial axis — when the user's domain has different BCs on different axes. Example: a PDE on a periodic channel with hard walls in the cross-stream direction is `fft_padding=["circular", "zero"]` for 2D. This routes through `nvsubquadratic.ops.mixed_fftconv` and forces `fft_backend="torch_fft"` (the subq_ops kernel is single-mode). Default to a single mode if the user hasn't specified mixed; ask if the domain has clearly heterogeneous boundaries.

## Wire it in

`nn.MultiheadAttention` and `Hyena` have three incompatible interfaces — tuple return, kwargs, and input layout. Assigning `Hyena` directly into an attention slot fails. One adapter, parameterized by the four axes:

```python
class HyenaAttnAdapter(nn.Module):
    """Drop-in attention replacement. Parameters reflect the four-axis decision."""

    def __init__(
        self,
        mixer: nn.Module,
        spatial_shape: tuple[int, ...],  # (T,), (H, W), or (D, H, W)
        host_layout: str,  # "tokens" or "feature_map"
        num_prefix_tokens: int = 0,  # CLS / registers; only meaningful for "tokens"
        return_tuple: bool = True,  # (out, None) for nn.MHA contract
    ):
        super().__init__()
        self.mixer = mixer
        self.spatial_shape = spatial_shape
        self.host_layout = host_layout
        self.num_prefix_tokens = num_prefix_tokens
        self.return_tuple = return_tuple

    def forward(self, query, key=None, value=None, **_ignored_kwargs):
        x = query  # self-attention: query == key == value
        if self.host_layout == "tokens":
            # [B, N, C] -> peel prefix -> [B, *spatial, C] -> mix -> flatten back -> re-attach prefix
            prefix, patches = (
                x[:, : self.num_prefix_tokens],
                x[:, self.num_prefix_tokens :],
            )
            B, _, C = patches.shape
            assert (
                patches.shape[1] == torch.prod(torch.tensor(self.spatial_shape)).item()
            ), f"expected {self.spatial_shape} flattened, got {patches.shape[1]} tokens"
            patches_nd = patches.view(B, *self.spatial_shape, C)
            out_nd = self.mixer(patches_nd, patches_nd, patches_nd)
            out = out_nd.view(B, -1, C)
            out = torch.cat([prefix, out], dim=1)
        else:  # "feature_map": [B, C, *S] -> [B, *S, C] -> mix -> back
            x_cl = x.moveaxis(1, -1).contiguous()
            out_cl = self.mixer(x_cl, x_cl, x_cl)
            out = out_cl.moveaxis(-1, 1).contiguous()

        return (out, None) if self.return_tuple else out
```

For hierarchical hosts where the natural API is a `use_hyena=True` flag inside the host class, edit in place — the sibling-file rule doesn't apply. Use an outer `nn.Linear(C, 3*C)` for QKV projection and an `nn.Linear(C, C)` for output projection if you want q/k/v streams to diverge (rather than self-mixing on a single tensor).

**Layout convention.** The library uses `BHL` (channels-first, `[B, H, *spatial]`) as the fast path internally and exposes `_w_reshape` wrappers for channels-last (`BLH`) callers. The adapter above does the explicit `moveaxis`, which works and is the safest pattern for retrofitting. If the user is constructing directly from `nvsubquadratic.ops.*` (rather than going through the full `Hyena` module), prefer the `*_w_reshape` variants instead — see `docs/architecture.md` for the full BHL/BLH convention.

## Foot-guns

These break the first forward pass or the first large-input forward pass.

1. **INT32 unfold overflow in large 3D short conv.** `F.conv3d` uses `im2col` with INT32 indexing. For typical channel counts, spatial extent ≥ 160³ overflows. Symptom: `RuntimeError: Input tensor is too large`. There is no drop-in fix in this repo today (real 3D Hyena configs in `examples/well/v2/*/hyena.py` run at 64³ with plain `nn.Conv3d` and don't hit this); options at larger extent are (a) patch-merge to a smaller working grid, (b) replace the short conv with `nn.Identity()` (the global Hyena conv still mixes spatial information; only fine-grained QKV smoothing is lost), or (c) chunk the short conv along the batch dim. 1D and 2D rarely hit this.

1. **RoPE divisibility.** With `use_rope=True`, the per-block hidden dim must be divisible by `2 * data_dim`: `% 2` for 1D, `% 4` for 2D, `% 6` for 3D. In hierarchical encoders, this must hold at every stage that uses Hyena (`embed_dim * 2^stage`). Validate at construction; fail loudly with a helpful message.

1. **`L_cache` memory in higher dim.** The SIREN coordinate cache allocates an `L^D` volumetric buffer; memory scales roughly as `(2L − 1)^D × D × 4` bytes. For D=3: L=32 → ~3 MB, L=256 → ~1.6 GB, L=512 → ~12.8 GB. Set `L_cache` to the minimum spatial extent you need; the grid expands automatically beyond it.

1. **Shape contract.** `nn.MultiheadAttention(batch_first=True)` returns `(out, attn_weights)` and accepts `need_weights=`/`attn_mask=`/`key_padding_mask=`. `Hyena.forward` returns a single tensor and rejects extra kwargs. The adapter above handles both; never assign `Hyena(...)` directly to an `nn.MultiheadAttention` slot.

1. **Pretrained-weight loading.** Hyena blocks have an entirely different `state_dict` prefix than attention blocks. Loading an attention checkpoint into a Hyena variant produces a wall of missing/unexpected keys. Either filter to shared submodules (patch_embed, downsample, MLP, norms), or skip pretrained loading for Hyena variants.

1. **`subq_ops` backend constraint set.** `fft_backend="subq_ops"` asserts loudly on any one of: `data_dim ≠ 2`, `fft_padding ≠ "zero"` (so circular and per-axis-mixed both disqualify), `is_causal=True`, `use_fp16_fft=True`. The canonical 2D vision case is fine; extending to 3D, circular padding, mixed BCs, or causal LMs requires flipping to `fft_backend="torch_fft"`. Easy to miss when copy-pasting a 2D config as a 3D starting point.

## Smoke-test stub

Append a `__main__` block that constructs the model and runs one forward pass at the user's stated input shape. This catches axis-1 (`data_dim`) and axis-3 (host layout) mismatches before training.

## Filename and location convention

- Sibling file, same directory as the user's original.
- Replace `attention` with `hyena` (or `hybrid` if mixed); keep all other tokens. If the host has no `attention` token in the filename, append `_hyenand` before the extension.
- Exception: conditional-swap hosts (`use_hyena=True` flag inside the host) edit in place.

## Verification

After writing:

1. Re-read the file and confirm the four-axis choices are reflected (`data_dim`, `causal`-derived knobs, layout, return type).
1. Confirm imports are syntactically correct.
1. Run the smoke-test stub on synthetic input of the stated shape.

## Reference configs in this repo

Copy parameter values, not whole files. The repo's curated index lives at `docs/examples/index.md` and `docs/repository_overview.md`; the highest-leverage pointers per axis combination:

| Axes (data_dim, causal, layout) | File                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------- |
| 2D, non-causal, tokens (ViT)    | `examples/vit5_imagenet/v5_patch/_base_config.py`                                        |
| 2D, non-causal, hybrid pattern  | `examples/vit5_imagenet/vit5_hybrid/_base_config.py`                                     |
| 2D, non-causal, feature_map     | `examples/well/v2/active_matter/hyena.py` (FFT_PADDING="circular")                       |
| 3D, non-causal, feature_map     | `examples/well/v2/supernova_explosion_64/hyena.py` (zero) / `MHD_64/hyena.py` (circular) |
| 1D, non-causal, smallest        | `examples/mnist_classification/ccnn_4_160_hyena_rope_qknorm.py`                          |
| Diffusion (HF diffusers)        | `examples/imagenet_diffusion/ccnn_12_768_hyena_qknorm.py`                                |

For full per-module API and the math primer on the FFT ops, see `docs/api_reference/modules.rst` and `docs/architecture.md` — both are auto-built from the rich inline docstrings.
