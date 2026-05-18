# HyenaND default hyperparameters by modality

These defaults are grounded in the paper and the repo's reference configs (`examples/imagenet_classification/ccnn_7_512_hyena*.py`, `examples/vit5_imagenet/v5_patch/_base_config.py`, `examples/well/v2/*.py`, `examples/mnist_classification/ccnn_4_160_hyena_rope_qknorm.py`). Use them as the starting point for any retrofit; tune only if there's a specific reason.

## Common (all modalities)

| Knob | Value | Why |
|------|-------|-----|
| `omega_0` (SIREN first-layer ω) | 10.0 for vision/PDE, 100.0 for genomics | Frequency band; higher for 1D where positional recall matters more |
| `hidden_omega_0` | 1.0 | SIREN hidden-layer ω; preserves init scale |
| `mlp_hidden_dim` (SIREN width) | 32 | Small MLP; SIREN is shallow but wide enough to express smooth kernels |
| `num_layers` (SIREN depth) | 3 | Paper-standard depth |
| `embedding_dim` (SIREN coordinate embedding) | 32 | First-layer positional embedding |
| `use_bias` | True | |
| First-layer init | `U(-1/sqrt(N), 1/sqrt(N))` | Preserves unit-variance pre-activations (§3.2.3) |
| FiLM heads init | zero | Training starts from the unmodulated baseline (§3.2.3) |

## Vision (2D, image classification or diffusion)

```python
data_dim = 2
use_rope = False
fft_padding = "circular"  # or "zero" for CLS-row variants
mask_cfg = LazyConfig(GaussianModulationND)(
    data_dim=2,
    num_channels=hidden_dim,
    min_attenuation_at_step=0.1,
    max_attenuation_at_limit=0.95,
    init_extent=1.0,
    parametrization="direct",
)
gate_nonlinear_cfg = LazyConfig(torch.nn.SiLU)()
gate_nonlinear_2_cfg = LazyConfig(torch.nn.Sigmoid)()
qk_norm_cfg = LazyConfig(L2Norm)()
pixelhyena_norm_cfg = LazyConfig(torch.nn.GroupNorm)(num_groups=1, num_channels=hidden_dim)
short_conv = LazyConfig(torch.nn.Conv2d)(
    in_channels=3 * hidden_dim, out_channels=3 * hidden_dim,
    kernel_size=3, groups=3 * hidden_dim, padding=1, bias=False,
)
```

For ViT-5-style backbones with registers, use `ViT5HyenaAdapter` and `RegisterPooling` (see `examples/vit5_imagenet/v5_patch/_base_config.py` lines 270–353).

**Hybrid patterns (12 blocks):**
- Best on ImageNet: `(HA)×6` (82.1 top-1)
- Second-best: `(HHHA)×3` (82.0 top-1)
- Pure: `H×12` (81.5 top-1, matches attention baseline)

## Medical 3D segmentation (SwinUNETR-style hierarchical encoders)

```python
data_dim = 3
use_rope = False
fft_padding = "zero"
mask_cfg = LazyConfig(GaussianModulationND)(data_dim=3, num_channels=hidden_dim, ...)
short_conv = LazyConfig(torch.nn.Conv3d)(
    in_channels=3 * hidden_dim, out_channels=3 * hidden_dim,
    kernel_size=3, groups=3 * hidden_dim, padding=1, bias=False,
)
```

**Stage patterns (4-stage encoder, paper §5.5 PanTS, mean Dice):**

| Pattern | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Mean Dice |
|---------|---------|---------|---------|---------|-----------|
| `AAAA` (Swin baseline) | Attn | Attn | Attn | Attn | 0.7496 |
| `HHHH` (all-Hyena) | Hyena | Hyena | Hyena | Hyena | 0.7510 |
| `HAHA` (striped) | Hyena | Attn | Hyena | Attn | 0.7535 |
| `HHAA` (hierarchical, **best**) | Hyena | Hyena | Attn | Attn | **0.7559** |

Recommend `HHAA` by default. Memory savings (~11%) are roughly invariant to placement.

## Genomics / 1D causal LM (striped Hyena)

```python
data_dim = 1
use_rope = True              # critical — without RoPE, positional recall collapses
rope_base = 10000.0
fft_padding = "causal"       # preserves autoregressive structure
mask_cfg = LazyConfig(GaussianModulationND)(  # exponential decay with causal zeroing
    data_dim=1, num_channels=hidden_dim,
    min_attenuation_at_step=0.1,
    max_attenuation_at_limit=0.95,
    init_extent=1.0,
    parametrization="exp_decay",
)
gate_nonlinear_cfg = LazyConfig(torch.nn.Identity)()  # linear gating for AR
short_conv = LazyConfig(torch.nn.Conv1d)(
    in_channels=3 * hidden_dim, out_channels=3 * hidden_dim,
    kernel_size=3, groups=3 * hidden_dim, padding=1, bias=False,
)
```

**Mixing ratios (Evo2-1B, 8192-bp sequences, §5.2, perplexity lower is better):**

| Config | Pattern (24 blocks) | Validation PPL |
|--------|---------------------|----------------|
| `T` (full transformer) | 24 attention | 2.9235 ± 0.0039 |
| `H₀` (full Hyena) | 24 Hyena | 2.8282 ± 0.0279 |
| `H₁` (1 MHA) | 23 H + 1 A | 2.8308 ± 0.0108 |
| `H₂` (2 MHA, **best**) | 22 H + 2 A | **2.7729 ± 0.0006** |
| `H₃` (3 MHA) | 21 H + 3 A | 2.8214 ± 0.0313 |
| `H₄` (4 MHA) | 20 H + 4 A | 2.8312 ± 0.0088 |

Default for genomics retrofits: H₂-style pattern (sparse attention, ~1 A every 12 blocks for a 24-block model; for 12 blocks, place one A near the middle and one near the output).

## PDE fields (The Well, 2D or 3D)

Same as vision, with:

- `fft_padding = "circular"` — physics is on a torus or with reflecting BCs
- Patch size matters: smaller patches (p=2, p=4) widen HyenaND's advantage; defaults from `examples/well/v2/`
- Mask: per-axis Gaussian, isotropic init

## When to use registers + FiLM (input-dependent kernels)

Always for vision and PDE (§3.2.2). Default `num_registers = 4` for ViT-5-Small; scale with hidden_dim. The `KernelFiLMGenerator` + `RegisterPooling` combo from `_base_config.py` lines 276–337 is the canonical recipe.

For genomics, FiLM is optional — the Evo2 striped configs in the paper do not use it.

## Things that look like knobs but aren't really

- `grid_type`: use `"single"` for non-registered inputs, `"double"` for CLS-row + registers (see `_base_config.py` line 302)
- `L_cache`: set to the maximum sequence length you'll see + 1 (CLS row); the kernel caches at this length
- `parametrization` for Gaussian mask: `"direct"` for vision, `"exp_decay"` for 1D AR

## Reference configs to copy from

| Use case | File |
|----------|------|
| Smallest end-to-end Hyena example | `examples/mnist_classification/ccnn_4_160_hyena_rope_qknorm.py` |
| ImageNet ViT-5 with FiLM + registers | `examples/vit5_imagenet/v5_patch/_base_config.py` |
| ImageNet hybrid (pattern-driven) | `examples/vit5_imagenet/vit5_hybrid/_base_config.py` |
| Diffusion (HF diffusers retrofit) | `examples/imagenet_diffusion/ccnn_12_768_hyena_qknorm.py` |
| PDE (The Well) | `examples/well/v2/*.py` |
| CCNN-style (non-ViT) classification | `examples/imagenet_classification/ccnn_7_512_hyena_circular.py` |
