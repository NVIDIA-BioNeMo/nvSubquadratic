# Documentation Tracker

## Goals

- Audience: external collaborators reading alongside the research paper
- Format: inline docstrings only — tutorial-style explanations and math context live in the docstrings, not separate pages
- Convention: PRs touching a module must update its docstring (to be enforced via CONVENTIONS.md)

## Conventions (to be formalised in CONVENTIONS.md)

- All public classes and functions get a Google-style docstring
- Math/motivation context belongs in the class-level docstring, not method-level
- Take notation and intuition from the paper where applicable
- Keep docstrings honest — if a parameter name is misleading, rename it rather than papering over it in the docstring

## Scope

`nvsubquadratic/` and `experiments/` only. README and experiment-overview.md are out of scope for this branch.

______________________________________________________________________

## Progress

Work bottom-up: primitive ops → modules → networks → experiments.

### `nvsubquadratic/ops/` — FFT convolution primitives

| File                       | Status | Notes                                                                                   |
| -------------------------- | ------ | --------------------------------------------------------------------------------------- |
| `README.md`                | \[x\]  | Folder overview, decision tree, math primer (new file)                                  |
| `fftconv.py`               | \[x\]  | Module + key per-fn docstrings rewritten with math                                      |
| `circular_fftconv.py`      | \[x\]  | Already strong; left as-is                                                              |
| `circular_fftconv_fp16.py` | \[x\]  | Already strong; relies on FP16_FFTCONV_DERIVATION.md                                    |
| `fftconv_fp16.py`          | \[x\]  | Already adequate; left as-is                                                            |
| `fftconv_multihead.py`     | \[x\]  | Module docstring expanded with multi-head/low-rank math                                 |
| `fftconv_chunked.py`       | \[x\]  | Already strong; left as-is                                                              |
| `fftconv_custom.py`        | \[x\]  | Module docstring expanded with motivation; 1D causal wrappers added in 1D PR            |
| `causal_conv1d_custom.py`  | \[x\]  | New (1D PR): thin wrappers for direct `causal_conv1d` + fused `b2b_causal_conv1d`       |
| `mixed_fftconv.py`         | \[x\]  | New (#120): per-axis mixed boundary-condition FFT conv; see `docs/ops/MIXED_BC_PLAN.md` |

### `nvsubquadratic/modules/` — Building blocks

| File                               | Status | Notes                                                                                       |
| ---------------------------------- | ------ | ------------------------------------------------------------------------------------------- |
| `kernels_nd.py`                    | \[x\]  | Learned kernel parametrisation — RFF + SIREN, FiLM-conditioned variants                     |
| `hyena_nd.py`                      | \[x\]  | Hyena operator (ND) — two-gate sandwich, AllToAll CP, BC-aware convolution                  |
| `ckconv_nd.py`                     | \[x\]  | CKConv (ND) — implicit kernel `k_θ(p) = MLP_θ(pos_enc(p))`, FFT domain, BC modes            |
| `ckconv_multihead_nd.py`           | \[x\]  | Multi-head CKConv — H heads, dense d×d kernel per head, low-rank U·V factorisation          |
| `mamba_nd.py`                      | \[x\]  | Mamba SSM (ND) — selective SSM, ZOH discretisation, raster-scan ND, bidirectional mode      |
| `attention.py`                     | \[x\]  | Scaled dot-product attention — multi-head, RoPE, ND spatial, O(L²) FLOP formula             |
| `vit5_attention.py`                | \[x\]  | ViT5 attention — register-aware 2D RoPE, QK-norm, CUDA-graph-safe buffers                   |
| `vit5_hyena_adapter.py`            | \[x\]  | Hyena adapter for ViT5 — drop-in for vit5_attention, register-token + hierarchy support     |
| `sequence_mixer.py`                | \[x\]  | Operator-agnostic dispatch layer (Hyena / Attention / CKConv / Mamba)                       |
| `condition_mixer.py`               | \[x\]  | Cross-attention conditioning mixer — both global (B,C) and spatial (B,\*,C) signals         |
| `residual_block.py`                | \[x\]  | Residual block — pre-norm + mixer + MLP, optional FiLM/AdaLN-Zero conditioning              |
| `vit5_residual_block.py`           | \[x\]  | ViT5 residual block — LayerScale, register-token conditioning, no condition-mixer branch    |
| `patchify.py`                      | \[x\]  | Patch embedding — strided conv, 1D/2D/3D, channels-last layout                              |
| `position_encoding.py`             | \[x\]  | Axis-factorised learned PE — ND broadcast-expand, float32 output caveat                     |
| `masks_nd.py`                      | \[x\]  | Exponential + Gaussian receptive-field windows; mask convention 1=included, 0=excluded      |
| `mlp.py`                           | \[x\]  | Two-layer MLP — GELU/SwiGLU/GLU variants, expansion-ratio math, QuACK backend noted         |
| `film.py`                          | \[x\]  | FiLM conditioning — γ(c)⊙x + β(c), SIREN-based kernel generator                             |
| `grn.py`                           | \[x\]  | GRN — per-channel L2 norm, inter-channel competition, ConvNeXt V2 reference                 |
| `layer_scale.py`                   | \[x\]  | LayerScale — per-channel λ⊙F(x), init_values guidance, \_no_weight_decay tag                |
| `rms_norm.py`                      | \[ \]  | RMS normalisation                                                                           |
| `rms_norm_channel_first.py`        | \[ \]  | Channel-first RMS norm                                                                      |
| `drop_path.py`                     | \[ \]  | Stochastic depth                                                                            |
| `causal_conv1d.py`                 | \[ \]  | Causal 1D conv                                                                              |
| `subq_ops_causal_conv1d.py`        | \[x\]  | New (1D PR): `nn.Conv1d`-compatible depthwise wrapper around `subq_ops.causal_conv1d`       |
| `schedulers.py`                    | \[ \]  | LR schedulers                                                                               |
| `distributed_depthwise_conv_nd.py` | \[ \]  | Distributed depthwise conv                                                                  |
| `patch_merging.py`                 | \[ \]  | Pending (feat/patch-merging PR): Swin-style 2×2 patch merging with register-row passthrough |

### `nvsubquadratic/networks/` — Full architectures

| File                                  | Status | Notes                                                                                                                               |
| ------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `general_purpose_resnet.py`           | \[ \]  |                                                                                                                                     |
| `classification_resnet.py`            | \[ \]  |                                                                                                                                     |
| `vit5_classification.py`              | \[ \]  | ViT5 classification head                                                                                                            |
| `vit5_hierarchical_classification.py` | \[ \]  | Pending (feat/patch-merging PR): Swin-style 4-stage hierarchical ViT-5 classifier with GAP readout and optional register-row layout |
| `huggingface_diffusers.py`            | \[ \]  | HF diffusers integration                                                                                                            |
| `jit.py` / `jit_utils.py`             | \[ \]  | TorchScript utilities                                                                                                               |

### `nvsubquadratic/parallel/` — Distributed primitives

| File           | Status | Notes            |
| -------------- | ------ | ---------------- |
| `a2a_comms.py` | \[ \]  | All-to-all comms |

### `nvsubquadratic/` — Top-level

| File             | Status | Notes              |
| ---------------- | ------ | ------------------ |
| `lazy_config.py` | \[ \]  | Lazy config system |
| `metrics/`       | \[ \]  | Metric utilities   |
| `utils/`         | \[ \]  | General utilities  |
| `testing/`       | \[ \]  | Testing utilities  |

### `experiments/` — Training infrastructure

| File                                           | Status | Notes          |
| ---------------------------------------------- | ------ | -------------- |
| `run.py`                                       | \[ \]  | Entry point    |
| `trainer.py`                                   | \[ \]  | Training loop  |
| `default_cfg.py`                               | \[ \]  | Default config |
| `lightning_wrappers/base_lightning_wrapper.py` | \[ \]  | Base wrapper   |
| `lightning_wrappers/classification_wrapper.py` | \[ \]  |                |
| `lightning_wrappers/regression_wrapper.py`     | \[ \]  |                |
| `lightning_wrappers/diffusion_wrapper.py`      | \[ \]  |                |
| `lightning_wrappers/autoregressive_wrapper.py` | \[ \]  |                |
| `lightning_wrappers/arc_wrapper.py`            | \[ \]  |                |
| `lightning_wrappers/well_lightning_wrapper.py` | \[ \]  |                |
| `datamodules/arc.py`                           | \[ \]  |                |
| `datamodules/mnist.py`                         | \[ \]  |                |
| `datamodules/tinyimagenet.py`                  | \[ \]  |                |
| `datamodules/ucf101.py`                        | \[ \]  |                |
| `datamodules/dali_imagenet_fused.py`           | \[ \]  |                |
| `datamodules/spatial_recall_dataset.py`        | \[ \]  |                |
| `utils/cli.py`                                 | \[ \]  |                |
| `utils/checkpointing.py`                       | \[ \]  |                |
| `callbacks/`                                   | \[ \]  |                |
