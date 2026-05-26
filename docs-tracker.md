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

`nvsubquadratic/` and `experiments/` inline docstrings, plus the
`docs/` narrative pages (Getting Started, Architecture, Examples,
Benchmarks) that wrap them.  Out of scope on this branch: rewriting the
top-level README or `examples/overview_tracker.md` themselves — the
docs narrative pages link to those sources rather than duplicating them.

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

| File                               | Status | Notes                                                                                             |
| ---------------------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| `kernels_nd.py`                    | \[x\]  | Learned kernel parametrisation — RFF + SIREN, FiLM-conditioned variants                           |
| `hyena_nd.py`                      | \[x\]  | Hyena operator (ND) — two-gate sandwich, AllToAll CP, BC-aware convolution                        |
| `ckconv_nd.py`                     | \[x\]  | CKConv (ND) — implicit kernel `k_θ(p) = MLP_θ(pos_enc(p))`, FFT domain, BC modes                  |
| `ckconv_multihead_nd.py`           | \[x\]  | Multi-head CKConv — H heads, dense d×d kernel per head, low-rank U·V factorisation                |
| `mamba_nd.py`                      | \[x\]  | Mamba SSM (ND) — selective SSM, ZOH discretisation, raster-scan ND, bidirectional mode            |
| `attention.py`                     | \[x\]  | Scaled dot-product attention — multi-head, RoPE, ND spatial, O(L²) FLOP formula                   |
| `vit5_attention.py`                | \[x\]  | ViT5 attention — register-aware 2D RoPE, QK-norm, CUDA-graph-safe buffers                         |
| `vit5_hyena_adapter.py`            | \[x\]  | Hyena adapter for ViT5 — drop-in for vit5_attention, register-token + hierarchy support           |
| `sequence_mixer.py`                | \[x\]  | Operator-agnostic dispatch layer (Hyena / Attention / CKConv / Mamba)                             |
| `condition_mixer.py`               | \[x\]  | Cross-attention conditioning mixer — both global (B,C) and spatial (B,\*,C) signals               |
| `residual_block.py`                | \[x\]  | Residual block — pre-norm + mixer + MLP, optional FiLM/AdaLN-Zero conditioning                    |
| `vit5_residual_block.py`           | \[x\]  | ViT5 residual block — LayerScale, register-token conditioning, no condition-mixer branch          |
| `patchify.py`                      | \[x\]  | Patch embedding — strided conv, 1D/2D/3D, channels-last layout                                    |
| `position_encoding.py`             | \[x\]  | Axis-factorised learned PE — ND broadcast-expand, float32 output caveat                           |
| `masks_nd.py`                      | \[x\]  | Exponential + Gaussian receptive-field windows; mask convention 1=included, 0=excluded            |
| `mlp.py`                           | \[x\]  | Two-layer MLP — GELU/SwiGLU/GLU variants, expansion-ratio math, QuACK backend noted               |
| `film.py`                          | \[x\]  | FiLM conditioning — γ(c)⊙x + β(c), SIREN-based kernel generator                                   |
| `grn.py`                           | \[x\]  | GRN — per-channel L2 norm, inter-channel competition, ConvNeXt V2 reference                       |
| `layer_scale.py`                   | \[x\]  | LayerScale — per-channel λ⊙F(x), init_values guidance, \_no_weight_decay tag                      |
| `rms_norm.py`                      | \[x\]  | RMSNorm + PerHeadRMSNorm — QuACK/PyTorch backends, math formula, QK-norm usage                    |
| `rms_norm_channel_first.py`        | \[x\]  | Channel-first RMSNorm — normalises dim=1, `channels_first` sentinel, Hyena usage                  |
| `drop_path.py`                     | \[x\]  | Stochastic depth — functional + Module, inverted-dropout scaling, causal vs training              |
| `causal_conv1d.py`                 | \[x\]  | CausalConv1D — left-only pad formula, symmetric mode, Mamba usage context                         |
| `subq_ops_causal_conv1d.py`        | \[x\]  | New (1D PR): `nn.Conv1d`-compatible depthwise wrapper around `subq_ops.causal_conv1d`             |
| `schedulers.py`                    | \[x\]  | ResumableSequentialLR — PyTorch ≤2.10 bug fix, load_state_dict LR propagation                     |
| `distributed_depthwise_conv_nd.py` | \[x\]  | CP-aware 1D/2D/3D depthwise conv — group weight sharing, channel slicing, causal padding          |
| `patch_merging.py`                 | \[ \]  | Pending (#122 — `feat/patch-merging`): Swin-style 2×2 patch merging with register-row passthrough |

### `nvsubquadratic/networks/` — Full architectures

| File                                  | Status | Notes                                                                                                                                     |
| ------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `general_purpose_resnet.py`           | \[x\]  | ResidualNetwork — LazyConfig blocks, conditioning, readout crop, gradient checkpointing                                                   |
| `classification_resnet.py`            | \[x\]  | ClassificationResNet — GAP readout, resolution-agnostic, inherits ResidualNetwork                                                         |
| `vit5_classification.py`              | \[x\]  | ViT5 classification — token layout, hybrid blocks, CLS/GAP/register_concat readouts, FLOP count                                           |
| `vit5_hierarchical_classification.py` | \[ \]  | Pending (#122 — `feat/patch-merging`): Swin-style 4-stage hierarchical ViT-5 classifier with GAP readout and optional register-row layout |
| `huggingface_diffusers.py`            | \[x\]  | HF diffusers adapters — DiT and UVit wrappers, BHL↔BCHW translation, shared timestep state model                                          |
| `jit.py` / `jit_utils.py`             | \[x\]  | JiT diffusion backbone port (LTH14/JiT) — patch embed, attention/SwiGLU blocks, RoPE helpers, RMSNorm                                     |
| `baselines/unet_convnext.py`          | \[x\]  | UNet-ConvNeXt baseline ported from The Well — preserves upstream `skips[0]` bug for reproducibility                                       |
| `baselines/unet_convnext_v2.py`       | \[x\]  | Fixed-skip variant of UNet-ConvNeXt — consumes the missing finest-resolution skip                                                         |

### `nvsubquadratic/parallel/` — Distributed primitives

| File                | Status | Notes                                                                                                                                                                 |
| ------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `a2a_comms.py`      | \[x\]  | AllToAllSingle — CP sequence↔channel redistribution, zigzag splitting, autograd backward                                                                              |
| `utils.py`          | \[x\]  | CP utilities — `init_parallel_state`, zigzag split/gather across ranks, rank-0 logging routing                                                                        |
| `test_a2a_comms.py` | \[x\]  | Unit tests for the zigzag-splitting helpers.  Lives next to `a2a_comms.py` so the test imports stay one hop from the implementation; mirror in `tests/` not required. |

### `nvsubquadratic/` — Top-level

| File             | Status | Notes                                                                                 |
| ---------------- | ------ | ------------------------------------------------------------------------------------- |
| `lazy_config.py` | \[x\]  | LazyConfig + instantiate — deferred instantiation, nested configs, arithmetic strings |
| `metrics/`       | \[x\]  | cleanfid.py — CleanFID wrapper, FID formula, usage context                            |
| `utils/`         | \[x\]  | init.py (weight init factories), qk_norm.py (QK-norm, L2Norm module), quack_utils.py  |
| `testing/`       | \[x\]  | utils.py — compute_relative_error, TTrace reference, already had good docstrings      |

### `experiments/` — Training infrastructure

| File                                           | Status | Notes                                                                                                      |
| ---------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------------- |
| `run.py`                                       | \[x\]  | Entry point — CLI parse, W&B init, network + wrapper instantiation, Trainer.fit                            |
| `trainer.py`                                   | \[x\]  | construct_trainer — checkpoint callbacks, precision, wall-time, W&B upload                                 |
| `default_cfg.py`                               | \[x\]  | Typed dataclass configs — Train/Trainer/Scheduler/Wandb/Optimizer/AutoResume                               |
| `lightning_wrappers/base_lightning_wrapper.py` | \[x\]  | LightningWrapperBase — param groups, scheduler, checkpoint resume, profiling                               |
| `lightning_wrappers/classification_wrapper.py` | \[x\]  | ClassificationWrapper — cross_entropy / soft_target_ce / bce loss modes                                    |
| `lightning_wrappers/regression_wrapper.py`     | \[x\]  | RegressionWrapper — MAE/MSE loss, base for WELLRegressionWrapper                                           |
| `lightning_wrappers/diffusion_wrapper.py`      | \[x\]  | DiffusionWrapper — JiT continuous-time diffusion, ODE sampler                                              |
| `lightning_wrappers/autoregressive_wrapper.py` | \[x\]  | Already had good module + class docstrings; left as-is                                                     |
| `lightning_wrappers/arc_wrapper.py`            | \[ \]  | (untracked new file — out of scope until merged)                                                           |
| `lightning_wrappers/well_lightning_wrapper.py` | \[x\]  | Already had good class docstring; left as-is                                                               |
| `datamodules/arc.py`                           | \[ \]  | (untracked new file — out of scope until merged)                                                           |
| `datamodules/mnist.py`                         | \[x\]  | MNIST datamodule — channels-last reshape, train/val split                                                  |
| `datamodules/emnist.py`                        | \[x\]  | EMNIST datamodule — all five splits (digits, letters, balanced, bymerge, byclass)                          |
| `datamodules/tinyimagenet.py`                  | \[x\]  | TinyImageNet HF-backed datamodule — RandAugment, Mixup/CutMix, token access                                |
| `datamodules/ucf101.py`                        | \[x\]  | UCF101 — video/sequence modes, frames_per_clip, deterministic workers                                      |
| `datamodules/dali_imagenet_fused.py`           | \[x\]  | DALI ImageNet — fused GPU augmentation, MixupConfig/AugmentConfig, repeated aug                            |
| `datamodules/spatial_recall_dataset.py`        | \[x\]  | Already had comprehensive module + class docstrings; left as-is                                            |
| `datamodules/pde/well.py`                      | \[x\]  | WELL benchmark datamodule — persistent HDF5 handles, NVMe staging, RAM preload, val/test normalisation fix |
| `datamodules/utils/dali_rand_augment.py`       | \[x\]  | DALI RandAugment matching timm — per-op magnitude/probability noise, increasing-transforms suite           |
| `utils/cli.py`                                 | \[x\]  | CLI helpers — load_config_from_file, apply_config_overrides, run name generation                           |
| `utils/checkpointing.py`                       | \[x\]  | Already had good per-function docstrings; left as-is                                                       |
| `callbacks/`                                   | \[x\]  | walltime_checkpointer (added module docstring); all others already had good docs                           |
