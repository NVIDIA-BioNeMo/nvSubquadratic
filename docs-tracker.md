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
`docs/` narrative pages (Getting Started, Architecture, Repository
Overview, Examples, Benchmarks, Reports) that wrap them.  Out of
scope on this branch: rewriting the top-level README or
`examples/overview_tracker.md` themselves — the docs narrative pages
link to those sources rather than duplicating them.
`benchmarks/`, `scripts/visualization/`, and `reports/` are in scope at
a lighter bar — module-level docstrings (4-question format: what /
hardware / how to invoke / where the output goes) and one README per
subdirectory, no Sphinx API reference entry; per-topic `REPORT.md`
files live under `reports/`.  The top-level `visualizations/` directory
was retired into `reports/` by the repo-organization PR.

______________________________________________________________________

## Progress

Work bottom-up: primitive ops → modules → networks → experiments.

### `nvsubquadratic/ops/` — FFT convolution primitives

| File                      | Status | Notes                                                                                               |
| ------------------------- | ------ | --------------------------------------------------------------------------------------------------- |
| `README.md`               | \[x\]  | Folder overview, decision tree, math primer (new file)                                              |
| `fftconv.py`              | \[x\]  | Module + key per-fn docstrings rewritten with math                                                  |
| `circular_fftconv.py`     | \[x\]  | Already strong; left as-is                                                                          |
| `fftconv_chunked.py`      | \[x\]  | Already strong; left as-is                                                                          |
| `fftconv_custom.py`       | \[x\]  | Module docstring expanded with motivation; 1D causal wrappers added in 1D PR                        |
| `causal_conv1d_custom.py` | \[x\]  | New (1D PR): thin wrappers for direct `causal_conv1d` + fused `b2b_causal_conv1d`                   |
| `mixed_fftconv.py`        | \[x\]  | New (#120): per-axis mixed boundary-condition FFT conv; see `docs/ops/mixed_boundary_conditions.md` |

### `nvsubquadratic/modules/` — Building blocks

| File                               | Status | Notes                                                                                             |
| ---------------------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| `kernels_nd.py`                    | \[x\]  | Learned kernel parametrisation — RFF + SIREN, FiLM-conditioned variants                           |
| `hyena_nd.py`                      | \[x\]  | Hyena operator (ND) — two-gate sandwich, AllToAll CP, BC-aware convolution                        |
| `ckconv_nd.py`                     | \[x\]  | CKConv (ND) — implicit kernel `k_θ(p) = MLP_θ(pos_enc(p))`, FFT domain, BC modes                  |
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
| `baselines/unet_convnext.py`          | \[x\]  | UNet-ConvNeXt baseline ported from The Well — preserves upstream `skips[0]` bug for reproducibility                                       |
| `baselines/unet_convnext_v2.py`       | \[x\]  | Fixed-skip variant of UNet-ConvNeXt — consumes the missing finest-resolution skip                                                         |

### `nvsubquadratic/parallel/` — Distributed primitives

| File           | Status | Notes                                                                                          |
| -------------- | ------ | ---------------------------------------------------------------------------------------------- |
| `a2a_comms.py` | \[x\]  | AllToAllSingle — CP sequence↔channel redistribution, zigzag splitting, autograd backward       |
| `utils.py`     | \[x\]  | CP utilities — `init_parallel_state`, zigzag split/gather across ranks, rank-0 logging routing |

(Unit tests for the zigzag helpers now live at
[tests/parallel/test_a2a_comms.py](tests/parallel/test_a2a_comms.py);
the `tests/` tree is out of scope for this tracker.)

### `nvsubquadratic/` — Top-level

| File             | Status | Notes                                                                                 |
| ---------------- | ------ | ------------------------------------------------------------------------------------- |
| `lazy_config.py` | \[x\]  | LazyConfig + instantiate — deferred instantiation, nested configs, arithmetic strings |
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
| `lightning_wrappers/autoregressive_wrapper.py` | \[x\]  | Already had good module + class docstrings; left as-is                                                     |
| `lightning_wrappers/arc_wrapper.py`            | \[ \]  | (untracked new file — out of scope until merged)                                                           |
| `lightning_wrappers/well_lightning_wrapper.py` | \[x\]  | Already had good class docstring; left as-is                                                               |
| `datamodules/arc.py`                           | \[ \]  | (untracked new file — out of scope until merged)                                                           |
| `datamodules/mnist.py`                         | \[x\]  | MNIST datamodule — channels-last reshape, train/val split                                                  |
| `datamodules/emnist.py`                        | \[x\]  | EMNIST datamodule — all five splits (digits, letters, balanced, bymerge, byclass)                          |
| `datamodules/tinyimagenet.py`                  | \[x\]  | TinyImageNet HF-backed datamodule — RandAugment, Mixup/CutMix, token access                                |
| `datamodules/dali_imagenet_fused.py`           | \[x\]  | DALI ImageNet — fused GPU augmentation, MixupConfig/AugmentConfig, repeated aug                            |
| `datamodules/spatial_recall_dataset.py`        | \[x\]  | Already had comprehensive module + class docstrings; left as-is                                            |
| `datamodules/pde/well.py`                      | \[x\]  | WELL benchmark datamodule — persistent HDF5 handles, NVMe staging, RAM preload, val/test normalisation fix |
| `datamodules/utils/dali_rand_augment.py`       | \[x\]  | DALI RandAugment matching timm — per-op magnitude/probability noise, increasing-transforms suite           |
| `utils/cli.py`                                 | \[x\]  | CLI helpers — load_config_from_file, apply_config_overrides, run name generation                           |
| `utils/checkpointing.py`                       | \[x\]  | Already had good per-function docstrings; left as-is                                                       |
| `callbacks/`                                   | \[x\]  | walltime_checkpointer (added module docstring); all others already had good docs                           |

### `benchmarks/` — Throughput and profiling scripts

Lighter bar: module-level docstring only (4-question format) plus one
README per subdirectory.  No Sphinx API reference entry.

| File                                             | Status | Notes                                                                                                      |
| ------------------------------------------------ | ------ | ---------------------------------------------------------------------------------------------------------- |
| `README.md`                                      | \[x\]  | ViT-5-Small headline throughput tables; included into the Sphinx `Benchmarks` page                         |
| `benchmark_imagenet_diffusion_gpu.py`            | \[x\]  | GPU memory/time benchmark for ImageNet diffusion at batch=1                                                |
| `benchmark_patch_size_2d.py`                     | \[x\]  | Moved from `scripts/`; forward-time vs patch-size sweep for 2D residual-net mixers (attn / Hyena / Mamba2) |
| `compare_flops.py`                               | \[x\]  | FLOP comparison across ViT-5-Small variants (Attention / Hyena / Hyena+FiLM); already had a good docstring |
| `ops/README.md`                                  | \[x\]  | Op-level benchmark overview (fftconv2d / mlp / subq_ops fftconv)                                           |
| `ops/bench_fftconv2d.py`                         | \[x\]  | Already had a full docstring                                                                               |
| `ops/bench_mlp.py`                               | \[x\]  | torch vs QuACK MLP correctness + timing                                                                    |
| `ops/bench_subquadratic_fftconv.py`              | \[x\]  | Sanity gate for the CUDA fft_causal_conv1d kernel                                                          |
| `vit5_imagenet/README.md`                        | \[x\]  | Per-script overview + pointer to the historical profiling report                                           |
| `vit5_imagenet/bench_vit5_baseline.py`           | \[x\]  | Baseline (eager, torchvision dataloader) throughput                                                        |
| `vit5_imagenet/bench_vit5_compile.py`            | \[x\]  | `torch.compile` configuration sweep                                                                        |
| `vit5_imagenet/bench_vit5_hyena.py`              | \[x\]  | Attention vs Hyena vs Hyena-FiLM throughput; already had a full docstring                                  |
| `vit5_imagenet/bench_vit5_optimized.py`          | \[x\]  | Production-optimised pipeline (BF16 + compile + DALI fused)                                                |
| `vit5_imagenet/bench_vit5_profile.py`            | \[x\]  | Per-phase forward+backward profiling                                                                       |
| `vit5_imagenet/benchmark_imagenet_throughput.py` | \[x\]  | Moved from `scripts/`; inference-only ImageNet-1k throughput across four ViT-5 architectures               |
| `vit5_imagenet/verify_dali_fused.py`             | \[x\]  | DALI fused output sanity checks + visual comparison                                                        |
| `vit5_imagenet/validate_checkpoint.py`           | \[x\]  | Loads a W&B "best" checkpoint and runs val/test on ImageNet-1k                                             |
| `vit5_imagenet/bench_*.sh`                       | \[x\]  | One-line SLURM-driver header explaining what each invokes (co-located with the `.py` benchmarks)           |
| `well/README.md`                                 | \[x\]  | Per-script overview for the WELL benchmark suite                                                           |
| `well/bench_ab_comparison.py`                    | \[x\]  | A/B baseline-vs-optimised dataloader+training; already had a full docstring                                |
| `well/bench_dataloader.py`                       | \[x\]  | Isolated WELL dataloader throughput                                                                        |
| `well/bench_training_step.py`                    | \[x\]  | End-to-end WELL training-step throughput                                                                   |
| `well/parse_bench.py`                            | \[x\]  | Parses the SLURM sweep driver's stdout into a summary table                                                |
| `well/profile_timing.py`                         | \[x\]  | Phase profiling on the Gray-Scott Hyena WELL config                                                        |
| `well/profile_training_loop.py`                  | \[x\]  | Diagnoses the gap between pure compute and PL-reported step time                                           |
| `well/profile_batch_size.py`                     | \[x\]  | Moved from `scripts/`; sweeps batch sizes against an 80 GB budget for each supernova_explosion_64 model    |
| `well/verify_vrmse.py`                           | \[x\]  | Cross-checks VRMSE implementation against manual computation                                               |

### `scripts/visualization/` — Visualization tools

| File                                 | Status | Notes                                                                           |
| ------------------------------------ | ------ | ------------------------------------------------------------------------------- |
| `README.md`                          | \[x\]  | Streamlit vs Gradio kernel-viewer divergence + when to reach for each           |
| `visualize_kernels.py`               | \[x\]  | Streamlit kernel/mask viewer (consumes `.json`); usage paths corrected          |
| `visualize_kernels_app.py`           | \[x\]  | Gradio kernel/mask viewer (consumes `.npz`, headless-friendly); paths corrected |
| `visualize_repeated_aug.py`          | \[x\]  | DALI repeated-augmentation sanity check; already had a working docstring        |
| `visualize_patch_size_throughput.py` | \[x\]  | Moved from `scripts/`; patch-size sweep throughput plot                         |

### `reports/` — Topic-folder technical reports

Each topic owns a `REPORT.md` plus the scripts and figures it cites.
Index of topics lives in `reports/README.md`.

| Topic                                 | Status | Notes                                                                                      |
| ------------------------------------- | ------ | ------------------------------------------------------------------------------------------ |
| `ckconv_block_diagonal_kernel/`       | \[x\]  | Block-diagonal multi-ω₀ SIREN + block-aligned Gaussian mask (existing topic)               |
| `siren_omega0_dimensional_scaling/`   | \[x\]  | SIREN ω₀ dimensional scaling rule (existing topic)                                         |
| `spatial_recall/`                     | \[x\]  | Qualitative target-vs-prediction snapshots for the 1D/2D/3D EMNIST spatial-recall suite    |
| `vit5_imagenet_dataloader_profiling/` | \[x\]  | Feb-2026 ViT-5 ImageNet CPU-decode bottleneck investigation; motivated the DALI-fused move |
