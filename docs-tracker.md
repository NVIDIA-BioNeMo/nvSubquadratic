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

| File                       | Status | Notes                    |
| -------------------------- | ------ | ------------------------ |
| `fftconv.py`               | \[ \]  | Core ND FFT conv         |
| `circular_fftconv.py`      | \[ \]  | Circular variant         |
| `circular_fftconv_fp16.py` | \[ \]  | FP16 circular            |
| `fftconv_fp16.py`          | \[ \]  | FP16 base                |
| `fftconv_multihead.py`     | \[ \]  | Multi-head variant       |
| `fftconv_chunked.py`       | \[ \]  | Memory-efficient chunked |
| `fftconv_custom.py`        | \[ \]  | Custom kernel entry      |

### `nvsubquadratic/modules/` — Building blocks

| File                               | Status | Notes                                     |
| ---------------------------------- | ------ | ----------------------------------------- |
| `kernels_nd.py`                    | \[ \]  | Learned kernel parametrisation            |
| `hyena_nd.py`                      | \[ \]  | Hyena operator (ND) — key paper component |
| `ckconv_nd.py`                     | \[ \]  | CKConv (ND)                               |
| `ckconv_multihead_nd.py`           | \[ \]  | Multi-head CKConv                         |
| `mamba_nd.py`                      | \[ \]  | Mamba SSM (ND)                            |
| `attention.py`                     | \[ \]  | Standard attention                        |
| `vit5_attention.py`                | \[ \]  | ViT5 attention variant                    |
| `vit5_hyena_adapter.py`            | \[ \]  | Hyena adapter for ViT5                    |
| `sequence_mixer.py`                | \[ \]  | Mixer abstraction                         |
| `condition_mixer.py`               | \[ \]  | Conditioning mixer                        |
| `residual_block.py`                | \[ \]  | Residual block                            |
| `vit5_residual_block.py`           | \[ \]  | ViT5 residual block                       |
| `patchify.py`                      | \[ \]  | Patch embedding                           |
| `position_encoding.py`             | \[ \]  | Position encodings                        |
| `masks_nd.py`                      | \[ \]  | ND masking utils                          |
| `mlp.py`                           | \[ \]  | MLP block                                 |
| `film.py`                          | \[ \]  | FiLM conditioning                         |
| `grn.py`                           | \[ \]  | GRN normalisation                         |
| `layer_scale.py`                   | \[ \]  | LayerScale                                |
| `rms_norm.py`                      | \[ \]  | RMS normalisation                         |
| `rms_norm_channel_first.py`        | \[ \]  | Channel-first RMS norm                    |
| `drop_path.py`                     | \[ \]  | Stochastic depth                          |
| `causal_conv1d.py`                 | \[ \]  | Causal 1D conv                            |
| `schedulers.py`                    | \[ \]  | LR schedulers                             |
| `distributed_depthwise_conv_nd.py` | \[ \]  | Distributed depthwise conv                |

### `nvsubquadratic/networks/` — Full architectures

| File                        | Status | Notes                    |
| --------------------------- | ------ | ------------------------ |
| `general_purpose_resnet.py` | \[ \]  |                          |
| `classification_resnet.py`  | \[ \]  |                          |
| `arc_resnet.py`             | \[ \]  | ARC task network         |
| `arc_embedding.py`          | \[ \]  | ARC embedding            |
| `vit5_classification.py`    | \[ \]  | ViT5 classification head |
| `huggingface_diffusers.py`  | \[ \]  | HF diffusers integration |
| `jit.py` / `jit_utils.py`   | \[ \]  | TorchScript utilities    |

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
