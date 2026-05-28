# Repository overview

A map of what lives where at the repo root.  The library code
(`nvsubquadratic/`) and the training driver (`experiments/`) sit
alongside the per-task configs (`examples/`), the perf measurement
tree (`benchmarks/`), and the supporting infrastructure (`scripts/`,
`tests/`, `reports/`, `docs/`).

## Layout

```text
nvSubquadratic-private/
├── nvsubquadratic/          library code — see "library tree" below
├── experiments/             training framework (PyTorch Lightning)
│   ├── run.py                 CLI entry point
│   ├── trainer.py             construct_trainer (checkpoints, precision, W&B)
│   ├── default_cfg.py         typed ExperimentConfig dataclasses
│   ├── lightning_wrappers/    task-specific wrappers (classification, diffusion, regression, …)
│   ├── datamodules/           LightningDataModule subclasses (ImageNet, MNIST, WELL, …)
│   ├── callbacks/             FiLM monitor, image-grid viz, EMA, walltime checkpointer, …
│   └── utils/                 cli + checkpointing helpers
├── examples/                LazyConfig recipes that feed experiments.run
│   ├── mnist_classification/
│   ├── imagenet_classification/
│   ├── imagenet_diffusion/
│   ├── vit5_imagenet/         ViT-5 baseline suite (v1–v5)
│   ├── spatial_recall_{1,2,3}d/ and spatial_recall_v2/
│   ├── ucf101_classification/
│   ├── well/                  The Well PDE benchmark suite
│   └── overview_tracker.md    active experimental roadmap
├── benchmarks/              performance measurement (the canonical home)
│   ├── README.md              ViT-5-Small headline throughput tables
│   ├── compare_flops.py       FLOP comparison across ViT-5 variants
│   ├── benchmark_imagenet_diffusion_gpu.py
│   ├── benchmark_patch_size_2d.py
│   ├── ops/                   op-level benchmarks (fftconv2d / MLP / subq-ops)
│   ├── vit5_imagenet/         ViT-5 throughput, profile, verify, validate
│   └── well/                  WELL dataloader / training-step / VRMSE
├── reports/                 frozen-in-time investigations with regen scripts
│   ├── ckconv_block_diagonal_kernel/
│   ├── siren_omega0_dimensional_scaling/
│   ├── spatial_recall/
│   └── vit5_imagenet_dataloader_profiling/
├── scripts/                 utilities (data prep, sanity, SLURM, viz)
│   ├── slurm/                 SLURM submit scripts (portable wrapper + per-experiment)
│   ├── data/                  data prep (ImageNet folder extraction, FID stats, …)
│   ├── evaluation/            eval helpers (FID, CMMD)
│   ├── visualization/         kernel viewers + throughput plot
│   └── check_gpu_availability.py, license_check.py, …
├── tests/                   correctness tests
│   ├── conftest.py            shared fixtures
│   ├── ops/, modules/, networks/, parallel/   per-package test trees
│   └── test_*.py              top-level integration tests
├── docs/                    Sphinx documentation site (this site)
│   ├── conf.py, index.rst     site config + landing
│   ├── getting_started.md, architecture.md, repository_overview.md
│   ├── api_reference/         curated API per area
│   ├── ops/                   FFT-ops math primer + FP16 derivation
│   ├── examples/index.md, benchmarks.md, reports.md
│   └── _templates/, _static/  autosummary templates + custom CSS
├── docs-tracker.md          documentation coverage status per file
├── CONVENTIONS.md           Google-style docstring guide and PR checklist
├── README.md                top-level install / overview
├── pyproject.toml           project metadata, dependencies, ruff config
├── Dockerfile               production container
├── nvsubquadratic.def       Apptainer/Singularity recipe
└── setup_conda_env.sh       local conda bootstrap
```

## Library tree (`nvsubquadratic/`)

The library itself is organised bottom-up: function-only convolution
primitives in `ops/`, then `nn.Module`-shaped mixers and blocks in
`modules/`, then full architectures in `networks/`.

```text
nvsubquadratic/
├── lazy_config.py                deferred-instantiation system (LazyConfig, instantiate)
├── ops/                          function-only convolution primitives
│   ├── fftconv.py                fp32 reference FFT conv (1D / 2D / 3D, linear)
│   ├── fftconv_fp16.py           half-precision linear-conv variants
│   ├── circular_fftconv.py       fp32 periodic-boundary FFT conv
│   ├── circular_fftconv_fp16.py  half-precision periodic variants
│   ├── fftconv_multihead.py      multi-head + low-rank factorisations
│   ├── fftconv_chunked.py        peak-FFT-memory chunking helpers
│   ├── fftconv_custom.py         subquadratic_ops_torch CUDA wrappers
│   ├── causal_conv1d_custom.py   direct (non-FFT) CUDA causal conv1d wrappers
│   └── mixed_fftconv.py          per-axis mixed boundary-condition FFT conv
├── modules/                      nn.Module building blocks
│   ├── hyena_nd.py               Hyena ND mixer (two-gate sandwich, CP)
│   ├── mamba_nd.py               Mamba SSM (ND, selective, raster scan)
│   ├── attention.py              multi-head attention (RoPE, ND)
│   ├── vit5_attention.py         ViT-5 register-aware attention
│   ├── vit5_hyena_adapter.py     Hyena drop-in for ViT-5
│   ├── sequence_mixer.py         operator-agnostic QKV dispatch
│   ├── condition_mixer.py        cross-attention conditioning mixer
│   ├── kernels_nd.py             SIREN / RFF kernels (multi-ω₀, block-diag)
│   ├── ckconv_nd.py              CKConv ND (implicit k_θ(p))
│   ├── ckconv_multihead_nd.py    multi-head CKConv (low-rank)
│   ├── distributed_depthwise_conv_nd.py   CP-aware depthwise convs
│   ├── causal_conv1d.py          left-only-padded Conv1d wrapper
│   ├── subq_ops_causal_conv1d.py nn.Conv1d-compatible CUDA depthwise
│   ├── residual_block.py         pre-norm + mixer + MLP (+ AdaLN-Zero)
│   ├── vit5_residual_block.py    ViT-5 residual block (LayerScale, registers)
│   ├── patchify.py               strided patch embedding / unpatchify
│   ├── position_encoding.py      axis-factorised learned PE
│   ├── masks_nd.py               exponential / Gaussian / block-aligned masks
│   ├── mlp.py                    GELU / SwiGLU / GLU MLP
│   ├── film.py                   FiLM kernel generator + register pooling
│   ├── grn.py                    GlobalResponseNorm (ConvNeXt V2)
│   ├── layer_scale.py            LayerScale γ·F(x)
│   ├── drop_path.py              stochastic depth
│   ├── rms_norm.py               RMSNorm + PerHeadRMSNorm
│   ├── rms_norm_channel_first.py channel-first RMSNorm
│   └── schedulers.py             ResumableSequentialLR
├── networks/                     end-to-end architectures
│   ├── general_purpose_resnet.py ResidualNetwork (LazyConfig stack)
│   ├── classification_resnet.py  GAP-readout classification head
│   ├── vit5_classification.py    ViT-5 hybrid Hyena/attention backbone
│   ├── jit.py                    JiT diffusion backbone
│   ├── jit_utils.py              JiT helpers (RoPE, RMSNorm, sin-cos PE)
│   ├── huggingface_diffusers.py  HF DiT / UVit adapters
│   └── baselines/
│       ├── unet_convnext.py      Well UNet-ConvNeXt baseline
│       └── unet_convnext_v2.py   …with fixed finest-skip
├── parallel/                     context-parallel primitives
│   ├── a2a_comms.py              AllToAllSingle (autograd-aware)
│   └── utils.py                  init_parallel_state + zigzag split/gather
├── utils/                        cross-area utilities
│   ├── init.py                   weight-init factories
│   ├── qk_norm.py                QK normalization (apply + L2Norm)
│   ├── rope.py                   rotary position embedding (1D / 2D / 3D)
│   └── quack_utils.py            QuACK capability probe
├── metrics/
│   └── cleanfid.py               FID via cleanfid
└── testing/
    └── utils.py                  compute_relative_error
```

## What each top-level directory does

**`nvsubquadratic/`** — The library.  Function-only ops, `nn.Module`
building blocks, full networks, context-parallel primitives, and a
deferred-instantiation system (`LazyConfig`) that every example
config relies on.  See {doc}`api_reference/index` for the curated API.

**`experiments/`** — The training driver.  Lightning wrappers,
datamodules, callbacks, the `construct_trainer` helper, and the
`run.py` CLI entry point.  Consumes a network + datamodule + wrapper
via a `LazyConfig` tree from `examples/` and runs it through Lightning.

**`examples/`** — Per-task training recipes.  Each subdirectory is a
config tree (LazyConfig dataclasses) that fully describes one
experiment.  Running it is
`python -m experiments.run --config examples/.../<config>.py`.  The
live roadmap is at
[`examples/overview_tracker.md`](https://github.com/NVIDIA-BioNeMo/nvSubquadratic-private/blob/main/examples/overview_tracker.md).

**`benchmarks/`** — The single home for performance measurement.
Op-level microbenchmarks (`benchmarks/ops/`), end-to-end model
throughput (`vit5_imagenet/`, `well/`), and FLOP / scaling
comparisons.  Headline numbers are pulled into the {doc}`benchmarks`
docs page.

**`reports/`** — Frozen-in-time technical investigations.  One
`REPORT.md` per topic alongside the scripts and figures it cites.
Indexed at {doc}`reports`.

**`scripts/`** — Utility / glue scripts.  SLURM submit drivers
(`scripts/slurm/`), data prep (`scripts/data/`), evaluation helpers
(`scripts/evaluation/`), kernel viewers (`scripts/visualization/`),
and standalone sanity scripts.  No benchmarks live here — those moved
to `benchmarks/`.

**`tests/`** — Correctness tests, mirroring the library's per-package
structure: `tests/ops/`, `tests/modules/`, `tests/networks/`,
`tests/parallel/`.

**`docs/`** — This Sphinx documentation site.  Narrative pages
(Getting Started, Architecture, this Repository Overview, Examples,
Benchmarks, Reports, Ops Overview) plus the curated
{doc}`api_reference/index`.

## Where to go next

- {doc}`architecture` — the three-layer
  nvSubquadratic / subquadratic-ops / megatron-core story.
- {doc}`api_reference/index` — the curated API for each
  `nvsubquadratic/` area.
- {doc}`examples/index` — per-dataset training recipes that compose
  these networks.
- {doc}`ops/README` — math primer for the FFT convolution primitives.
