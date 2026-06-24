# Repository overview

A map of what lives where at the repo root.  The library code
(`nvsubquadratic/`) and the training driver (`experiments/`) sit alongside
the per-task configs (`examples/`), the perf-measurement tree
(`benchmarks/`), and the supporting infrastructure (`scripts/`, `tests/`,
`reports/`, `docs/`).

## Layout

```text
nvSubquadratic/
├── nvsubquadratic/     library code — ops, modules, networks, parallel, utils
├── experiments/        PyTorch Lightning training driver (run.py, wrappers, datamodules, callbacks)
├── examples/           per-task LazyConfig training recipes fed to experiments.run
├── benchmarks/         performance measurement (throughput, FLOP / scaling, op-level)
├── reports/            frozen-in-time technical investigations with regen scripts
├── scripts/            utilities (data prep, evaluation, SLURM, visualization)
├── tests/              correctness tests mirroring the library layout
├── docs/               this Sphinx documentation site
├── CONVENTIONS.md      docstring style guide and PR checklist
├── README.md           top-level install / overview
├── pyproject.toml      project metadata, dependencies, ruff config
├── Dockerfile          production container
├── nvsubquadratic.def  Apptainer / Singularity recipe
└── setup_conda_env.sh  local conda bootstrap
```

## The library (`nvsubquadratic/`)

The library is organised bottom-up: function-only convolution primitives,
then `nn.Module`-shaped building blocks, then full architectures.

- **`ops/`**: function-only FFT convolution primitives (linear / circular /
  mixed boundary, fp32 / fp16, chunked, and fused-CUDA wrappers).
- **`modules/`**: `nn.Module` building blocks: mixers (Hyena, Mamba,
  attention, CKConv), learned kernels, residual blocks, norms, and MLPs.
- **`networks/`**: end-to-end architectures (ResNet / CCNN, ViT-5, the JiT
  diffusion backbone, and UNet-ConvNeXt baselines).
- **`parallel/`**: context-parallel primitives (`init_parallel_state`,
  AllToAll, zigzag split / gather).
- **`utils/`**, **`metrics/`**, **`testing/`**: weight init, RoPE, QK-norm,
  and the QuACK probe; FID; relative-error helpers.

See {doc}`api_reference/index` for the curated, per-symbol API.

## Where to go next

- {doc}`architecture`: the three-layer
  nvSubquadratic / subquadratic-ops / megatron-core story.
- {doc}`api_reference/index`: the curated API for each
  `nvsubquadratic/` area.
- {doc}`ops/README`: math primer for the FFT convolution primitives.
