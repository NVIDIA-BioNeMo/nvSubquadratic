# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nvSubquadratic is a PyTorch-native library for subquadratic alternatives to attention, currently implementing multi-dimensional (1D, 2D, 3D) Hyena operators. It provides O(N) or O(N log N) complexity vs O(N²) for attention. Developed across NVIDIA Research teams (nvResearch, NeMo, BioNeMo).

## Build & Install

```bash
# Dev container (recommended): VS Code → "Reopen in Container"
# Local:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt
pip install --no-build-isolation -e .
pre-commit install && pre-commit install --hook-type pre-push
```

## Testing

```bash
python -m pytest nvsubquadratic/ tests/ -v              # all tests
python -m pytest tests/test_causality_hyena.py -v       # single file
python -m pytest -m "not slow" tests/                   # skip slow tests
torchrun --nproc_per_node=2 tests/torchrun_sequence_mixer_cp_test.py --context_parallel_size=2 --dtype=float32  # distributed (2+ GPUs)
```

Pytest is configured in `pyproject.toml` with coverage (`--cov=nvsubquadratic`) and duration reporting enabled by default. Pre-push hooks run the full test suite and block on failure.

## Linting & Formatting

Ruff handles both linting and formatting. Pre-commit hooks run automatically on commit.

```bash
pre-commit run --all-files    # run all hooks manually
ruff check --fix .            # lint with auto-fix
ruff format .                 # format
```

Config: `pyproject.toml` — line length 119, Google-style docstrings, isort with `nvsubquadratic` as first-party.

## Running Experiments

```bash
PYTHONPATH=. python experiments/run.py --config examples/vit5_imagenet/v2/vit5_small_pretrain_multihead_hyena_cls_row_apex_fix_init.py
PYTHONPATH=. python experiments/run.py --config <config.py> --config.train.batch_size=256  # CLI overrides
```

## Architecture

### Core: Hyena Operator (`nvsubquadratic/modules/hyena_nd.py`)

Gated global convolutional mixer. Per-block computation:
```
Q, K, V ← linear projections (done externally)
  → short_conv([Q; K; V])     # depthwise short conv on concatenated QKV
  → optional RoPE(Q, K)       # rotary positional encoding
  → optional QK-Norm(Q, K)    # per-channel normalization
  → z = Q ⊙ σ(K)              # first multiplicative gate
  → optional PixelHyena-Norm  # GroupNorm/RMSNorm
  → h = GlobalConv(z)         # long-range conv (FFTConv or CKConv)
  → y = h ⊙ σ(V)              # second multiplicative gate
  → optional Output-Norm
```

### FFT Convolutions (`nvsubquadratic/ops/`)

- `fftconv.py` — 1D/2D/3D FFT convolutions with causal, zero-padded, and circular modes
- `fftconv_chunked.py` — memory-efficient chunked variants
- Supports BLH `[batch, *spatial, hidden]` and BHL `[batch, hidden, *spatial]` layouts

### ViT-5 Integration (`nvsubquadratic/modules/vit5_*.py`, `nvsubquadratic/networks/vit5_classification.py`)

Vision Transformer 5 architecture for ImageNet. Patch embedding → learnable APE → CLS/register tokens → N residual blocks (pre-norm, attention/Hyena, LayerScale, DropPath, MLP). Supports RoPE precomputation, SDPA backend auto-selection, and torch.compile.

### Lazy Configuration (`nvsubquadratic/lazy_config.py`)

Deferred instantiation system using `LazyConfig`. Stores class references with arguments, resolves at instantiation time. Used throughout experiment configs.

```python
from nvsubquadratic.lazy_config import LazyConfig, instantiate
cfg = LazyConfig(SomeModule)(param1=value1, nested=LazyConfig(Other)())
module = instantiate(cfg)
```

### Key Module Map

| Path | Purpose |
|------|---------|
| `nvsubquadratic/modules/` | Core PyTorch modules (Hyena, attention, CKConv, MLP, norms, etc.) |
| `nvsubquadratic/ops/` | FFT-based convolution operations |
| `nvsubquadratic/networks/` | Complete model architectures (ViT-5, ResNet, diffusion) |
| `nvsubquadratic/parallel/` | Distributed training (All-to-All context parallelism) |
| `nvsubquadratic/utils/` | RoPE, QK normalization helpers |
| `experiments/` | PyTorch Lightning experiment framework (`run.py` is entry point) |
| `examples/` | Experiment configs (ImageNet, MNIST, spatial recall, diffusion) |
| `benchmarks/vit5_imagenet/` | Performance benchmarking and profiling scripts |

### Tensor Layout Convention

- **BLH**: `[batch, *spatial_dims, hidden]` — preferred for Hyena modules
- **BHL**: `[batch, hidden, *spatial_dims]` — preferred for FFT ops
- Wrapper functions handle automatic reshaping between layouts

### RoPE Dimensions

- 1D: applied over sequence length
- 2D: per-head dim split into (Y, X) halves, applied to (H, W)
- 3D: per-head dim split into (Z, X, Y) thirds, applied to (H, W, D)

## Code Conventions

- NVIDIA Apache 2.0 SPDX license headers required on all Python files (check: `pre-commit run license-check --all-files`)
- Google-style docstrings (enforced by ruff D rules)
- Type hints throughout
- `pyproject.toml` is the canonical dependency source; `Pipfile.lock` exists only for nSpect security scanning (regenerate with `pipenv lock`)

## Dependencies

Core: PyTorch ≥2.0, `subquadratic-ops-torch-cu12` (CUDA kernels), `megatron-core`, `einops`, `omegaconf`, `pytorch-lightning`, `wandb`, `timm==1.0.22`

Requires: Python ≥3.11, CUDA 12.0+, Ampere/Hopper GPU
