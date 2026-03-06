# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nvSubquadratic is a PyTorch-native library for subquadratic alternatives to attention (primarily multi-dimensional Hyena operators). It targets NVIDIA GPUs (Ampere/Hopper) and depends on `subquadratic-ops` for optimized CUDA kernels. The main use case is vision model training (ViT-5 variants) on ImageNet at scale using multi-node SLURM clusters.

## Common Commands

```bash
# Install (editable, after PyTorch with CUDA is installed)
pip install -r requirements-dev.txt
pip install --no-build-isolation -e .

# Run all tests
python -m pytest nvsubquadratic/ tests/ -v --tb=short

# Run a single test file or test
python -m pytest tests/test_basic.py -v
python -m pytest tests/test_basic.py::test_name -v

# Distributed tests (requires 2+ GPUs)
torchrun --nproc_per_node=2 tests/torchrun_sequence_mixer_cp_test.py --context_parallel_size=2 --dtype=float32

# Lint and format
ruff check --fix .
ruff format .

# Run pre-commit hooks manually
pre-commit run --all-files

# Launch a training experiment
python experiments/run.py examples/vit5_imagenet/v2/vit5_small_pretrain_apex_dali_fused.py

# Launch with config overrides (dot notation)
python experiments/run.py examples/vit5_imagenet/v2/vit5_small_pretrain_apex_dali_fused.py train.batch_size=64 train.learning_rate=1e-3

# SLURM submission
sbatch slurm/submit.sh examples/vit5_imagenet/v2/vit5_small_pretrain_apex_dali_fused.py
```

## Architecture

### Lazy Configuration System

All components use `LazyConfig` (in `nvsubquadratic/lazy_config.py`) for deferred instantiation. Configs are pure Python files returning `ExperimentConfig` via `get_config()`. Components are specified as `LazyConfig(ClassName)(param1=val1, ...)` and only instantiated at runtime by `instantiate()`. This enables nested composition—a network config contains block configs which contain mixer configs, etc.

### Training Flow

`experiments/run.py` orchestrates everything:
1. Loads a Python config file → `ExperimentConfig` dataclass (`experiments/default_cfg.py`)
2. Applies CLI overrides via dot notation
3. Instantiates: datamodule → network → lightning wrapper → trainer
4. Handles checkpoint resume (W&B autoresume or explicit checkpoint path)
5. Runs `trainer.fit()` → `trainer.validate()` → `trainer.test()`

`experiments/trainer.py` constructs the PyTorch Lightning `Trainer` with callbacks (checkpointing, W&B upload, walltime shutdown for SLURM, cache cleanup).

### Module Hierarchy

- **Networks** (`nvsubquadratic/networks/`): Full architectures (e.g., `ViT5ClassificationNet`). Accept/return dicts.
- **Modules** (`nvsubquadratic/modules/`): Building blocks composed via LazyConfig:
  - `vit5_residual_block.py`: Pre-norm residual blocks with LayerScale and DropPath
  - `vit5_attention.py`: Multi-head attention with RoPE and QK normalization
  - `sequence_mixer.py`: QKV wrapper that applies inner mixers (Hyena, attention, etc.)
  - `hyena_nd.py`: Gated global convolutional mixer (the core subquadratic operator)
  - `mlp.py`: FFN with configurable activations (GELU, GLU, SwiGLU)
  - `patchify.py`, `film.py`, `ckconv_nd.py`, `kernels_nd.py`: Supporting modules
- **Ops** (`nvsubquadratic/ops/`): Low-level operations and their tests
- **Lightning Wrappers** (`experiments/lightning_wrappers/`): Task-specific training logic (classification, diffusion, regression, autoregressive). The classification wrapper handles hard labels, soft targets (Mixup/CutMix), and BCE.

### Data Pipeline

`experiments/datamodules/` contains PyTorch Lightning data modules. The primary one for ImageNet is `dali_imagenet_fused.py`—a GPU-resident NVIDIA DALI pipeline with fused decode/crop/augmentations. It supports NVMe data staging for distributed training.

### Experiment Configs

Configs live in `examples/` organized by task (e.g., `examples/vit5_imagenet/`). Each is a Python file defining an `ExperimentConfig` with nested `LazyConfig` objects for all components. Finetune configs import from pretrain configs and override specific fields.

### SLURM Integration

`slurm/submit.sh` handles multi-node training: deterministic run naming (MD5 of config), container setup, W&B autoresume, and walltime-aware checkpointing.

## Code Conventions

- **Ruff** for linting and formatting (line length 119, Google-style docstrings)
- **License headers** required on all Python files in `nvsubquadratic/`, `tests/`, `examples/` (checked via `scripts/license_check.py`, run manually with `pre-commit run license-check --hook-type manual`)
- **Pytest** with coverage (`--cov=nvsubquadratic`). Slow tests marked with `@pytest.mark.slow`
- Pre-push hook runs full test suite; push is blocked on failure
- `isort` via Ruff with `nvsubquadratic` as known first-party
