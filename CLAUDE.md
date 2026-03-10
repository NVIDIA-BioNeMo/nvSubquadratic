# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**nvSubquadratic** is a PyTorch-native library for subquadratic alternatives to attention (primarily multi-dimensional Hyena operators). It is used for large-scale training experiments on NVIDIA infrastructure, with the primary focus being ImageNet classification using ViT-5-style architectures with Hyena sequence mixers as attention replacements.

Dependencies:
- `subquadratic-ops` (CUDA kernels for CausalConv1d, FFT conv, etc.)
- `megatron-core` (distributed training / model parallelism)
- `pytorch-lightning` (training loop)
- `NVIDIA DALI` (high-performance data loading for ImageNet)

Requires CUDA 12+, Python 3.11+, Ampere or Hopper GPU.

## Commands

### Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt
pip install --no-build-isolation -e .
```

### Lint / Format

```bash
ruff check .         # lint
ruff format .        # format
ruff check --fix .   # auto-fix
```

### Tests

```bash
# All tests (from repo root):
PYTHONPATH=. python -m pytest nvsubquadratic/ tests/

# Single test file:
PYTHONPATH=. python -m pytest tests/test_vit5_components.py -v -o "addopts="

# Skip slow tests:
PYTHONPATH=. python -m pytest tests/ -m "not slow"

# Distributed tests (requires 2+ GPUs):
torchrun --nproc_per_node=2 tests/torchrun_sequence_mixer_cp_test.py
```

### Run an Experiment

```bash
PYTHONPATH=. python -m experiments.run --config examples/vit5_imagenet/v2/vit5_small_pretrain_hyena_gap_apex.py

# With CLI overrides (dot-notation):
PYTHONPATH=. python -m experiments.run --config examples/... dataset.batch_size=64 train.iterations=1000
```

### SLURM (cluster)

Submit via `slurm/submit.sh` (4-node, 32-GPU). Edit `CONFIG_FILE`, `CONFIG_OVERRIDES`, `DATA_DIR`, `WANDB_API_KEY` inside the script before submitting:

```bash
sbatch slurm/submit.sh
```

The submit script automatically handles checkpoint resume via `autoresume.enabled=True` if a `last.ckpt` exists.

### Pre-commit hooks

```bash
pre-commit install
pre-commit install --hook-type pre-push   # runs full test suite on push
```

## Architecture

### Two top-level Python packages

- **`nvsubquadratic/`** — the library (installable as `nvsubquadratic`)
- **`experiments/`** — training infrastructure (not installed; always run with `PYTHONPATH=.`)

### `nvsubquadratic/` structure

| Directory | Purpose |
|-----------|---------|
| `modules/` | Building blocks: `HyenaND`, `CKConvND`, `SIRENKernelND`, `QKVSequenceMixer`, `ViT5ResidualBlock`, `MLP`, `RMSNorm`, FiLM, Mamba, etc. |
| `networks/` | Full network architectures: `ViT5ClassificationNet`, `HuggingFaceDiffusers`, ResNets, JiT baselines |
| `ops/` | Low-level ops (fftconv, wrappers around `subquadratic-ops-torch`) |
| `parallel/` | Context-parallel / distributed utilities |
| `metrics/` | FID and other evaluation metrics |
| `lazy_config.py` | Custom config system (see below) |

### `experiments/` structure

| Directory/File | Purpose |
|----------------|---------|
| `run.py` | Main entry point for all experiments |
| `trainer.py` | Constructs the PyTorch Lightning `Trainer` with callbacks |
| `default_cfg.py` | Dataclass config schemas (`ExperimentConfig`, `DiffusionExperimentConfig`, etc.) |
| `datamodules/` | Data modules: DALI ImageNet (`dali_imagenet_fused.py`), MNIST, EMNIST, UCF101, TinyImageNet |
| `lightning_wrappers/` | PL modules: `ClassificationWrapper`, `DiffusionWrapper`, `RegressionWrapper` |
| `callbacks/` | Custom callbacks: walltime checkpointer, W&B uploader, image grid logger |
| `utils/` | CLI parsing, checkpointing utilities, W&B helpers |

### Config system (`LazyConfig`)

Configs are **plain Python files** (not YAML). Each config file defines an `ExperimentConfig` (or `DiffusionExperimentConfig`) instance and assigns `LazyConfig` objects for lazy instantiation.

`LazyConfig(SomeClass)(arg1=val, arg2=val)` stores the class reference + args. `instantiate(cfg)` resolves and constructs the object at runtime.

Config files live in `examples/<task>/<variant>.py`. They are self-contained and import directly from `nvsubquadratic` and `experiments`.

CLI overrides use dot-notation: `dataset.batch_size=256`, `scheduler.name=wsd`.

### Key training flow (`experiments/run.py`)

1. Load Python config file → parse CLI overrides
2. Instantiate datamodule, network, Lightning wrapper
3. Optionally `torch.compile` the network
4. Set up W&B logger (with autoresume support)
5. Optionally download + load pretrained checkpoint (`start_from_checkpoint`)
6. `trainer.fit()` → `trainer.validate()` → `trainer.test()`

### Active research focus: ViT-5 ImageNet

Primary experiment configs: `examples/vit5_imagenet/v2/`. Key ablations:
- **Attention baseline**: `ViT5Attention` (standard multi-head attention)
- **Hyena-GAP**: Hyena replaces attention, no CLS token, global average pooling readout
- **Hyena-CLS-row**: Hyena with CLS token placed as an extra row in the 2D grid (14×14 → 15×14)
- **Multi-head Hyena**: multiple Hyena heads with dense within-head mixing

ViT-5-Small: 12 blocks, dim 384, patch size 16, 224×224, LAMB optimizer, batch 2048, 800 epochs.

Training uses DALI fused pipeline (`experiments/datamodules/dali_imagenet_fused.py`) with local NVMe staging. W&B project: `implicit-long-convs/nvsubquadratic`.

### Schedulers

Supported: `cosine`, `wsd` (warmup-stable-decay), `constant`. Configured via `SchedulerConfig` with `warmup_iterations_percentage`, `stable_iterations_percentage`, `total_iterations`.

### Ruff configuration

Line length 119. Google-style docstrings. `__init__.py` and test files have relaxed rules.
