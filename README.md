# nvSubquadratic

A unified PyTorch-native library for subquadratic alternatives to quadratic attention methods.

## Overview

nvSubquadratic consolidates efforts from across NVIDIA Research teams (nvResearch, NeMo, BioNeMo) into a single, consistent API. Currently supporting multi-dimensional (1D, 2D, 3D) Hyena operators with optimized CUDA kernels. Hyena operators provide subquadratic alternatives to attention mechanisms, achieving O(N) or O(N log N) complexity compared to the O(N²) scaling of traditional attention.

## Dependencies

**subquadratic-ops**: This library depends on `subquadratic-ops` for high-performance CUDA kernels. The `subquadratic-ops` library provides optimized implementations of:

- **B2B CausalConv1d**: Back-to-back causal convolutions for striped Hyena architectures
- **CausalConv1d**: Standard causal convolutions with various kernel sizes (2-256)
- **FFT CausalConv1d**: FFT-based causal convolutions for large kernel sizes (up to 8K-16M)

**Requirements**:

- CUDA-compatible NVIDIA GPU (Ampere or Hopper architecture)
- CUDA Toolkit 12.0 or higher
- Python 3.11 or higher

**quack-kernels (optional)**:

- Used by `RMSNorm` for a fused CUDA kernel when available.
- **Supported GPUs**: Hopper and Blackwell only (H100, B200, B300). There is no quack-kernels build for Ampere (e.g. RTX A6000, A100) or older architectures.
- On unsupported GPUs or when quack is not installed, `RMSNorm` uses a pure-PyTorch fallback automatically. No separate “version” fixes Ampere; use the fallback or run on H100/B200 for the kernel.
- Optional install: `pip install quack-kernels` (on H100/B200); or `pip install -e ".[quack]"` if using the optional dependency group.

## Architecture

nvSubquadratic provides a **high-level PyTorch interface** that depends on the **subquadratic-ops library** for optimized CUDA kernels. This separation provides clear boundaries between API design and performance optimization:

- **nvSubquadratic**: Focuses on API design, user experience, and PyTorch integration
- **subquadratic-ops**: Focuses on kernel optimization and CUDA performance
- **megatron-core**: Provides distributed training and model parallelism capabilities

## Installation

### Package Manager

This project uses **pip** with `pyproject.toml` for dependency management. A `Pipfile.lock` is maintained for nSpect security scanning compliance.

### Dev Container (Recommended)

Open in VS Code and select "Reopen in Container". The devcontainer extension will automatically build the Docker image and set up the development environment with all dependencies pre-installed.

### Docker

```bash
# Build and run
docker build -t nvsubquadratic:dev .
docker run --gpus all -p 8888:8888 -v $(pwd):/workspaces/nvSubquadratic-private nvsubquadratic:dev
```

The Dockerfile builds NVIDIA Apex from source for a broad set of NVIDIA archs by default (`7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0` — Volta through Blackwell). Two build-args let you tune the compile:

- `TORCH_CUDA_ARCH_LIST` — narrow to your GPU(s) to speed up the build (e.g. `9.0` for H100, `8.6` for A6000, `8.9` for L4).
- `MAX_JOBS` — number of parallel nvcc jobs. Defaults to unconstrained. Set to a small number (e.g. `2`) if the build OOMs (typical under qemu emulation).

```bash
docker build \
    --build-arg TORCH_CUDA_ARCH_LIST="9.0" \
    -t nvsubquadratic:dev .
```

### Enroot (SLURM clusters)

For SLURM deployments that use enroot/pyxis, [`scripts/slurm/enroot/build_sqsh.sh`](scripts/slurm/enroot/build_sqsh.sh) builds the Docker image and converts it to an enroot `.sqsh` in one step. It selects the right `TORCH_CUDA_ARCH_LIST` and `MAX_JOBS` per platform:

```bash
# H100 (x86-64, default)
scripts/slurm/enroot/build_sqsh.sh

# GB200 (ARM64) — uses qemu emulation on an x86 build host
PLATFORM=arm64 scripts/slurm/enroot/build_sqsh.sh
```

### Apptainer

```bash
# Build SIF (add --fakeroot if required on your system)
apptainer build nvsubquadratic.sif nvsubquadratic.def

# Interactive shell with GPUs and live code from your checkout
apptainer shell --nv --bind $(pwd):/workspaces/nvSubquadratic-private nvsubquadratic.sif

# Run a command inside the image (example: tests)
apptainer exec --nv --bind $(pwd):/workspaces/nvSubquadratic-private nvsubquadratic.sif python -m pytest nvsubquadratic/ tests/

# Use the default runscript (starts Jupyter Lab as defined in the .def)
apptainer run --nv --bind $(pwd):/workspaces/nvSubquadratic-private nvsubquadratic.sif --no-browser
```

### Conda (recommended for local development)

```bash
bash setup_conda_env.sh
conda activate nvsubquadratic
```

This script creates the `nvsubquadratic` conda environment with Python 3.12 and PyTorch 2.10 (CUDA 12.9), installs all dev dependencies, builds NVIDIA Apex from source, and installs `quack-kernels`.

### Local Installation (venv)

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support first (before package dependencies)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu129

# Install development dependencies
pip install -r requirements-dev.txt

# Install the package in editable mode (installs remaining dependencies from pyproject.toml)
pip install --no-build-isolation -e .
```

## Development

Pre-commit hooks are automatically installed in the dev container. For other installation methods:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

### Updating Dependencies for Security Scanning

This project maintains a `Pipfile.lock` for nSpect security scanning compliance. When you update dependencies in `pyproject.toml`, regenerate the lock file:

```bash
# Install pipenv (if not already installed)
pip install pipenv

# Regenerate Pipfile.lock
pipenv lock

# Note: Continue using pip for actual installations (pip install -e .)
```

### Testing

See [`tests/README.md`](tests/README.md) for full details on test suites, markers, and SLURM usage.

```bash
# Unit tests (CPU-safe, no external data needed)
PYTHONPATH=. python -m pytest tests/ -m "not nightly" -v -o addopts=""

# Nightly validation (requires GPU, DALI, ImageNet, wandb)
source .env && PYTHONPATH=. python -m pytest tests/ -m nightly -v -o addopts=""
```

### Documentation

All public classes and functions carry **Google-style docstrings** with math
context, shape annotations, and paper references.  See [`CONVENTIONS.md`](CONVENTIONS.md)
for the style guide and PR checklist.

#### Viewing the docs

The API reference is built with Sphinx. Sources live under [`docs/`](docs/) and
the rendered site is published to the `gh-pages` branch on every push to `main`
via [`.github/workflows/docs.yml`](.github/workflows/docs.yml).

Build and preview locally:

```bash
pip install -r docs/requirements.txt
pip install -e . --no-deps
make -C docs html SPHINXBUILD="python -m sphinx"
python -m http.server 8000 --directory docs/_build/html
```

Open <http://localhost:8000> to browse.  The autosummary stubs under
`docs/api/generated/` are regenerated on every build (gitignored).

While editing, you can also hover over any symbol in VS Code / PyCharm to
see the rendered docstring, or run `help(SomeClass)` in a REPL.

### CI

GPU tests run automatically on pull requests via a self-hosted runner.
Runner provisioning is maintained out-of-tree; contact the maintainers for access.

### Pre-commit Hooks

**On commit:**

- Code formatting (Ruff)
- Import sorting (Ruff)
- YAML validation
- Markdown formatting
- Secret detection

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for the
DCO sign-off requirement and the PR/issue flow.  Pull requests from
external forks run through the same CI pipeline; the GPU stage requires a
codeowner ([.github/CODEOWNERS](.github/CODEOWNERS)) to approve workflow
runs from outside collaborators before the self-hosted runner picks them
up — this is the standard GitHub "Require approval for outside
collaborators" gate.

For security-sensitive findings, please follow [SECURITY.md](SECURITY.md)
instead of opening a public issue or PR.
