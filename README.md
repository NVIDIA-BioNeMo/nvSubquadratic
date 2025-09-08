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
- Python 3.12 or higher

## Architecture

nvSubquadratic provides a **high-level PyTorch interface** that depends on the **subquadratic-ops library** for optimized CUDA kernels. This separation provides clear boundaries between API design and performance optimization:

- **nvSubquadratic**: Focuses on API design, user experience, and PyTorch integration
- **subquadratic-ops**: Focuses on kernel optimization and CUDA performance

## Installation

### Dev Container (Recommended)

```bash
# Set GitLab token for subquadratic-ops
export GITLAB_TOKEN="your_gitlab_token_here"

# Open in VS Code and select "Reopen in Container"
```

### Docker

```bash
# Set GitLab token
export GITLAB_TOKEN="your_gitlab_token_here"

# Build and run
docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t nvsubquadratic:dev .
docker run --gpus all -p 8888:8888 -v $(pwd):/workspace nvsubquadratic:dev
```

### Local Installation

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install package
pip install -e .

# Install subquadratic-ops (requires GitLab token)
export GITLAB_TOKEN="your_gitlab_token_here"
pip install subquadratic-ops==v0.0.1+cuda12.9 --index-url https://__token__:${GITLAB_TOKEN}@gitlab-master.nvidia.com/api/v4/projects/180496/packages/pypi/simple
```

## Development

Pre-commit hooks are automatically installed in the dev container. For other installation methods:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

### Pre-commit Hooks

**On commit:**

- Code formatting (Ruff)
- Import sorting (Ruff)
- YAML validation
- Markdown formatting
- Secret detection

**On push:**

- Runs all tests (push is blocked if tests fail)
