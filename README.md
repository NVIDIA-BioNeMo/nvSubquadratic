# nvSubquadratic

A unified PyTorch-native library for subquadratic alternatives to quadratic attention method.

## Overview

nvSubquadratic consolidates efforts from across NVIDIA Research teams (nvResearch, NeMo, BioNeMo) into a single, consistent API. Currently supporting multi-dimensional (1D, 2D, 3D) Hyena operators with optimized CUDA kernels. Hyena operators provide subquadratic alternatives to attention mechanisms, achieving O(N) or O(N log N) complexity compared to the O(N²) scaling of traditional attention.

## Key Features \[WIP\]

- **Multi-dimensional Support**: 1D (sequences), 2D (images), 3D (videos/volumes)
- **Multiple Operator Types**: Short (direct conv), Medium (parameterized), Long (FFT-based)
- **Flexible Directionality**: Causal and bidirectional variants
- **PyTorch-Idiomatic**: Follows Conv1d/Conv2d/Conv3d design patterns
- **Performance Tiers**: Reference PyTorch implementations with optional optimized CUDA kernels
- **ONNX Export**: Support for production deployment

## Installation

### Option 1: Dev Container (Recommended for VS Code Users)

The easiest way to get started with full IDE integration is using the provided dev container configuration.

#### Prerequisites

- Docker with NVIDIA Container Toolkit installed
- VS Code with the "Dev Containers" extension
- GitLab token for accessing private `subquadratic-ops` package

#### Setup and Run

1. **Set your GitLab token** (required for subquadratic-ops installation):

   ```bash
   export GITLAB_TOKEN="your_gitlab_token_here"
   ```

1. **Open in VS Code**:

   - Open this repository in VS Code
   - When prompted, click "Reopen in Container" or use Command Palette: `Dev Containers: Reopen in Container`

1. **Automatic Setup**:

   - The container will build automatically using the Dockerfile
   - All dependencies will be installed
   - VS Code extensions will be installed
   - The workspace will be ready for development

#### Dev Container Features

- **Full IDE Integration**: VS Code with Python, Jupyter, and development extensions
- **Pre-configured Environment**: Python interpreter, testing, linting, formatting
- **GPU Support**: NVIDIA Container Toolkit integration
- **Volume Mounts**: AWS credentials, SSH keys, cache directories
- **Git Integration**: GitLens extension for enhanced Git workflow
- **Code Quality**: Ruff, Pylance, auto-formatting, and spell checking
- **Jupyter Support**: Built-in Jupyter notebook support

#### Troubleshooting

**Container Build Issues:**

- Ensure Docker is running and has access to GPUs
- Verify `GITLAB_TOKEN` is set in your environment
- Check Docker logs if build fails

**VS Code Integration:**

- Install the "Dev Containers" extension if not already installed
- Use Command Palette: `Dev Containers: Rebuild Container` if issues persist

### Option 2: Docker Container (Standalone)

The easiest way to get started is using the provided Docker container with all dependencies pre-installed.

#### Prerequisites

- Docker with NVIDIA Container Toolkit installed
- GitLab token for accessing private `subquadratic-ops` package

#### Build and Run Container

1. **Set your GitLab token** (required for subquadratic-ops installation):

   ```bash
   export GITLAB_TOKEN="your_gitlab_token_here"
   ```

1. **Build the container**:

   ```bash
   docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t nvsubquadratic:dev .
   ```

1. **Run the container**:

   ```bash
   # For Jupyter Lab (recommended for development)
   docker run --gpus all -p 8888:8888 -v $(pwd):/workspace nvsubquadratic:dev

   # For interactive shell
   docker run --gpus all -it -v $(pwd):/workspace nvsubquadratic:dev /bin/bash
   ```

1. **Access Jupyter Lab** (if using Jupyter):

   - Open your browser to `http://localhost:8888`
   - Use the token provided in the terminal output

#### Container Features

- Pre-installed PyTorch with CUDA support
- All development dependencies (pytest, ruff, pre-commit, etc.)
- Jupyter Lab for interactive development
- Volume mounting for live code editing
- GPU support via NVIDIA Container Toolkit

#### Troubleshooting

**GitLab Token Issues:**

- If you don't have a GitLab token, the container will still build but without `subquadratic-ops`
- You can add the token later by rebuilding: `docker build --build-arg GITLAB_TOKEN=$GITLAB_TOKEN -t nvsubquadratic:dev .`

**GPU Access:**

- Ensure NVIDIA Container Toolkit is installed: `nvidia-docker` or `docker run --gpus all`
- Verify GPU access inside container: `python -c "import torch; print(torch.cuda.is_available())"`

**Port Conflicts:**

- If port 8888 is busy, use a different port: `docker run -p 8889:8888 ...`
- Update the Jupyter URL accordingly: `http://localhost:8889`

### Option 3: Local Installation

**Note:** This method requires manual setup of PyTorch with CUDA support, which can be complex. The container methods above are recommended.

```bash
# Install PyTorch with CUDA support (choose appropriate version for your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install from source
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt

# Install subquadratic-ops (requires GitLab token)
export GITLAB_TOKEN="your_gitlab_token_here"
pip install subquadratic-ops==v0.0.1+cuda12.9 --index-url https://__token__:${GITLAB_TOKEN}@gitlab-master.nvidia.com/api/v4/projects/180496/packages/pypi/simple
```

**Prerequisites for Local Installation:**

- Python 3.9+
- CUDA 12.1+ installed on your system
- GitLab token for subquadratic-ops
- Proper NVIDIA drivers

<!-- ## Quick Start

```python
import torch
from nvsubquadratic import Hyena1d, Hyena2d, Hyena3d

# 1D Sequence modeling (language, audio)
hyena_1d = Hyena1d(hidden_dim=512, operator_type="long", directionality="causal")
x1, x2, v = torch.randn(8, 512, 1000), torch.randn(8, 512, 1000), torch.randn(8, 512, 1000)
output_1d = hyena_1d(x1, x2, v)  # Shape: (8, 512, 1000)

# 2D Image processing
hyena_2d = Hyena2d(hidden_dim=256, operator_type="medium", directionality="bidirectional")
x1, x2, v = torch.randn(4, 256, 64, 64), torch.randn(4, 256, 64, 64), torch.randn(4, 256, 64, 64)
output_2d = hyena_2d(x1, x2, v)  # Shape: (4, 256, 64, 64)

# 3D Video/volumetric processing
hyena_3d = Hyena3d(hidden_dim=128, operator_type="short", temporal_causal=True)
x1, x2, v = torch.randn(2, 128, 16, 32, 32), torch.randn(2, 128, 16, 32, 32), torch.randn(2, 128, 16, 32, 32)
output_3d = hyena_3d(x1, x2, v)  # Shape: (2, 128, 16, 32, 32)
``` -->
