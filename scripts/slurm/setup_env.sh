#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=all
#SBATCH --time=01:00:00
#SBATCH --job-name=setup-env
#SBATCH --container-image=/shared/images/nvsubquadratic_cuda129.sqsh
#SBATCH --container-mounts=/home/david.romero:/home/david.romero,/shared:/shared,/scratch:/scratch
#SBATCH --output=/home/david.romero/projects/nvSubquadratic-private/logs/setup-env_%j.out
#SBATCH --error=/home/david.romero/projects/nvSubquadratic-private/logs/setup-env_%j.err

set -euo pipefail

PROJECT_ROOT=/home/david.romero/projects/nvSubquadratic-private
ENV_DIR=/home/david.romero/miniconda3/envs/nv-subq
CONDA_BASE=/home/david.romero/miniconda3

echo "=== $(hostname) — $(date) ==="
nvidia-smi
export CUDA_HOME=/usr/local/cuda
echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $($CUDA_HOME/bin/nvcc --version | tail -1)"

# Use miniconda base Python (mounted from HOME) to create the venv
echo "=== Creating nv-subq venv ==="
PYTHON="$CONDA_BASE/bin/python"
echo "Using: $PYTHON ($($PYTHON --version))"
rm -rf "$ENV_DIR"
"$PYTHON" -m venv "$ENV_DIR" --without-pip
source "$ENV_DIR/bin/activate"
python3 --version

# Bootstrap pip
echo "=== Bootstrapping pip ==="
python3 -c "import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', '/tmp/get-pip.py')"
python3 /tmp/get-pip.py
pip --version

# Install torch cu129
echo "=== Installing torch cu129 ==="
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu129

# Verify torch sees the GPU
echo "=== Verifying torch+CUDA ==="
python3 -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0)}')"

# Install project deps from pyproject.toml
echo "=== Installing project ==="
cd "$PROJECT_ROOT"
pip install -e "."

# Install NVIDIA apex from source with CUDA extensions
echo "=== Installing NVIDIA apex (with CUDA extensions) ==="
pip install -v --no-build-isolation --no-cache-dir \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    https://github.com/NVIDIA/apex/archive/refs/heads/master.zip

# Install quack-kernels (--no-deps to avoid pulling CUDA 13 torch, then add its deps individually)
echo "=== Installing quack-kernels ==="
pip install --no-deps quack-kernels==0.3.9
pip install --no-deps nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base apache-tvm-ffi torch-c-dlpack-ext

# Install the_well
echo "=== Installing the_well ==="
pip install the_well==1.2.0

# Verify key imports
echo "=== Final verification ==="
python3 -c "
import torch; print(f'torch={torch.__version__}')
from apex.optimizers import FusedLAMB
import apex_C; print('apex CUDA extensions=OK')
m = torch.nn.Linear(8, 8).cuda()
opt = FusedLAMB(m.parameters(), lr=1e-3); print('apex FusedLAMB=OK')
import nvidia.dali; print(f'dali={nvidia.dali.__version__}')
import pytorch_lightning as pl; print(f'lightning={pl.__version__}')
import omegaconf; print(f'omegaconf={omegaconf.__version__}')
import timm; print(f'timm={timm.__version__}')
import wandb; print(f'wandb={wandb.__version__}')
import nvsubquadratic; print('nvsubquadratic=OK')
import quack; print('quack-kernels=OK')
print('ALL IMPORTS OK')
"

echo "=== DONE — nv-subq env ready at $ENV_DIR ==="
