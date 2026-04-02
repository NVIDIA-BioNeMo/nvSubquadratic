#!/usr/bin/env bash
# Set up the nvsubquadratic conda development environment.
#
# Usage:
#   bash setup_conda_env.sh            # create or update the environment
#
# Prerequisites:
#   - conda (Miniforge / Miniconda / Anaconda)
#   - NVIDIA GPU with CUDA 12.9 drivers
#   - nvcc on PATH (included in the nvcr.io CUDA devel image; on bare metal
#     install the CUDA 12.9 toolkit separately)

set -euo pipefail

ENV_NAME=nvsubquadratic
PYTHON_VERSION=3.12
TORCH_INDEX=https://download.pytorch.org/whl/cu129
TORCH_VERSION=2.10.0
TORCHVISION_VERSION=0.25.0

# ── 1. Create or recreate the conda environment ───────────────────────────────
if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "Removing existing '${ENV_NAME}' environment..."
    conda env remove -n "${ENV_NAME}" -y
fi

echo "Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" ninja -y

# Use full paths — avoids `conda run` resolving to the wrong active environment.
ENV_PREFIX=$(conda env list | grep -E "^${ENV_NAME}[[:space:]]" | awk '{print $NF}')
PIP="${ENV_PREFIX}/bin/pip"

# ── 2. Install PyTorch ────────────────────────────────────────────────────────
echo "Installing PyTorch ${TORCH_VERSION} (CUDA 12.9)..."
"${PIP}" install --no-cache-dir \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX}"

# ── 3. Install dev + project dependencies ────────────────────────────────────
echo "Installing development dependencies..."
"${PIP}" install --no-cache-dir -r requirements-dev.txt

echo "Installing the project with [quack] extra..."
"${PIP}" install --no-cache-dir --no-build-isolation -e ".[quack]"

# ── 4. Build and install NVIDIA Apex from source ─────────────────────────────
echo "Building NVIDIA Apex from source (this may take several minutes)..."
"${PIP}" install -v \
    --disable-pip-version-check \
    --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git

# ── 5. Set up pre-commit hooks ────────────────────────────────────────────────
if [ -f ".pre-commit-config.yaml" ]; then
    echo "Installing pre-commit hooks..."
    "${ENV_PREFIX}/bin/pre-commit" install
    "${ENV_PREFIX}/bin/pre-commit" install --hook-type pre-push
fi

echo ""
echo "Setup complete. Activate the environment with:"
echo "  conda activate ${ENV_NAME}"
