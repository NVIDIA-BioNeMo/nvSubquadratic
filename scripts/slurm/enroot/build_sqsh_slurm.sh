#!/bin/bash
#
# Build the nvSubquadratic .sqsh ON A SLURM COMPUTE NODE, without Docker.
#
# build_sqsh.sh needs `docker buildx` plus, for an arm64 image on an x86 host,
# QEMU binfmt emulation. Cluster login nodes typically have neither (no docker
# daemon, no qemu-aarch64 registered, no root to install either), which leaves
# no way to produce a GB200 image from the login node at all.
#
# This script routes around that: pyxis/enroot pulls the CUDA base image on a
# GB200 node, `--container-remap-root` gives root inside it, the Dockerfile's RUN
# steps are replayed natively, and `--container-save` writes the result out as a
# .sqsh. Building on the target architecture also removes QEMU entirely, so the
# apex/mamba compiles that build_sqsh.sh throttles to MAX_JOBS=1 to survive
# emulation run fully parallel here.
#
# Usage:
#   scripts/slurm/enroot/build_sqsh_slurm.sh
#
# Env overrides:
#   OUTPUT_SQSH   output path  (default: $LUSTRE_ENROOT/nvsubquadratic-arm64.sqsh)
#   PARTITION     slurm partition       (default: 36x2-a01r)
#   ACCOUNT       slurm account         (default: healthcareeng_bionemo)
#   TIME_LIMIT    slurm walltime        (default: 04:30:00; 36x2-a01r caps at 5h)
#   CUDA_ARCHS    TORCH_CUDA_ARCH_LIST  (default: 10.0;12.0 — GB200 + B200/5090)
#   MAX_JOBS      parallel build jobs   (default: 32)
#   NVCC_THREADS  nvcc --threads        (default: 4)
#   INSTALL_MAMBA / INSTALL_FA4         (default: true — this is the benchmark image)
#
# Keep the pins below in sync with the Dockerfile — this replays it, it does not
# parse it, so nothing links the two at runtime. They no longer drift silently:
# scripts/check_version_pins.py fails pre-commit/CI if this file, the Dockerfile,
# setup_conda_env.sh or the install docs disagree with pyproject's torch range.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

LUSTRE_ENROOT="${LUSTRE_ENROOT:-/lustre/fsw/healthcareeng_bionemo/farhadr/enroot}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${LUSTRE_ENROOT}/nvsubquadratic-arm64.sqsh}"
PARTITION="${PARTITION:-36x2-a01r}"
ACCOUNT="${ACCOUNT:-healthcareeng_bionemo}"
TIME_LIMIT="${TIME_LIMIT:-04:30:00}"

BASE_IMAGE="${BASE_IMAGE:-nvcr.io#nvidia/cuda:13.0.3-devel-ubuntu22.04}"
CUDA_ARCHS="${CUDA_ARCHS:-10.0;12.0}"
MAX_JOBS="${MAX_JOBS:-32}"
NVCC_THREADS="${NVCC_THREADS:-4}"
INSTALL_MAMBA="${INSTALL_MAMBA:-true}"
INSTALL_FA4="${INSTALL_FA4:-true}"

# Pins mirrored from the Dockerfile. TORCH_VERSION must satisfy pyproject's
# `torch>=2.12.0,<2.13.0`: if it does not, step 10's `.[all]` silently upgrades
# torch after apex/mamba/causal-conv1d have already been compiled against the
# older one, leaving extensions built against headers that no longer match.
TORCH_VERSION="${TORCH_VERSION:-2.12.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.27.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
DALI_PACKAGE="${DALI_PACKAGE:-nvidia-dali-cuda130}"
MINIFORGE_VERSION="${MINIFORGE_VERSION:-25.3.0-3}"
MINIFORGE_NAME="${MINIFORGE_NAME:-Miniforge3}"
FLASH_ATTN4_VERSION="${FLASH_ATTN4_VERSION:-4.0.0b23}"
CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.6.0.dev0}"

LOG_DIR="${LOG_DIR:-${REPO_ROOT}/benchmarks/results}"
mkdir -p "${LOG_DIR}"
if [[ -e "${OUTPUT_SQSH}" ]]; then
    echo "Error: ${OUTPUT_SQSH} already exists. Move it aside or set OUTPUT_SQSH."
    exit 1
fi

echo "Base image:  ${BASE_IMAGE}"
echo "Output:      ${OUTPUT_SQSH}"
echo "Partition:   ${PARTITION}   walltime ${TIME_LIMIT}"
echo "Arches:      ${CUDA_ARCHS}   MAX_JOBS=${MAX_JOBS}  NVCC_THREADS=${NVCC_THREADS}"
echo "Baselines:   mamba=${INSTALL_MAMBA}  fa4=${INSTALL_FA4}"
echo "Torch:       ${TORCH_VERSION} / ${TORCHVISION_VERSION} from ${TORCH_INDEX_URL}"

# ── In-container build, replaying the Dockerfile's RUN steps ─────────────────
# The token reaches the container as a file (mounted read-only from $HOME), not
# as an env var: srun's environment is visible via `scontrol show job`, and a
# saved container would carry an exported env var into the image.
read -r -d '' BUILD <<'OUTER' || true
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export CONDA_DIR=/opt/conda
export LANG=C.UTF-8 LC_ALL=C.UTF-8
export PATH="${CONDA_DIR}/bin:${PATH}"

step() { echo; echo "══════ $* ══════"; date "+%H:%M:%S"; }

step "1/10 apt packages"
apt-get update > /dev/null
apt-get install --no-install-recommends --yes \
    wget bzip2 ca-certificates git tini sudo build-essential ninja-build > /dev/null

step "2/10 miniforge -> ${CONDA_DIR}"
wget --no-hsts --quiet \
    "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh" \
    -O /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p "${CONDA_DIR}"
rm /tmp/miniforge.sh
conda clean --tarballs --index-cache --packages --yes
find "${CONDA_DIR}" -follow -type f -name '*.a' -delete
find "${CONDA_DIR}" -follow -type f -name '*.pyc' -delete
conda clean --force-pkgs-dirs --all --yes
echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc
echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc
chmod -R a+rX "${CONDA_DIR}"

step "3/10 python 3.12"
conda install --yes python=3.12
conda clean --all --yes

step "4/10 torch ${TORCH_VERSION} + DALI (torch re-asserted after DALI)"
pip install --no-cache-dir torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
pip install --no-cache-dir "${DALI_PACKAGE}"
pip install --no-cache-dir torch=="${TORCH_VERSION}" torchvision=="${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"
conda clean --all --yes
python -c "import torch; print('torch', torch.__version__)"

step "5/10 ubuntu user"
groupadd -r ubuntu 2>/dev/null || true
useradd -r -g ubuntu -G sudo -m -s /bin/bash ubuntu 2>/dev/null || true
echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

step "6/10 apex (source build, arches ${TORCH_CUDA_ARCH_LIST})"
MAX_JOBS="${MAX_JOBS}" pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git

step "7/10 mamba baseline (INSTALL_MAMBA=${INSTALL_MAMBA})"
if [ "${INSTALL_MAMBA}" != "true" ]; then
    echo "skipped"
else
    # mamba-ssm depends on an unpinned torch; freeze the stack so pip cannot swap
    # torch / nvidia-cudnn out from under it (that breaks cuDNN at runtime).
    pip freeze | grep -iE '^(torch|torchvision|torchaudio|nvidia-|triton|pytorch-triton)' \
        > /tmp/mamba-constraints.txt
    git clone --depth 1 https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d
    git clone --depth 1 https://github.com/state-spaces/mamba.git /tmp/mamba
    python /mnt/src/scripts/docker/patch_mamba_cuda_arches.py \
        /tmp/causal-conv1d/setup.py /tmp/mamba/setup.py
    MAX_JOBS="${MAX_JOBS}" NVCC_THREADS="${NVCC_THREADS}" \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
        -c /tmp/mamba-constraints.txt /tmp/causal-conv1d /tmp/mamba
    rm -rf /tmp/causal-conv1d /tmp/mamba
fi

step "8/10 flash-attention-4 (INSTALL_FA4=${INSTALL_FA4})"
if [ "${INSTALL_FA4}" != "true" ]; then
    echo "skipped"
else
    # cutlass-dsl is EXCLUDED from the constraints on purpose: pinning it to a
    # version already in the image strands FA4 on a mismatched JIT API.
    pip freeze | grep -iE '^(torch|torchvision|torchaudio|nvidia-|triton|pytorch-triton)' \
        | grep -viE '^nvidia-cutlass-dsl' > /tmp/fa4-constraints.txt
    pip install --no-cache-dir --pre -c /tmp/fa4-constraints.txt \
        "nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}" \
        "flash-attn-4[cu13]==${FLASH_ATTN4_VERSION}"
    # Read the DSL version from package metadata: nvidia_cutlass_dsl exposes no
    # __version__, so reading the attribute fails and makes a working FA4 install
    # look broken.
    { python -c "
import importlib.metadata as md
import flash_attn.cute
from flash_attn.cute import flash_attn_func
print('flash-attn-4 OK / cutlass-dsl', md.version('nvidia-cutlass-dsl'))"; } \
        || echo "WARNING: flash-attn-4 import check failed at build; verify on-GPU."
fi

step "9/10 project source + dev deps"
mkdir -p /workspaces/nvSubquadratic
# Copy rather than bind-mount: the saved image must be self-contained. Exclude
# the local results/artifacts so they do not bloat the .sqsh.
tar -C /mnt/src --exclude=.git --exclude=benchmarks/results --exclude='*.sqsh' \
    -cf - . | tar -C /workspaces/nvSubquadratic -xf -
cd /workspaces/nvSubquadratic
pip install --no-cache-dir -r requirements-dev.txt
git config --global --add safe.directory /workspaces/nvSubquadratic

step "10/10 nvsubquadratic [all] (incl. subq_ops from public PyPI)"
pip install --no-cache-dir wheel-stub
pip install --no-cache-dir --no-build-isolation ".[all]" \
    --extra-index-url "${TORCH_INDEX_URL}"

chown -R ubuntu:ubuntu /workspaces
mkdir -p /home/ubuntu
chown -R ubuntu:ubuntu /home/ubuntu
echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /home/ubuntu/.bashrc

step "verification"
python - <<PY
import sys
import torch

# The extensions above were compiled against TORCH_VERSION. If step 10 resolved a
# different torch (pyproject's floor not matching this pin), they are now built
# against headers that do not match the installed torch — which shows up later as
# an undefined symbol, or worse, as numbers that are quietly wrong. Fail here.
want = "${TORCH_VERSION}"
if not torch.__version__.startswith(want):
    print(f"FATAL: built extensions against torch {want} but {torch.__version__} is installed.")
    print("       Something in .[all] upgraded torch. Align the TORCH_VERSION pin with")
    print("       pyproject's torch requirement and rebuild.")
    sys.exit(1)
PY

python - <<'PY'
import importlib.metadata as md
def v(p):
    try:
        return md.version(p)
    except Exception:
        return "MISSING"
import torch
print("torch          ", torch.__version__)
print("torch cuda     ", torch.version.cuda)
print("subq_ops       ", v("subquadratic-ops-torch-cu13"))
print("mamba-ssm      ", v("mamba-ssm"))
print("causal-conv1d  ", v("causal-conv1d"))
print("flash-attn-4   ", v("flash-attn-4"))
print("apex           ", v("apex"))
import subquadratic_ops_torch
from subquadratic_ops_torch.fused_fft_conv2d import fused_fft_conv2d
print("fused_fft_conv2d: OK")
import nvsubquadratic
print("nvsubquadratic : OK")
PY
echo
echo "BUILD COMPLETE"
date "+%H:%M:%S"
OUTER

# Inject the build-time variables the heredoc referenced (quoted heredoc above so
# nothing expanded on the login node — the token must not land in the job script).
PREAMBLE="$(cat <<EOF
export MINIFORGE_NAME='${MINIFORGE_NAME}'
export MINIFORGE_VERSION='${MINIFORGE_VERSION}'
export TORCH_VERSION='${TORCH_VERSION}'
export TORCHVISION_VERSION='${TORCHVISION_VERSION}'
export TORCH_INDEX_URL='${TORCH_INDEX_URL}'
export DALI_PACKAGE='${DALI_PACKAGE}'
export TORCH_CUDA_ARCH_LIST='${CUDA_ARCHS}'
export MAX_JOBS='${MAX_JOBS}'
export NVCC_THREADS='${NVCC_THREADS}'
export INSTALL_MAMBA='${INSTALL_MAMBA}'
export INSTALL_FA4='${INSTALL_FA4}'
export FLASH_ATTN4_VERSION='${FLASH_ATTN4_VERSION}'
export CUTLASS_DSL_VERSION='${CUTLASS_DSL_VERSION}'
EOF
)"

srun \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes=1 --ntasks=1 --exclusive \
    --time="${TIME_LIMIT}" \
    --job-name=nvsubq-imgbuild \
    --output="${LOG_DIR}/imgbuild-%j.out" \
    --error="${LOG_DIR}/imgbuild-%j.out" \
    --container-image="${BASE_IMAGE}" \
    --container-remap-root \
    --container-mounts="${REPO_ROOT}:/mnt/src:ro" \
    --container-save="${OUTPUT_SQSH}" \
    bash -c "${PREAMBLE}
${BUILD}"

echo
echo "Done: ${OUTPUT_SQSH}"
ls -lh "${OUTPUT_SQSH}"
