# Development Dockerfile for nvSubquadratic
#
# Build instructions:
#   docker build -t nvsubquadratic:dev .
#
# Layer order is intentional for CI cache efficiency:
#   1. Base image + conda + torch + DALI  (never changes)
#   2. Apex build                         (changes only if apex version bumped)
#   3. requirements-dev.txt               (changes when dev deps change)
#   4. COPY . . + pip install             (changes on every code push — fast)

FROM nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu22.04

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=25.3.0-3

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
    wget bzip2 ca-certificates \
    git \
    tini \
    > /dev/null && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc && \
    chmod -R a+rX ${CONDA_DIR}

RUN conda install --yes \
    python=3.12 \
    && conda clean --all --yes

RUN pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu129 \
    && pip install --no-cache-dir nvidia-dali-cuda120 \
    && pip install --no-cache-dir \
       torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu129 \
    && conda clean --all --yes

# Create ubuntu user with sudo privileges
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y sudo && \
    groupadd -r ubuntu && \
    useradd -r -g ubuntu -G sudo -m -s /bin/bash ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install system build dependencies
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    git

WORKDIR /workspaces/nvSubquadratic

# ── Heavy build: Apex from source (cached until apex commit changes) ──────────
# This layer is intentionally placed before COPY so code changes do not
# trigger a rebuild. Apex does not depend on the project source.
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG MAX_JOBS=""
RUN MAX_JOBS="${MAX_JOBS}" pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git

# ── Mamba baseline: mamba-ssm + causal-conv1d from source ─────────────────────
# Optional comparison baseline (e.g. the 2D forward-time benchmark's Mamba2
# series); NOT a project dependency, so it is installed explicitly here rather
# than via an extra. Placed before COPY of the full tree so unrelated code
# changes do not trigger a rebuild. The tiny patch script is copied first so we
# can rewrite upstream setup.py before compiling.
#
# Upstream hardcodes -gencode for sm_75..sm_120 and ignores TORCH_CUDA_ARCH_LIST,
# which OOMs / gcc-ICEs under QEMU arm64. We clone, patch gencodes to match
# TORCH_CUDA_ARCH_LIST, and honor NVCC_THREADS (arm64 build sets this to 1).
#
# mamba-ssm depends on an unpinned `torch`, so a plain install lets pip swap
# torch / nvidia-cudnn-cu12 out from under the 2.10 stack — which breaks cuDNN
# (CUDNN_STATUS_NOT_INITIALIZED at runtime). Freeze the current torch/nvidia/
# triton versions into a constraints file so the mamba install cannot touch them.
ARG NVCC_THREADS=4
COPY scripts/docker/patch_mamba_cuda_arches.py /tmp/patch_mamba_cuda_arches.py
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|nvidia-|triton|pytorch-triton)' > /tmp/mamba-constraints.txt && \
    cat /tmp/mamba-constraints.txt && \
    git clone --depth 1 https://github.com/Dao-AILab/causal-conv1d.git /tmp/causal-conv1d && \
    git clone --depth 1 https://github.com/state-spaces/mamba.git /tmp/mamba && \
    python /tmp/patch_mamba_cuda_arches.py /tmp/causal-conv1d/setup.py /tmp/mamba/setup.py && \
    MAX_JOBS="${MAX_JOBS}" NVCC_THREADS="${NVCC_THREADS}" \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE \
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    -c /tmp/mamba-constraints.txt \
    /tmp/causal-conv1d /tmp/mamba && \
    rm -rf /tmp/causal-conv1d /tmp/mamba

# ── FlashAttention-4 baseline (Blackwell / Hopper) ────────────────────────────
# Optional baseline for the forward-time benchmark's FlashAttention-4 series
# (attn_impl="fa4"); NOT a project dependency. FA4 is a pure-Python CuTe-DSL wheel
# that JIT-compiles its kernel at runtime on the target GPU, so — unlike apex /
# mamba above — there is NO CUDA source build here: this layer is cheap and
# QEMU-safe. It needs a Hopper/Blackwell GPU + CUDA >= 12.3 at *run* time (the
# build host needs no GPU; the JIT fires on the first forward). Alpha release, so
# --pre. This CUDA 12.9 image uses the default (cu12) wheel; on a CUDA 13 base use
# `flash-attn-4[cu13]`. Pin torch/nvidia/triton so the install cannot swap the
# 2.10 stack out from under cuDNN (same failure the mamba layer guards against).
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|nvidia-|triton|pytorch-triton)' > /tmp/fa4-constraints.txt && \
    pip install --no-cache-dir --pre -c /tmp/fa4-constraints.txt flash-attn-4 && \
    python -c "from flash_attn.cute import flash_attn_func; print('flash-attn-4: import OK')" \
    || echo "WARNING: flash-attn-4 import check skipped/failed (JIT probes CUDA at build?); verify on-GPU."

# ── Dev deps: cached until requirements-dev.txt changes ──────────────────────
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# ── Source: invalidated on every code change (fast — just package install) ────
COPY . .

RUN git config --global --add safe.directory /workspaces/nvSubquadratic

# Full-fat dev/CI image: install every optional extra so the whole test suite
# (distributed/Megatron CP tests, timm baselines, DALI, subq_ops CUDA kernels)
# can run. After the 0.1.1 dependency restructure, megatron-core/timm/etc. are
# optional extras ([distributed]/[baselines]/...), so a bare install no longer
# pulls them — [all] restores the complete pre-restructure dependency set.
RUN pip install --no-cache-dir wheel-stub \
    && pip install --no-cache-dir --no-build-isolation ".[all]" \
       --extra-index-url https://download.pytorch.org/whl/cu129

# Set up ubuntu user's home directory and permissions
RUN chown -R ubuntu:ubuntu /workspaces && \
    mkdir -p /home/ubuntu && \
    chown -R ubuntu:ubuntu /home/ubuntu && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /home/ubuntu/.bashrc

# Switch to ubuntu user
USER ubuntu

# Set environment variables for development mode
ENV PYTHONPATH=/workspaces/nvSubquadratic

# Expose Jupyter port
EXPOSE 8888

# Development command
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
