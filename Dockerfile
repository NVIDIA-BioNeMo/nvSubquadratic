# Development Dockerfile for nvSubquadratic (CUDA 13.2 / PyTorch cu132)
#
# Build instructions:
#   DOCKER_BUILDKIT=1 docker build -t nvsubquadratic:dev .
#
# Requires an NVIDIA driver >= 580 on the host (CUDA 13.x minimum), or the
# forward-compatibility package. Volta (sm_70) GPUs are no longer supported.
#
# Layer order is intentional for CI cache efficiency:
#   1. Base image + conda + torch + DALI  (never changes)
#   2. Apex build                         (changes only if apex version bumped)
#   3. requirements-dev.txt               (changes when dev deps change)
#   4. COPY . . + pip install             (changes on every code push — fast)

FROM nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu22.04

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=25.3.0-3

# ── CUDA 13.2 toolchain pins ─────────────────────────────────────────────────
# CUDA 13 wheels live under the cu132 index. The first PyTorch release built
# against CUDA 13.2 is 2.12.0, so the previous 2.10.0 pin cannot be kept here.
ARG TORCH_VERSION=2.12.1
ARG TORCHVISION_VERSION=0.27.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
# DALI ships one build per CUDA major version; cuda130 is the CUDA 13.x build.
ARG DALI_PACKAGE=nvidia-dali-cuda130

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

# The torch install is repeated after DALI on purpose: DALI pulls its own
# nvidia-* CUDA runtime wheels and can shuffle the ones torch depends on, so we
# re-assert the pinned cu132 build afterwards.
RUN pip install --no-cache-dir \
    torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir ${DALI_PACKAGE} \
    && pip install --no-cache-dir \
       torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX_URL} \
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
# CUDA 13 dropped offline compilation for Maxwell/Pascal/Volta — `nvcc
# --list-gpu-arch` in 13.2 starts at compute_75, so 7.0 must be removed or the
# apex build fails with "Unsupported gpu architecture 'compute_70'".
# Newly available in 13.2 if you need them: 8.7, 8.8, 10.3 (B300), 11.0, 12.1.
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG MAX_JOBS=""
RUN MAX_JOBS="${MAX_JOBS}" pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git

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
       --extra-index-url ${TORCH_INDEX_URL}

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
