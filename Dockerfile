# Development Dockerfile for nvSubquadratic
#
# Build instructions:
#   docker build -t nvsubquadratic:dev .

FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=25.3.0-3

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
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
        torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
        && conda clean --all --yes

# Re-declare ARG after FROM to make it available in build stage
ARG GITLAB_TOKEN

# Create ubuntu user with sudo privileges
RUN apt-get update && apt-get install -y sudo && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r ubuntu && \
    useradd -r -g ubuntu -G sudo -m -s /bin/bash ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

RUN conda install --yes \
        python=3.12 \
        && conda clean --all --yes

RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
        && conda clean --all --yes
RUN pip install --no-cache-dir \
        einops \
        pytorch-lightning \
        wandb \
        huggingface_hub \
        datasets \
        Pillow \
        "pyarrow>=14.0.0,<20.0.0" \
        "diffusers>=0.25.0" \
        "clean-fid>=0.1.35" \
        "megatron-core" \
        "omegaconf>=2.3.0" \
        "rich>=13.0.0"

# ARG PYTORCH_VERSION=25.06

# # Base image with PyTorch
# FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

# # Re-declare ARG after FROM to make it available in build stage
# ARG GITLAB_TOKEN

# # Set working directory
WORKDIR /workspaces/nvSubquadratic-private

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  && rm -rf /var/lib/apt/lists/*

# Copy the entire project for development (as root first for package installation)
COPY . .

# Install development dependencies first (as root, system-wide)
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install the package (as root, system-wide)
RUN pip install --no-cache-dir .

# Install subquadratic_ops wheel file (as root, system-wide)
# pip will automatically select the correct architecture (x86_64 / arm64)
# GITLAB_TOKEN is required for this installation
RUN if [ -n "${GITLAB_TOKEN}" ]; then echo "Installing subquadratic-ops with token..." && pip install subquadratic-ops==v0.0.1+cuda12.9 --index-url https://__token__:${GITLAB_TOKEN}@gitlab-master.nvidia.com/api/v4/projects/180496/packages/pypi/simple; else echo "Skipping subquadratic-ops installation because GITLAB_TOKEN is not available. Please set GITLAB_TOKEN environment variable."; fi

# Set up ubuntu user's home directory and permissions
RUN chown -R ubuntu:ubuntu /workspaces && \
    mkdir -p /home/ubuntu && \
    chown -R ubuntu:ubuntu /home/ubuntu && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /home/ubuntu/.bashrc

# Switch to ubuntu user
USER ubuntu

# Set environment variables for development mode
ENV PYTHONPATH="/workspaces/nvSubquadratic-private:${PYTHONPATH}"

# Expose Jupyter port
# EXPOSE 8888

# Development command
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]