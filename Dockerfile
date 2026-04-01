# Development Dockerfile for nvSubquadratic
#
# Build instructions:
#   docker build -t nvsubquadratic:dev .

FROM nvcr.io/nvidia/cuda:12.9.0-devel-ubuntu22.04

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
    torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu129 \
    && conda clean --all --yes

# Create ubuntu user with sudo privileges
RUN apt-get update && apt-get install -y sudo && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r ubuntu && \
    useradd -r -g ubuntu -G sudo -m -s /bin/bash ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory
WORKDIR /workspaces/nvSubquadratic-private

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project for development (as root first for package installation)
COPY . .

# Set up git safe directory
RUN git config --global --add safe.directory /workspaces/nvSubquadratic-private

# Install development dependencies first (as root, system-wide)
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install the package with quack-kernels (as root, system-wide).
# extra-index-url ensures the resolver picks cu129 wheels that match this image
# and does not replace them with a CPU or different-CUDA build from PyPI.
RUN pip install --no-cache-dir --no-build-isolation ".[quack]" \
    --extra-index-url https://download.pytorch.org/whl/cu129

# Build and install NVIDIA Apex with C++ and CUDA extensions.
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    git+https://github.com/NVIDIA/apex.git

# Set up ubuntu user's home directory and permissions
RUN chown -R ubuntu:ubuntu /workspaces && \
    mkdir -p /home/ubuntu && \
    chown -R ubuntu:ubuntu /home/ubuntu && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /home/ubuntu/.bashrc

# Switch to ubuntu user
USER ubuntu

# Set environment variables for development mode
ENV PYTHONPATH=/workspaces/nvSubquadratic-private

# Expose Jupyter port
EXPOSE 8888

# Development command
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
