# Development Dockerfile for nvSubquadratic
#
# Build instructions:
#   docker build -t nvsubquadratic:dev .

ARG PYTORCH_VERSION=25.06
ARG GITLAB_TOKEN

# Base image with PyTorch
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  git \
  && rm -rf /var/lib/apt/lists/*

# Use the existing ubuntu user and give it sudo privileges
ARG USERNAME=ubuntu
RUN mkdir -p /etc/sudoers.d \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

# Set environment variables
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Copy requirements first for better caching
COPY pyproject.toml README.md requirements-dev.txt ./

# Copy the package source
COPY nvsubquadratic/ ./nvsubquadratic/

# Install the package
RUN pip install --no-cache-dir . && \
  pip cache purge

# Install subquadratic_ops wheel file - pip will automatically select the correct architecture (x86_64 / arm64)
# GITLAB_TOKEN is required for this installation
RUN if [ -n "${GITLAB_TOKEN}" ]; then echo "Installing subquadratic-ops with token..." && pip install subquadratic-ops==v0.0.1+cuda12.9 --index-url https://__token__:${GITLAB_TOKEN}@gitlab-master.nvidia.com/api/v4/projects/180496/packages/pypi/simple; else echo "Skipping subquadratic-ops installation because GITLAB_TOKEN is not available. Please set GITLAB_TOKEN environment variable."; fi

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code
COPY . .

# Install in development mode
RUN pip install -e .

# Install pre-commit hooks (both commit and pre-push)
RUN pre-commit install && pre-commit install --hook-type pre-push

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
