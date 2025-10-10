# Development Dockerfile for nvSubquadratic
#
# Build instructions:
#   docker build -t nvsubquadratic:dev .

ARG PYTORCH_VERSION=25.06
ARG GITLAB_TOKEN

# Base image with PyTorch
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

# Set working directory
WORKDIR /workspaces/nvSubquadratic-private

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

# Change ownership of workspaces directory to ubuntu user
RUN chown -R $USERNAME:$USERNAME /workspaces

USER $USERNAME

# Set environment variables for development mode
ENV PYTHONPATH="/workspaces/nvSubquadratic-private:${PYTHONPATH}"

# Copy the entire project for development
COPY --chown=$USERNAME:$USERNAME . .

# Install development dependencies first
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install the package in development mode (editable install)
RUN pip install --no-cache-dir -e .

# Install subquadratic_ops wheel file - pip will automatically select the correct architecture (x86_64 / arm64)
# GITLAB_TOKEN is required for this installation
RUN if [ -n "${GITLAB_TOKEN}" ]; then echo "Installing subquadratic-ops with token..." && pip install subquadratic-ops==v0.0.1+cuda12.9 --index-url https://__token__:${GITLAB_TOKEN}@gitlab-master.nvidia.com/api/v4/projects/180496/packages/pypi/simple; else echo "Skipping subquadratic-ops installation because GITLAB_TOKEN is not available. Please set GITLAB_TOKEN environment variable."; fi

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
