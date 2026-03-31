# Slurm/enroot image — extends the base nvsubquadratic image with the packages
# required by examples/vit5_imagenet/v5_patch/README.md but omitted from the
# base Dockerfile (which skips them to keep CI lean):
#
#   - nvidia-dali-cuda120   GPU-accelerated decode + crop + augment for ImageNet
#   - NVIDIA Apex           FusedLAMB optimizer, built from source with CUDA ext
#   - quack-kernels         Fused Triton RMSNorm (optional but recommended)
#
# Build (handled by build_sqsh.sh, but can also be done manually):
#   docker build \
#       --build-arg BASE_IMAGE=nvsubquadratic:latest \
#       -t nvsubquadratic-slurm:latest \
#       -f slurm/enroot/Dockerfile.slurm .

ARG BASE_IMAGE=nvsubquadratic:latest
FROM ${BASE_IMAGE}

USER root

# NVIDIA DALI
RUN pip install --no-cache-dir nvidia-dali-cuda120

# QuACK kernels (fused Triton RMSNorm; pure-PyTorch fallback if absent)
RUN pip install --no-cache-dir quack-kernels

# DALI / quack-kernels can upgrade torch from PyPI (e.g. 2.11+cu130). Apex requires
# torch.version.cuda to match /usr/local/cuda (12.8 here); force cu128 wheels aligned
# with pyproject.toml torch/torchvision bounds.
RUN pip install --no-cache-dir --upgrade \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.10.0,<2.11.0" \
    "torchvision>=0.25.0,<0.26.0"

# Apex (FusedLAMB) — must be built from source with CUDA extensions.
# Not checked into the repo, so we clone it here.
RUN git clone --depth 1 https://github.com/NVIDIA/apex.git /tmp/apex && \
    cd /tmp/apex && \
    APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation . && \
    rm -rf /tmp/apex

USER ubuntu
