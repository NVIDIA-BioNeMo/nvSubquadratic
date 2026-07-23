#!/bin/bash
#
# Build script: builds the Docker image and converts it to an enroot .sqsh
#
# Usage:
#   ./build_sqsh.sh
#
# Optional env vars:
#   PLATFORM     x86_64 (default, H100) | arm64 (GB200)
#   DOCKER_TAG   image tag    (default: nvsubquadratic:<platform>)
#   OUTPUT_SQSH  output file  (default: nvsubquadratic-<platform>.sqsh)
#   MAX_JOBS     parallel nvcc/ninja jobs (arm64 default: 1)
#   NVCC_THREADS nvcc --threads (arm64 default: 1)
#   INSTALL_BASELINES  turn ON both benchmark baselines below (default: false).
#                  Use for the benchmark image:  INSTALL_BASELINES=true ./build_sqsh.sh
#   INSTALL_MAMBA  build the Mamba2 baseline (default: INSTALL_BASELINES; true = slow
#                  QEMU source build; false → benchmark's Mamba2 series 'unavailable')
#   INSTALL_FA4    build the FlashAttention-4 baseline (default: INSTALL_BASELINES;
#                  the alpha flash-attn-4 wheel)

set -euo pipefail

PLATFORM="${PLATFORM:-x86_64}"

case "${PLATFORM}" in
    x86_64) DOCKER_PLATFORM="linux/amd64"; TARGET_HW="H100 (x86-64)"; CUDA_ARCHS="9.0"; MAX_JOBS_DEFAULT=""; NVCC_THREADS_DEFAULT="4" ;;
    arm64)  DOCKER_PLATFORM="linux/arm64"; TARGET_HW="GB200 (ARM64)"; CUDA_ARCHS="10.0;12.0"; MAX_JOBS_DEFAULT="1"; NVCC_THREADS_DEFAULT="1" ;;
    *)      echo "Error: unknown PLATFORM=${PLATFORM}. Use x86_64 or arm64."; exit 1 ;;
esac

MAX_JOBS="${MAX_JOBS:-${MAX_JOBS_DEFAULT}}"
NVCC_THREADS="${NVCC_THREADS:-${NVCC_THREADS_DEFAULT}}"

if [[ "${PLATFORM}" == "arm64" && "$(uname -m)" != "aarch64" ]]; then
    echo "Warning: building linux/arm64 on $(uname -m) uses QEMU emulation."
    echo "         Apex/mamba CUDA compiles are slow and may ICE/OOM (gcc segfault)."
    echo "         Defaults: MAX_JOBS=1 NVCC_THREADS=1; mamba gencodes narrowed to ${CUDA_ARCHS}."
    echo "         Keep plenty of free host RAM+swap (64GB+ combined recommended)."
fi

# Preflight: free RAM matters more than MAX_JOBS under QEMU (single TUs still spike).
if command -v free >/dev/null 2>&1; then
    avail_gb=$(free -g | awk '/^Mem:/{print $7}')
    swap_gb=$(free -g | awk '/^Swap:/{print $2}')
    if [[ "${PLATFORM}" == "arm64" && "${avail_gb}" =~ ^[0-9]+$ && "${avail_gb}" -lt 24 ]]; then
        echo "Warning: only ~${avail_gb}GiB MemAvailable (swap=${swap_gb}GiB)."
        echo "         Free RAM or add swap before retrying; gcc ICE usually means OOM."
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DOCKER_TAG="${DOCKER_TAG:-nvsubquadratic:${PLATFORM}}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRIPT_DIR}/nvsubquadratic-${PLATFORM}.sqsh}"
# Benchmark-only baselines, OFF by default (leaner/faster image). Turn on BOTH with
# INSTALL_BASELINES=true (the benchmark-image shortcut), or each individually.
INSTALL_BASELINES="${INSTALL_BASELINES:-false}"
INSTALL_MAMBA="${INSTALL_MAMBA:-${INSTALL_BASELINES}}"
INSTALL_FA4="${INSTALL_FA4:-${INSTALL_BASELINES}}"

echo "Platform: ${DOCKER_PLATFORM} (${TARGET_HW})"
echo "Image:    ${DOCKER_TAG}"
echo "Output:   ${OUTPUT_SQSH}"
echo "Arches:   ${CUDA_ARCHS}  MAX_JOBS=${MAX_JOBS:-unset}  NVCC_THREADS=${NVCC_THREADS}"
echo "Baselines: mamba=${INSTALL_MAMBA}  fa4=${INSTALL_FA4}"

docker buildx build \
    --platform "${DOCKER_PLATFORM}" \
    --build-arg TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    --build-arg NVCC_THREADS="${NVCC_THREADS}" \
    --build-arg INSTALL_MAMBA="${INSTALL_MAMBA}" \
    --build-arg INSTALL_FA4="${INSTALL_FA4}" \
    -t "${DOCKER_TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    --load \
    "${REPO_ROOT}"

enroot import -o "${OUTPUT_SQSH}" "dockerd://${DOCKER_TAG}"

echo "Done: ${OUTPUT_SQSH}"
echo "  PLATFORM=arm64  ./build_sqsh.sh                        # GB200, lean (no baselines)"
echo "  PLATFORM=x86_64 ./build_sqsh.sh                        # H100 (default), lean"
echo "  INSTALL_BASELINES=true PLATFORM=arm64 ./build_sqsh.sh  # + Mamba2 & FA4 (benchmark image)"
