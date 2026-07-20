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

set -euo pipefail

PLATFORM="${PLATFORM:-x86_64}"

case "${PLATFORM}" in
    x86_64) DOCKER_PLATFORM="linux/amd64"; TARGET_HW="H100 (x86-64)"; CUDA_ARCHS="9.0"; MAX_JOBS_DEFAULT="" ;;
    arm64)  DOCKER_PLATFORM="linux/arm64"; TARGET_HW="GB200 (ARM64)"; CUDA_ARCHS="10.0;12.0"; MAX_JOBS_DEFAULT="1" ;;
    *)      echo "Error: unknown PLATFORM=${PLATFORM}. Use x86_64 or arm64."; exit 1 ;;
esac

MAX_JOBS="${MAX_JOBS:-${MAX_JOBS_DEFAULT}}"

if [[ "${PLATFORM}" == "arm64" && "$(uname -m)" != "aarch64" ]]; then
    echo "Warning: building linux/arm64 on $(uname -m) uses QEMU emulation."
    echo "         Apex CUDA compilation is slow and may ICE/OOM (gcc segfault)."
    echo "         Prefer building natively on GB200, or set MAX_JOBS=1 (default)."
    echo "         Ensure Docker has plenty of RAM (64GB+ recommended)."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DOCKER_TAG="${DOCKER_TAG:-nvsubquadratic:${PLATFORM}}"
OUTPUT_SQSH="${OUTPUT_SQSH:-${SCRIPT_DIR}/nvsubquadratic-${PLATFORM}.sqsh}"

echo "Platform: ${DOCKER_PLATFORM} (${TARGET_HW})"
echo "Image:    ${DOCKER_TAG}"
echo "Output:   ${OUTPUT_SQSH}"

docker buildx build \
    --platform "${DOCKER_PLATFORM}" \
    --build-arg TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    -t "${DOCKER_TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    --load \
    "${REPO_ROOT}"

enroot import -o "${OUTPUT_SQSH}" "dockerd://${DOCKER_TAG}"

echo "Done: ${OUTPUT_SQSH}"
echo "  PLATFORM=arm64  ./build_sqsh.sh   # GB200"
echo "  PLATFORM=x86_64 ./build_sqsh.sh   # H100 (default)"
