#!/bin/bash
#
# Build script:
#   1) Build base Docker image from the repo Dockerfile
#   2) Extend it via Dockerfile.slurm (adds DALI, Apex, QuACK)
#   3) Convert the final image to an enroot .sqsh
#
# Usage:
#   ./build_sqsh.sh
#
# Optional env vars:
#   PLATFORM     x86_64 (default, H100) | arm64 (GB200)
#   DOCKER_TAG   base image tag       (default: nvsubquadratic:latest)
#   SLURM_TAG    slurm image tag      (default: nvsubquadratic-slurm:latest)
#   OUTPUT_SQSH  output sqsh filename (default: nvsubquadratic-slurm-<platform>.sqsh)

set -euo pipefail

PLATFORM="${PLATFORM:-x86_64}"

case "${PLATFORM}" in
    x86_64)
        DOCKER_PLATFORM="linux/amd64"
        PLATFORM_SUFFIX="x86_64"
        TARGET_HW="H100 (x86-64)"
        ;;
    arm64)
        DOCKER_PLATFORM="linux/arm64"
        PLATFORM_SUFFIX="arm64"
        TARGET_HW="GB200 (ARM64)"
        ;;
    *)
        echo "Error: unknown PLATFORM=${PLATFORM}. Use x86_64 or arm64."
        exit 1
        ;;
esac

DOCKER_TAG="${DOCKER_TAG:-nvsubquadratic:latest}"
SLURM_TAG="${SLURM_TAG:-nvsubquadratic-slurm:${PLATFORM_SUFFIX}}"
OUTPUT_SQSH="${OUTPUT_SQSH:-nvsubquadratic-slurm-${PLATFORM_SUFFIX}.sqsh}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=============================================="
echo "Platform:     ${DOCKER_PLATFORM} (${TARGET_HW})"
echo "Base image:   ${DOCKER_TAG}"
echo "Slurm image:  ${SLURM_TAG}"
echo "Output sqsh:  ${OUTPUT_SQSH}"
echo "=============================================="

echo ""
echo "==> Building base image: ${DOCKER_TAG}"
docker buildx build \
    --platform "${DOCKER_PLATFORM}" \
    -t "${DOCKER_TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    --load \
    "${REPO_ROOT}"

echo ""
echo "==> Extending with DALI, Apex, QuACK: ${SLURM_TAG}"
docker buildx build \
    --platform "${DOCKER_PLATFORM}" \
    --build-arg BASE_IMAGE="${DOCKER_TAG}" \
    -t "${SLURM_TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile.slurm" \
    --load \
    "${SCRIPT_DIR}"

echo ""
echo "==> Converting to enroot sqsh: ${OUTPUT_SQSH}"
enroot import -o "${OUTPUT_SQSH}" "dockerd://${SLURM_TAG}"

echo ""
echo "=============================================="
echo "Done!"
echo "  Docker image: ${SLURM_TAG}"
echo "  Sqsh file:    ${OUTPUT_SQSH}"
echo "  Platform:     ${DOCKER_PLATFORM} (${TARGET_HW})"
echo ""
echo "To build for the other platform:"
echo "  PLATFORM=arm64  ./build_sqsh.sh   # GB200"
echo "  PLATFORM=x86_64 ./build_sqsh.sh   # H100 (default)"
echo "=============================================="
