#!/bin/bash
#
# Minimal build script (similar to the provided example):
#   1) Build Docker image from the repo Dockerfile
#   2) Convert to enroot sqsh
#
# Usage:
#   ./build_sqsh.sh
#
# Optional env vars:
#   DOCKER_TAG   (default: nvsubquadratic:latest)
#   OUTPUT_SQSH  (default: nvsubquadratic.sqsh)

set -euo pipefail

DOCKER_TAG="${DOCKER_TAG:-nvsubquadratic:latest}"
OUTPUT_SQSH="${OUTPUT_SQSH:-nvsubquadratic.sqsh}"


echo "Building Docker image: ${DOCKER_TAG}"
docker buildx build \
  -t "${DOCKER_TAG}" \
  -f ../../Dockerfile ../..

echo "Importing Docker image to sqsh: ${OUTPUT_SQSH}"
enroot import -o "${OUTPUT_SQSH}" "dockerd://${DOCKER_TAG}"

echo "Done. Image: ${DOCKER_TAG}"
