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
#   GITLAB_TOKEN   NVIDIA GitLab personal access token (scope: read_api), used to
#                  reach subquadratic-ops-torch-cu13 0.2.2. REQUIRED until 0.2.2 is
#                  published to public PyPI: pyproject's [cuda] extra pins >=0.2.2
#                  (for fused_fft_conv2d / fft_backend="subq_ops_fused") and public
#                  PyPI only has 0.2.1, so `.[all]` fails without it. Read from the
#                  environment or from ~/.gitlab_token. Passed as a BuildKit secret,
#                  never a --build-arg, so it stays out of the image history.
#   SUBQ_OPS_INDEX_URL  Full tokenised index URL, if you'd rather set it directly
#                  than have it built from GITLAB_TOKEN.

set -euo pipefail

PLATFORM="${PLATFORM:-x86_64}"

case "${PLATFORM}" in
    x86_64) DOCKER_PLATFORM="linux/amd64"; TARGET_HW="H100 (x86-64)"; CUDA_ARCHS_DEFAULT="9.0"; MAX_JOBS_DEFAULT=""; NVCC_THREADS_DEFAULT="4" ;;
    arm64)  DOCKER_PLATFORM="linux/arm64"; TARGET_HW="GB200 (ARM64)"; CUDA_ARCHS_DEFAULT="10.0;12.0"; MAX_JOBS_DEFAULT="1"; NVCC_THREADS_DEFAULT="1" ;;
    *)      echo "Error: unknown PLATFORM=${PLATFORM}. Use x86_64 or arm64."; exit 1 ;;
esac

# Override with CUDA_ARCHS=... (e.g. CUDA_ARCHS=10.0 to ease QEMU builds).
CUDA_ARCHS="${CUDA_ARCHS:-${CUDA_ARCHS_DEFAULT}}"
MAX_JOBS="${MAX_JOBS:-${MAX_JOBS_DEFAULT}}"
NVCC_THREADS="${NVCC_THREADS:-${NVCC_THREADS_DEFAULT}}"

if [[ "${PLATFORM}" == "arm64" && "$(uname -m)" != "aarch64" ]]; then
    echo "Warning: building linux/arm64 on $(uname -m) uses QEMU emulation."
    echo "         Apex/mamba CUDA compiles are slow and may ICE/OOM (gcc segfault)."
    echo "         Defaults: MAX_JOBS=1 NVCC_THREADS=1; mamba gencodes narrowed to ${CUDA_ARCHS}."
    echo "         Keep plenty of free host RAM+swap (64GB+ combined recommended)."

    # An OUTDATED qemu-user emulator mis-emulates the arm64 nvcc and crashes it with
    # SIGSEGV (even on `nvcc -V`), failing the apex/mamba builds. Register a current
    # qemu-aarch64 via tonistiigi/binfmt so cross-compilation works. One-time per host
    # boot; skip with SKIP_BINFMT_SETUP=1 if your host QEMU is already current.
    if [[ "${SKIP_BINFMT_SETUP:-0}" != "1" ]]; then
        echo "Refreshing QEMU arm64 emulation (tonistiigi/binfmt --install arm64)..."
        docker run --privileged --rm tonistiigi/binfmt --install arm64
    fi
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

# ── Internal package index for subquadratic-ops-torch-cu13 >= 0.2.2 ──────────
# pyproject's [cuda] extra pins >=0.2.2 (fused_fft_conv2d), which public PyPI does
# not carry, so the `.[all]` layer fails without this. Build the index URL from
# GITLAB_TOKEN (env, else ~/.gitlab_token) and hand it to BuildKit as a secret so
# the token never lands in the image history.
SUBQ_OPS_INDEX_URL="${SUBQ_OPS_INDEX_URL:-}"
if [[ -z "${SUBQ_OPS_INDEX_URL}" ]]; then
    if [[ -z "${GITLAB_TOKEN:-}" && -r "${HOME}/.gitlab_token" ]]; then
        # shellcheck disable=SC1090,SC2046
        eval "$(grep -E '^[[:space:]]*(export[[:space:]]+)?GITLAB_TOKEN=' "${HOME}/.gitlab_token" | tail -1)"
    fi
    if [[ -n "${GITLAB_TOKEN:-}" ]]; then
        SUBQ_OPS_INDEX_URL="https://__token__:${GITLAB_TOKEN}@gitlab-master.nvidia.com/api/v4/projects/180496/packages/pypi/simple"
    fi
fi
if [[ -z "${SUBQ_OPS_INDEX_URL}" ]]; then
    echo "Error: no GITLAB_TOKEN / SUBQ_OPS_INDEX_URL."
    echo "       pyproject pins subquadratic-ops-torch-cu13>=0.2.2, which is only on the"
    echo "       internal GitLab registry, so the '.[all]' layer will fail without it."
    echo "       Create a token (scope: read_api) at"
    echo "         https://gitlab-master.nvidia.com/-/user_settings/personal_access_tokens"
    echo "       then:  echo 'export GITLAB_TOKEN=glpat-...' > ~/.gitlab_token && chmod 600 ~/.gitlab_token"
    exit 1
fi
export SUBQ_OPS_INDEX_URL

echo "Platform: ${DOCKER_PLATFORM} (${TARGET_HW})"
echo "Image:    ${DOCKER_TAG}"
echo "Output:   ${OUTPUT_SQSH}"
echo "Arches:   ${CUDA_ARCHS}  MAX_JOBS=${MAX_JOBS:-unset}  NVCC_THREADS=${NVCC_THREADS}"
echo "Baselines: mamba=${INSTALL_MAMBA}  fa4=${INSTALL_FA4}"
echo "subq_ops index: gitlab-master (token supplied via BuildKit secret)"

DOCKER_BUILDKIT=1 docker buildx build \
    --platform "${DOCKER_PLATFORM}" \
    --build-arg TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" \
    --build-arg MAX_JOBS="${MAX_JOBS}" \
    --build-arg NVCC_THREADS="${NVCC_THREADS}" \
    --build-arg INSTALL_MAMBA="${INSTALL_MAMBA}" \
    --build-arg INSTALL_FA4="${INSTALL_FA4}" \
    --secret id=subq_index,env=SUBQ_OPS_INDEX_URL \
    -t "${DOCKER_TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    --load \
    "${REPO_ROOT}"

enroot import -o "${OUTPUT_SQSH}" "dockerd://${DOCKER_TAG}"

echo "Done: ${OUTPUT_SQSH}"
echo "  PLATFORM=arm64  ./build_sqsh.sh                        # GB200, lean (no baselines)"
echo "  PLATFORM=x86_64 ./build_sqsh.sh                        # H100 (default), lean"
echo "  INSTALL_BASELINES=true PLATFORM=arm64 ./build_sqsh.sh  # + Mamba2 & FA4 (benchmark image)"
