# slurm/enroot — Container Image Build

Builds the top-level [`Dockerfile`](../../../Dockerfile) and converts the result to an enroot `.sqsh` for use with `srun --container-image=...` / `pyxis` on SLURM clusters.

## Build

```bash
bash build_sqsh.sh                  # H100 (x86-64, default)
PLATFORM=arm64 bash build_sqsh.sh   # GB200 (ARM64, built via qemu emulation)
```

The script selects per-platform `--build-arg` values:

| `PLATFORM` | `TORCH_CUDA_ARCH_LIST` | `MAX_JOBS` | `NVCC_THREADS` | Target HW           |
| ---------- | ---------------------- | ---------- | -------------- | ------------------- |
| `x86_64`   | `9.0`                  | unset      | `4`            | H100                |
| `arm64`    | `10.0;12.0`            | `1`        | `1`            | GB200 (B200 / 5090) |

`MAX_JOBS=1` / `NVCC_THREADS=1` on arm64 serializes work to reduce OOM/gcc-ICE failures under QEMU. Upstream `mamba-ssm` / `causal-conv1d` ignore `TORCH_CUDA_ARCH_LIST`; the Dockerfile patches their `setup.py` so only the arches above are compiled. On x86_64, `MAX_JOBS` stays unset (parallel) for fastest builds.

**ARM64 on x86 hosts:** `PLATFORM=arm64` cross-builds via QEMU. Apex/mamba compilation can take many hours and may fail with `gcc: internal compiler error: Segmentation fault` if the host is memory-constrained. Keep ≥64GB combined free RAM+swap (`free -h`); Docker Engine on Linux uses host memory (no separate VM slider). Prefer a native `aarch64` host when possible.

## Override

| Env var              | Default                           |
| -------------------- | --------------------------------- |
| `PLATFORM`           | `x86_64`                          |
| `DOCKER_TAG`         | `nvsubquadratic:${PLATFORM}`      |
| `OUTPUT_SQSH`        | `nvsubquadratic-${PLATFORM}.sqsh` |
| `MAX_JOBS`           | platform default (see table)      |
| `NVCC_THREADS`       | platform default (see table)      |
| `INSTALL_BASELINES`  | `false` (turns on mamba + fa4)    |
| `GITLAB_TOKEN`       | read from `~/.gitlab_token`       |
| `SUBQ_OPS_INDEX_URL` | derived from `GITLAB_TOKEN`       |

## Required: `GITLAB_TOKEN`

`pyproject.toml`'s `[cuda]` extra pins `subquadratic-ops-torch-cu13>=0.2.2` (for
`fused_fft_conv2d` / `fft_backend="subq_ops_fused"`). Public PyPI only carries
0.2.1, so the `.[all]` layer **fails without an internal index**. Create a token
with scope `read_api` at
<https://gitlab-master.nvidia.com/-/user_settings/personal_access_tokens>, then:

```bash
echo 'export GITLAB_TOKEN=glpat-...' > ~/.gitlab_token && chmod 600 ~/.gitlab_token
```

`build_sqsh.sh` picks it up and passes it to BuildKit as a `--secret`, so it is
never written into the image's layer history. Remove this step once 0.2.2 is
published to public PyPI.
