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

**ARM64 on x86 hosts:** `PLATFORM=arm64` cross-builds via QEMU. Apex/mamba compilation can take many hours and may fail with `gcc: internal compiler error: Segmentation fault` if the host is memory-constrained. Keep ≥64GB combined free RAM+swap (`free -h`); Docker Engine on Linux uses host memory (no separate VM slider). Prefer a native `aarch64` host when possible — or skip Docker entirely with [`build_sqsh_slurm.sh`](#building-on-a-slurm-node-no-docker).

## Building on a SLURM node (no Docker)

[`build_sqsh_slurm.sh`](build_sqsh_slurm.sh) produces the same `.sqsh` **without Docker**, by replaying the Dockerfile's `RUN` steps inside a pyxis/enroot container on a compute node:

```bash
scripts/slurm/enroot/build_sqsh_slurm.sh
```

Use it when `build_sqsh.sh` is not an option — cluster login nodes typically have no Docker daemon, no registered `qemu-aarch64`, and no root to install either, which otherwise leaves no way to produce a GB200 image at all. It pulls the CUDA base with enroot, takes root via `--container-remap-root`, and writes the image out with `--container-save`.

Because it builds **natively on aarch64**, QEMU is out of the picture: the apex/mamba compiles that `build_sqsh.sh` throttles to `MAX_JOBS=1` to survive emulation run fully parallel here (default `MAX_JOBS=32`).

It replays the Dockerfile rather than parsing it, so its torch/CUDA pins are a copy. `scripts/check_version_pins.py` fails pre-commit and CI if they drift from `pyproject.toml`.

| Env var                         | Default                                    |
| ------------------------------- | ------------------------------------------ |
| `OUTPUT_SQSH`                   | `$LUSTRE_ENROOT/nvsubquadratic-arm64.sqsh` |
| `PARTITION` / `ACCOUNT`         | cluster defaults (override for your site)  |
| `CUDA_ARCHS`                    | `10.0;12.0` (GB200 + B200/5090)            |
| `MAX_JOBS` / `NVCC_THREADS`     | `32` / `4`                                 |
| `INSTALL_MAMBA` / `INSTALL_FA4` | `true` (benchmark image)                   |

## Override

| Env var             | Default                           |
| ------------------- | --------------------------------- |
| `PLATFORM`          | `x86_64`                          |
| `DOCKER_TAG`        | `nvsubquadratic:${PLATFORM}`      |
| `OUTPUT_SQSH`       | `nvsubquadratic-${PLATFORM}.sqsh` |
| `MAX_JOBS`          | platform default (see table)      |
| `NVCC_THREADS`      | platform default (see table)      |
| `INSTALL_BASELINES` | `false` (turns on mamba + fa4)    |
