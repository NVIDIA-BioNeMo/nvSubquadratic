# slurm/enroot — Container Image Build

Builds the top-level [`Dockerfile`](../../Dockerfile) and converts the result to an enroot `.sqsh` for use with `srun --container-image=...` / `pyxis` on SLURM clusters.

## Build

```bash
bash build_sqsh.sh                  # H100 (x86-64, default)
PLATFORM=arm64 bash build_sqsh.sh   # GB200 (ARM64, built via qemu emulation)
```

The script selects per-platform `--build-arg` values:

| `PLATFORM` | `TORCH_CUDA_ARCH_LIST` | `MAX_JOBS` | Target HW           |
| ---------- | ---------------------- | ---------- | ------------------- |
| `x86_64`   | `9.0`                  | unset      | H100                |
| `arm64`    | `10.0;12.0`            | `2`        | GB200 (B200 / 5090) |

`MAX_JOBS=2` on arm64 caps parallel nvcc jobs to avoid OOM under qemu emulation. On x86_64 it stays unset (parallel) for fastest builds.

## Override

| Env var       | Default                           |
| ------------- | --------------------------------- |
| `PLATFORM`    | `x86_64`                          |
| `DOCKER_TAG`  | `nvsubquadratic:${PLATFORM}`      |
| `OUTPUT_SQSH` | `nvsubquadratic-${PLATFORM}.sqsh` |
