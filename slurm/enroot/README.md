# slurm/enroot — Container Image Build

Builds the enroot `.sqsh` used for cluster training on top of the base `Dockerfile`.

The base image omits three packages to keep CI lean (no GPU required at build time):

- **`nvidia-dali-cuda120`** — fused GPU decode/augment for the ImageNet data pipeline
- **NVIDIA Apex** — `FusedLAMB` optimizer, must be compiled from source against CUDA
- **`quack-kernels`** — fused Triton RMSNorm (optional; falls back to pure PyTorch)

`Dockerfile.slurm` extends `nvsubquadratic:latest` with these, and `build_sqsh.sh`
runs both build stages and converts the result to a `.sqsh` file.

## Build

```bash
bash slurm/enroot/build_sqsh.sh           # H100 (x86-64, default)
PLATFORM=arm64 bash slurm/enroot/build_sqsh.sh  # GB200 (ARM64)
```

Override `DOCKER_TAG`, `SLURM_TAG`, or `OUTPUT_SQSH` as needed.
