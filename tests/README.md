# Tests

## Directory structure

```
tests/
├── conftest.py                          # Shared fixtures (device, dtype_fixture)
├── ops/                                 # Tests for nvsubquadratic/ops/
│   ├── test_fftconv.py                  # FFT convolution benchmarks (1D/2D/3D)
│   ├── test_fftconv_chunked.py          # Chunked FFT convolution correctness
│   ├── test_fftconv_fp16.py             # FP16 FFT convolution suite
│   ├── test_circular_fftconv.py         # Circular FFT convolution vs reference
│   ├── test_circular_vs_fftconv_perf.py # Circular vs standard FFT perf comparison
│   └── test_subq_ops_fft_conv2d.py      # subquadratic_ops_torch CUDA FFT conv2d
├── modules/                             # Tests for nvsubquadratic/modules/
│   ├── test_causality_hyena.py          # Hyena causality checks
│   ├── test_causality_attn_mamba.py     # Attention/Mamba causality checks
│   ├── test_self_attention.py           # Self-attention module tests
│   ├── test_hyena_film.py               # FiLM kernels, RegisterPooling, HyenaAdapter
│   ├── test_patchify.py                 # Patchify/unpatchify
│   ├── test_schedulers.py               # LR schedulers
│   ├── test_distributed_depthwise_conv_nd.py  # Distributed depthwise conv
│   ├── test_torch_compile.py            # torch.compile compatibility
│   ├── test_vit5_components.py          # ViT-5 components (RMSNorm, LayerScale, etc.)
│   └── torchrun_sequence_mixer_cp_test.py  # Multi-GPU sequence mixer test
├── networks/                            # Tests for nvsubquadratic/networks/
│   ├── test_diffusion_wrapper.py        # DiffusionWrapper tests
│   ├── test_diffusion_fid.py            # Diffusion FID evaluation
│   └── test_hf_diffusers_wrapper.py     # HuggingFace diffusers wrapper
├── test_basic.py                        # Basic import/sanity checks
├── test_autoregressive_wrapper.py       # Autoregressive wrapper
├── test_checkpoint_resume.py            # Checkpoint save/resume
├── test_classification_loss.py          # Classification loss
├── test_dali_rand_augment.py            # DALI RandAugment pipeline
├── test_gpu_jpeg_decode.py              # GPU JPEG decoding
├── test_image_grid_callback.py          # Image grid callback
├── test_nightly_validation.py           # Nightly: checkpoint validation (FiLM, Attn, GAP)
└── test_pixel_scaling.py                # Pixel scaling
```

## Test suites

| Suite       | Marker      | What it tests                                                         | Requirements                 |
| ----------- | ----------- | --------------------------------------------------------------------- | ---------------------------- |
| **Unit**    | *(default)* | ViT-5 components, FiLM/Hyena modules, token layout, gradient flow     | CPU (GPU optional)           |
| **Nightly** | `nightly`   | Full ImageNet validation of best FiLM, Attention, and GAP checkpoints | GPU, DALI, ImageNet, W&B key |

## Running tests

All commands assume you are in the project root.

### Unit tests

```bash
# All unit tests (excludes nightly)
PYTHONPATH=. python -m pytest tests/ -m "not nightly" -v -o addopts=""

# A single subdirectory
PYTHONPATH=. python -m pytest tests/ops/ -v -o addopts=""

# A single file
PYTHONPATH=. python -m pytest tests/modules/test_vit5_components.py -v -o addopts=""

# A single test class or method
PYTHONPATH=. python -m pytest tests/modules/test_hyena_film.py::TestRegisterPooling -v -o addopts=""
```

> **Note:** `-o addopts=""` overrides the default `--cov` flags in `pyproject.toml`,
> which require the coverage plugin. Omit it if you have `pytest-cov` installed.

### Nightly validation tests

These download the "best" checkpoint for each model from W&B and run a full
test pass on ImageNet-1k, asserting that accuracy has not regressed.

Prerequisites (all available inside the SLURM container):

- GPU with CUDA
- NVIDIA DALI (`nvidia.dali`)
- ImageNet at `/shared/data/image_datasets/imagenet`
- `WANDB_API_KEY` environment variable (provided by `source .env`)

```bash
# All three models (~2 min)
source .env && PYTHONPATH=. python -m pytest tests/ -m nightly -v -o addopts=""

# A single model
source .env && PYTHONPATH=. python -m pytest tests/test_nightly_validation.py::test_validate_film_model -v -o addopts=""
```

Inside SLURM:

```bash
srun --container-image=/shared/images/nvsubquadratic_cuda129.sqsh \
     --container-mounts="/home/dwromero:/home/dwromero,/shared:/shared,/scratch:/scratch" \
     --container-workdir=/home/dwromero/projects/nvSubquadratic-private \
     bash -c 'source .env && conda activate nv-subq && PYTHONPATH=. python -m pytest tests/ -m nightly -v -o addopts=""'
```

### All tests (unit + nightly)

```bash
source .env && PYTHONPATH=. python -m pytest tests/ -v -o addopts=""
```

## Adding new tests

- Mirror the `nvsubquadratic/` package structure: `tests/ops/`, `tests/modules/`, `tests/networks/`.
- Place unit tests in the matching subdirectory, e.g. `tests/ops/test_<module>.py`.
- Mark slow GPU-only tests with `@pytest.mark.slow`.
- Mark full-dataset validation tests with `@pytest.mark.nightly`.
- New markers must be registered in `pyproject.toml` under `[tool.pytest.ini_options]`.
