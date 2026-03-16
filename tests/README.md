# Tests

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

# A single file
PYTHONPATH=. python -m pytest tests/test_vit5_components.py -v -o addopts=""

# A single test class or method
PYTHONPATH=. python -m pytest tests/test_hyena_film.py::TestRegisterPooling -v -o addopts=""
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

## Test files

| File                         | Description                                                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `test_vit5_components.py`    | RMSNorm, LayerScale, DropPath, ViT5Attention, ViT5ResidualBlock, ViT5ClassificationNet, cross-validation, GAP readout |
| `test_hyena_film.py`         | RegisterPooling, KernelFiLMGenerator, ViT5ResidualBlock with FiLM, ViT5HyenaAdapter                                   |
| `test_nightly_validation.py` | Nightly: FiLM (peeaqdkq), Attention (44or24g1), GAP (tcji9tfx) checkpoint validation                                  |
| `conftest.py`                | Shared fixtures (`device`, `dtype_fixture`)                                                                           |

## Adding new tests

- Place unit tests in `tests/test_<module>.py`.
- Mark slow GPU-only tests with `@pytest.mark.slow`.
- Mark full-dataset validation tests with `@pytest.mark.nightly`.
- New markers must be registered in `pyproject.toml` under `[tool.pytest.ini_options]`.
