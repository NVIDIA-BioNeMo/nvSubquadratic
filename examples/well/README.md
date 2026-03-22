# The Well Benchmark Experiments

Experiments using [The Well](https://polymathic-ai.org/the_well/) (NeurIPS 2024), a 15TB collection of physics simulation datasets for PDE surrogate modeling.

Paper: [arXiv:2412.00568](https://arxiv.org/abs/2412.00568)

## Prerequisites

1. Python environment with `nvsubquadratic` and its dependencies installed (see root `pyproject.toml`)
1. The Well library installed:
   ```bash
   pip install the-well
   ```
   Or from source: `git clone https://github.com/PolymathicAI/the_well.git && cd the_well && pip install . && cd ..`

## Data Download

Set the `WELL_DATA_PATH` environment variable to your data directory (add to your `.bashrc` or equivalent):

```bash
export WELL_DATA_PATH=/path/to/your/well/data
```

Download a dataset:

```bash
# Download all splits for a dataset
bash scripts/download_well.sh active_matter

# Download a specific split
bash scripts/download_well.sh gray_scott_reaction_diffusion --split train

# Override the data path
bash scripts/download_well.sh MHD_64 --base-path /scratch/data/the_well
```

On a SLURM cluster:

```bash
sbatch slurm/download_well.sh active_matter
```

Run `bash scripts/download_well.sh` without arguments to see the full list of available datasets.

## Available Dataset Configs

| Dataset                         | Resolution | Dim | Fields | BCs          | Batch Size | Config                                         |
| ------------------------------- | ---------- | --- | ------ | ------------ | ---------- | ---------------------------------------------- |
| `active_matter`                 | 256x256    | 2D  | 3      | Periodic     | 16         | `examples/well/active_matter/`                 |
| `gray_scott_reaction_diffusion` | 128x128    | 2D  | 2      | Periodic     | 64         | `examples/well/gray_scott_reaction_diffusion/` |
| `rayleigh_benard`               | 512x128    | 2D  | 3+     | Mixed (wall) | 16         | `examples/well/rayleigh_benard/`               |
| `MHD_64`                        | 64x64x64   | 3D  | 4      | Periodic     | 2          | `examples/well/MHD_64/`                        |

Each dataset directory contains configs for different architectures. Currently `cfg_hyena.py` is provided for all datasets; `active_matter` also has `cfg_attention.py` and `cfg_ckconv.py`.

## Running Experiments

### Single GPU

```bash
PYTHONPATH=. python experiments/run.py --config examples/well/active_matter/cfg_hyena.py
```

### Multi-GPU (via SLURM srun)

```bash
PYTHONPATH=. srun --ntasks=4 --gres=gpu:4 python experiments/run.py \
    --config examples/well/active_matter/cfg_hyena.py \
    dataset.batch_size=4
```

### Config Overrides

Any config parameter can be overridden from the command line:

```bash
# Adjust batch size and training iterations
PYTHONPATH=. python experiments/run.py \
    --config examples/well/active_matter/cfg_hyena.py \
    dataset.batch_size=8 \
    train.iterations=50000

# Enable spatial downsampling (e.g., 4x -> 256x256 becomes 64x64)
PYTHONPATH=. python experiments/run.py \
    --config examples/well/active_matter/cfg_hyena.py \
    dataset.spatial_downsample_factor=4
```

## Architecture

The pipeline consists of three components:

1. **DataModule** (`experiments/datamodules/pde/well.py`): Wraps The Well's `WellDataModule` with normalization fixes, spatial downsampling, and channel calculation for the nvSubquadratic interface.

1. **Lightning Wrapper** (`experiments/lightning_wrappers/well_lightning_wrapper.py`): Handles input formatting (flattening timesteps into channels), single-step training, autoregressive rollout for validation/test, and metric computation on denormalized (physical-scale) data.

1. **Network**: Any `ResidualNetwork` architecture from nvSubquadratic (Hyena, Attention, CKConv). Input/output channels are set automatically from the dataset metadata.

### Data Flow

```
[B, T, H, W, C] input_fields  -->  flatten T into C  -->  [B, H, W, T*C + C_const]
                                                              |
                                                        ResidualNetwork
                                                              |
                                                        [B, H, W, C_out]
```

## Metrics

Metrics are computed on **denormalized** (physical-scale) data using The Well's `validation_metric_suite`. The primary metric is **VRMSE** (Volume-averaged Root Mean Squared Error). All metrics are logged to Weights & Biases.

## Adding a New Dataset

1. Copy an existing config directory (e.g., `examples/well/active_matter/`) to `examples/well/<new_dataset>/`
1. Update these parameters in the config file:
   - `WELL_DATASET_NAME`: exact name from The Well (see `scripts/download_well.sh` for the full list)
   - `BATCH_SIZE`: adjust for your GPU memory (start with The Well's default, see their benchmark configs)
   - `DATA_DIM`: 2 for 2D, 3 for 3D datasets
   - `FFT_PADDING`: `"circular"` for periodic BCs, `"zero"` for non-periodic
   - `SPATIAL_RESOLUTION`: native resolution of the dataset (per dimension)
   - `short_conv_cfg`: use `Conv2d` for 2D, `Conv3d` for 3D
   - `wandb.job_group`: descriptive group name
1. Download the data: `bash scripts/download_well.sh <new_dataset>`
1. Run: `PYTHONPATH=. python experiments/run.py --config examples/well/<new_dataset>/cfg_hyena.py`

## Experimental Results

### Active Matter (256x256, periodic BCs)

All models trained with 130k iterations, batch size 16, cosine schedule with 10% warmup. Metric: **test/VRMSE** (lower is better). Entries marked with * use `spatial_downsample_factor`.

#### Hyena

| Patch | Depth | Emb Dim | Params | test/VRMSE  |
| ----- | ----- | ------- | ------ | ----------- |
| 8     | 6     | 256     | 3.9M   | 0.02066     |
| 8     | 8     | 256     | 4.9M   | 0.01893     |
| 8     | 10    | 256     | 6.0M   | 0.01816     |
| 8     | 6     | 384     | 7.9M   | 0.01768     |
| 8     | 8     | 384     | 10.1M  | 0.01599     |
| 8     | 10    | 384     | 12.3M  | 0.01506     |
| 8     | 6     | 512     | 13.3M  | 0.01541     |
| 8     | 8     | 512     | 17.1M  | 0.01425     |
| 8     | 10    | 512     | 20.9M  | **0.01375** |
| 16\*  | 6     | 256     | 6.6M   | 0.03489     |
| 16\*  | 8     | 256     | 7.6M   | 0.03062     |
| 16\*  | 10    | 256     | 8.7M   | 0.02908     |
| 16\*  | 6     | 384     | 12.0M  | 0.02773     |
| 16\*  | 8     | 384     | 14.2M  | 0.02510     |
| 16\*  | 10    | 384     | 16.4M  | 0.02413     |
| 16\*  | 6     | 512     | 18.7M  | 0.02500     |
| 16\*  | 8     | 512     | 22.5M  | 0.02230     |
| 16\*  | 10    | 512     | 26.4M  | 0.02092     |

#### ViT (Attention)

| Patch | Depth | Emb Dim | Params | test/VRMSE  |
| ----- | ----- | ------- | ------ | ----------- |
| 8     | 6     | 256     | 3.3M   | 0.03793     |
| 8     | 8     | 256     | 4.1M   | 0.03054     |
| 8     | 10    | 256     | 4.9M   | 0.02699     |
| 8     | 6     | 384     | 6.8M   | 0.02484     |
| 8     | 8     | 384     | 8.6M   | 0.02424     |
| 8     | 10    | 384     | 10.4M  | 0.02111     |
| 8     | 6     | 512     | 11.5M  | 0.02555     |
| 8     | 8     | 512     | 14.7M  | 0.02273     |
| 8     | 10    | 512     | 17.8M  | **0.01976** |
| 16    | 6     | 256     | 6.0M   | 0.05758     |
| 16    | 8     | 256     | 6.8M   | 0.04979     |
| 16    | 10    | 256     | 7.6M   | 0.04497     |
| 16    | 6     | 384     | 10.9M  | 0.04495     |
| 16    | 8     | 384     | 12.7M  | 0.03743     |
| 16    | 10    | 384     | 14.4M  | 0.03514     |
| 16    | 6     | 512     | 16.9M  | 0.03798     |
| 16    | 8     | 512     | 20.1M  | 0.03355     |
| 16    | 10    | 512     | 23.3M  | 0.03148     |
