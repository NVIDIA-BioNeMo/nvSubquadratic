# The WELL Experiments Tracker

W&B project: [`dafidofff/nvsubquadratic-well`](https://wandb.ai/dafidofff/nvsubquadratic-well)

This tracker is for WELL experiments across datasets. The current focus is `supernova_explosion_64`; other WELL datasets can be added here as they become active.

## Datasets

| Dataset | Status | Notes |
|--------|--------|-------|
| `supernova_explosion_64` | Active | First WELL dataset under active investigation |

## Supernova Explosion 64

### Config files

| Config | Backbone | Mixer | Registers | FiLM conditioning | Planned duration | W&B run | Final val/VRMSE | Status | Notes |
|--------|----------|-------|-----------|-------------------|------------------|---------|-----------------|--------|-------|
| `supernova_explosion_64/cfg_hyena.py` | `ResidualNetwork` | Hyena | 0 | No | ~15 epochs already run | [`z6o20go9`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/z6o20go9) | `0.3266` | Completed baseline | Historical baseline used for reproducibility. This run only trained for ~15 epochs. |
| `supernova_explosion_64/cfg_attention.py` | `ViT5GeneralPurposeNet` | Attention | 14 | No | 30 epochs | — | — | Planned | ViT5-style dense prediction attention baseline with registers. |
| `supernova_explosion_64/cfg_vit5_attention.py` | `ViT5GeneralPurposeNet` | Attention | 14 | No | 30 epochs | — | — | Planned | Duplicate named ViT5 attention variant currently present in the repo. |
| `supernova_explosion_64/cfg_vit5_hyena.py` | `ViT5GeneralPurposeNet` | Hyena | 14 | No | 30 epochs | — | — | Planned | ViT5-style Hyena without FiLM conditioning. |
| `supernova_explosion_64/cfg_vit5_hyena_film_conditioned.py` | `ViT5GeneralPurposeNet` | Hyena | 14 | Yes | 30 epochs | — | — | Planned | ViT5-style Hyena with FiLM-conditioned kernel generation. |

### Shared supernova setup

| Setting | Value |
|--------|-------|
| Dataset | `supernova_explosion_64` |
| Task | Autoregressive / rollout regression via `WELLRegressionWrapper` |
| Input steps | 4 |
| Output steps | 1 |
| Rollout during validation | 1 |
| Spatial size | `64^3` |
| Precision | `bf16-mixed` |

### Training schedule

The currently planned supernova runs should use:

- `cfg_hyena.py`: historical baseline at `130_000` iterations, corresponding to roughly 15 epochs in the completed run.
- All other current supernova configs: `260_000` iterations, targeting roughly 30 epochs.

## How to launch a new WELL run

```bash
sbatch slurm/submit.sh examples/well/<dataset>/<config>.py
```

## How to monitor

```bash
# Job status
squeue -u dwessels2

# Tail stdout log
tail -f logs/<NAME>_<JOBID>.out

# Tail stderr
tail -f logs/<NAME>_<JOBID>.err
```