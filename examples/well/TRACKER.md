# The WELL Experiments Tracker

W&B project: [`dafidofff/nvsubquadratic-well`](https://wandb.ai/dafidofff/nvsubquadratic-well)

This tracker is for WELL experiments across datasets. The current focus is `supernova_explosion_64`; other WELL datasets can be added here as they become active.

## Datasets

| Dataset | Status | Notes |
|--------|--------|-------|
| `supernova_explosion_64` | Active | First WELL dataset under active investigation |

## Supernova Explosion 64

### Config files

| Config | Backbone | Mixer | Registers | FiLM conditioning | Planned duration | W&B run | Final val/VRMSE | test/VRMSE | test/NRMSE | test/PearsonR | Status | Notes |
|--------|----------|-------|-----------|-------------------|------------------|---------|-----------------|------------|------------|---------------|--------|-------|
| `supernova_explosion_64/cfg_hyena.py` | `ResidualNetwork` | Hyena | 0 | No | ~15 epochs already run | [`z6o20go9`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/z6o20go9) | `0.3266` | — | — | — | Completed baseline | Historical baseline used for reproducibility. This run only trained for ~15 epochs. |
| `supernova_explosion_64/cfg_attention.py` | `ViT5GeneralPurposeNet` | Attention | 14 | No | 30 epochs | — | — | — | — | — | Planned | ViT5-style dense prediction attention baseline with registers. |
| `supernova_explosion_64/cfg_vit5_attention.py` | `ViT5GeneralPurposeNet` | Attention | 14 | No | 30 epochs | — | — | — | — | — | Planned | Duplicate named ViT5 attention variant currently present in the repo. |
| `supernova_explosion_64/cfg_vit5_hyena.py` | `ViT5GeneralPurposeNet` | Hyena | 14 | No | 30 epochs | [`p7te253r`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/p7te253r) | `0.3561` | `0.3674` | `0.3643` | `0.9165` | Completed | ViT5-style Hyena without FiLM conditioning. |
| `supernova_explosion_64/cfg_vit5_hyena_film_conditioned.py` | `ViT5GeneralPurposeNet` | Hyena | 14 | Yes | 30 epochs | [`77n2mp0a`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/77n2mp0a) | `0.3615` | `0.3727` | `0.3696` | `0.9146` | Completed | ViT5-style Hyena with FiLM-conditioned kernel generation. |

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

### Observations

1. **Plain Hyena slightly outperforms FiLM-conditioned Hyena on supernova_explosion_64.** ViT5 Hyena without FiLM ([`p7te253r`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/p7te253r)) achieves test/VRMSE `0.3674` vs FiLM-conditioned ([`77n2mp0a`](https://wandb.ai/dafidofff/nvsubquadratic-well/runs/77n2mp0a)) at `0.3727`. The same trend holds across all test metrics (NRMSE `0.3643` vs `0.3696`, PearsonR `0.9165` vs `0.9146`). FiLM conditioning on the Hyena kernel does not appear to help on this dataset.

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