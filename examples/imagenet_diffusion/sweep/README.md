# Hyperparameter Sweep Priorities

This directory contains configuration files for the hyperparameter sweep of diffusion generative models on ImageNet.

## Priority 1: Large Model Scaling

1. **Large (64x64)**: `large_64.py`.
1. **Large (128x128)**: `large_128.py`.

## Priority 2: High-Res Baselines

1. **Base (128x128)**: Medium-resolution baseline. `base_128.py`.
1. **Base (256x256)**: Medium+-resolution baseline. `base_256.py`.
1. **XL (128x128)**: `xl_128.py`. **Extreme cost/Risk**.

## Priority 3: Low-Res Baseline

1. **Base (64x64)**: `base_64.py`. This is running currently under [this experiment](https://wandb.ai/dafidofff/nvsubquadratic/runs/vKZdfRjB?nw=nwuserdafidofff).

## Priority 4: Ablations

1. **Epsilon Prediction**: `base_64_epsilon.py`.
1. **Linear Schedule**: `base_64_linear_sched.py`.

## Resource Estimations (Nodes for GBS=512 / 1024)

| Experiment        | Nodes (GBS=512) | Nodes (GBS=1024) | Notes                  |
| :---------------- | :-------------- | :--------------- | :--------------------- |
| `large_64`        | 8               | 16               | Priority 1             |
| `large_128`       | 64              | 128              | Priority 1             |
| `base_128`        | 32              | 64               | Priority 2             |
| `base_256`        | 128             | 256              | Priority 2 (High cost) |
| `xl_128`          | 128             | 256              | Priority 2 (High cost) |
| `base_64`         | 4               | 8                | Priority 3             |
| `base_64_epsilon` | 4               | 8                | Priority 4             |
| `base_64_linear`  | 4               | 8                | Priority 4             |
