# EMNIST Spatial Recall 3D - Simple Copy

## Task Description

This is the simplest variant of the 3D spatial recall task:

- A single 2D EMNIST digit is placed on a depth slice of a 3D volume
- The model must recall this digit at the back-bottom-right corner (readout region)
- Fixed placement: digit is always placed at front-top-left (d=0, y=0, x=0)

## 3D Canvas Structure

The 3D canvas has shape `[C, D, H, W]`:

- **D** = depth (e.g., 8 slices)
- **H** = height (e.g., 64)
- **W** = width (e.g., 64)

The readout region is at:

- Depth: last slice (d = D-1, the "back" plane)
- Position: bottom-right corner of that slice

## Visualization

Run the visualization script to see sample data:

```bash
PYTHONPATH=. python experiments/datamodules/spatial_recall_dataset.py \
    --mode 3d \
    --placement fixed \
    --target-size 16 \
    --canvas-size 64 \
    --canvas-depth 8 \
    --dataset emnist \
    --batch-size 4
```

## Running Experiments

```bash
# Hyena XS
python -m nvsubquadratic.train \
    --config examples/spatial_recall_3d/emnist_regression_simple_copy/ccnn_hyena_xs.py
```

## Model Variants

| Model | Hidden Dim | Approx Params |
| ----- | ---------- | ------------- |
| XS    | 160        | ~800K         |
| S     | 256        | ~2M           |
| M     | 416        | ~5M           |

## Max Batch Sizes (80GB H100)

Due to 3D FFT convolutions requiring fp32 internally, memory is the main constraint.

| Model | fp32 max | bf16-mixed max |
| ----- | -------- | -------------- |
| XS    | 20       | 22             |
| S     | 12       | 14             |
| M     | 7        | 8              |

To achieve effective batch size 64, use gradient accumulation:

```bash
# XS: batch=16, accum=4 -> effective 64
sbatch run_cxis.sh examples/.../ccnn_hyena_xs.py \
    train.precision=bf16-mixed \
    dataset.base_datamodule_cfg.batch_size=16 \
    train.accumulate_grad_steps=4

# S: batch=8, accum=8 -> effective 64
sbatch run_cxis.sh examples/.../ccnn_hyena_s.py \
    train.precision=bf16-mixed \
    dataset.base_datamodule_cfg.batch_size=8 \
    train.accumulate_grad_steps=8

# M: batch=8, accum=8 -> effective 64
sbatch run_cxis.sh examples/.../ccnn_hyena_m.py \
    train.precision=bf16-mixed \
    dataset.base_datamodule_cfg.batch_size=8 \
    train.accumulate_grad_steps=8
```

Or use multi-GPU DDP:

```bash
# 4 GPUs with batch=16 each -> effective 64
sbatch --gres=gpu:4 --ntasks-per-node=4 --cpus-per-task=8 run_cxis.sh \
    examples/.../ccnn_hyena_xs.py \
    train.precision=bf16-mixed \
    dataset.base_datamodule_cfg.batch_size=16
```
