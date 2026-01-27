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
| XS    | 160        | ~700K-1M      |
| S     | 256        | ~1.8M-2.2M    |
| M     | 512        | ~7M-9M        |
