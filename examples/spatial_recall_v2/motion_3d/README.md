# 3D motion spatial recall

This moving-image copy task extends spatial recall from a static image to a
short video block. A source image is resized, translated along an ink-aware
monotone path and optionally rotated in quarter turns. Depth represents time.
The model must reproduce the entire block at the back-bottom-right readout
corner of a larger cubic canvas.

`SpatialRecall3DMotionDataset` returns a canvas `[C, S, S, S]` and target
`[C, b, b, b]`, where `S = canvas_size` and `b = block_size`. The target is
placed at the front-top-left corner (`placement="fixed"`) or a random origin
that does not overlap the readout (`placement="random"`). The readout is filled
with `readout_value` (zero by default). The target is the video itself, not the
source dataset's class label.

## Download-free example

Run from the repository root in an environment with the project's core Python
dependencies installed:

```python
import torch
from torch.utils.data import TensorDataset

from experiments.datamodules.spatial_recall_dataset import SpatialRecall3DMotionDataset

image = torch.zeros(1, 1, 8, 8)
image[:, :, 2:6, 3:5] = 1
source = TensorDataset(image, torch.zeros(1, dtype=torch.long))
dataset = SpatialRecall3DMotionDataset(
    base_dataset=source,
    digit_size=4,
    block_size=6,
    canvas_size=32,
    generator=torch.Generator().manual_seed(42),
    placement="fixed",
    max_step=2.0,
    spin=True,
)
canvas, target = dataset[0]
assert canvas.shape == (1, 32, 32, 32)
assert target.shape == (1, 6, 6, 6)
assert torch.equal(canvas[:, :6, :6, :6], target)
assert torch.count_nonzero(canvas[:, -6:, -6:, -6:]) == 0
```

## Lightning integration

`SpatialRecall3DMotionDataModule` wraps an image base datamodule such as
MNIST or EMNIST using its existing lazy configuration:

```python
from experiments.datamodules.emnist import EMNISTDataModule
from experiments.datamodules.spatial_recall_dataset import (
    SpatialRecall3DMotionDataModule,
)
from nvsubquadratic.lazy_config import LazyConfig

module = SpatialRecall3DMotionDataModule(
    base_datamodule_cfg=LazyConfig(EMNISTDataModule)(
        data_dir=".data/emnist",
        batch_size=16,
        data_type="image",
        num_workers=4,
        pin_memory=True,
        permuted=False,
        seed=42,
        normalize_input=True,
        split="byclass",
    ),
    digit_size=4,
    block_size=6,
    canvas_size=32,
    data_type="volume",
)
# Explicitly call module.prepare_data() to download EMNIST if needed,
# then module.setup("fit") to construct the train/validation datasets.
```

The wrapper derives `input_channels` and `output_channels` from the base
module's `input_channels`, including before setup. The target copies image
channels, so the base module's number of classes is irrelevant. Grayscale and
RGB image sources are supported; the base must expose `input_channels`.

The batch-transfer hook returns `{"input": x, "label": y, "condition": None}`.
With `data_type="volume"`, shapes are `[B, S, S, S, C]` and `[B, b, b, b, C]`.
A regression network reads the final `b` positions along each spatial axis;
for `ResidualNetwork`, use `data_dim=3` and `target_size=[b, b, b]`.

With `data_type="sequence"`, shapes are `[B, S**3, C]` and `[B, b**3, C]`.
Flattening uses depth-height-width order: token index is `d*S*S + h*S + w`.
The readout cube is **not** the final `b**3` tokens. A sequence model must
produce predictions for the full canvas, then gather the cube before comparing
to the label (in the model's readout or the loss):

```python
# prediction: full-canvas model output [B, S**3, C]
S, b = module.canvas_size, module.block_size
coords = torch.arange(S - b, S, device=prediction.device)
d, h, w = torch.meshgrid(coords, coords, coords, indexing="ij")
indices = (d * S * S + h * S + w).reshape(-1)
readout = prediction.index_select(1, indices)  # [B, b**3, C], label order
```

This gather must be wired explicitly for sequence mode. `ResidualNetwork`'s
built-in spatial crop supports the volume recipe above; its one-dimensional
tail crop does not select this cube, and a three-axis `target_size` is invalid
for a sequence input.

## Sampling semantics

- Require `0 < digit_size <= block_size` and `canvas_size >= 2 * block_size`,
  including fixed placement, so source and readout cannot overlap.
- `max_step` is a finite, nonnegative **per-axis continuous** step limit.
  Rounded increments are clamped to `ceil(max_step)`, including half-to-even
  rounding ties; this is not a
  bound on Euclidean speed or on pixel displacement due to rotation. Zero
  disables translation but does not disable spin.
- Sweeps shrink about their centre when needed to respect the continuous
  limit. They need not reach both endpoints or fill every spatial slice.
- Ink bounds use a threshold 15% above the source image's minimum intensity.
  Low-intensity background pixels may clip at the block boundary. Each block
  uses the per-channel source minimum as its background; the surrounding
  canvas uses zero.
- Spin selects an initial quarter-turn orientation and one to three turns
  (limited by the available time steps), with a random direction. Rotations
  happen before resizing. For a single time step only the initial orientation
  is observed.
- Sampling advances generator state on each access. Equal seeds reproduce
  equal access sequences; a sample is not a fixed function of its index.
  The datamodule derives loader and train/validation/test seeds from the base
  seed, global rank, and stream ID using NumPy `SeedSequence`. Base dataset
  splits retain the original seed on every rank. Motion streams differ across
  ranks, including with zero loader workers; workers seed their motion RNGs
  from the rank-dependent PyTorch worker seeds. This corrects the former
  rank-identical streams and changes samples for a given historical seed.
  Reproduction also requires the same rank, loader order, and worker count. Validation
  draws new motion samples on successive passes; compare frozen samples when
  exact repeated inputs are required.

These semantics describe the public implementation. They do not certify the
source used for historical research runs. No research results, cluster
manifests, colour-conditioning variants or unpublished model changes are
included with this dataset.

## Validation

```bash
python -m pytest tests/test_spatial_recall_motion.py -q -o addopts=''
```

The tests use synthetic tensors and run on CPU without dataset downloads.
