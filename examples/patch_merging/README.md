# Unified patch-merging experiments

This directory documents the spatial-grid experiments recovered alongside the
token-based hierarchy in `examples/vit5_imagenet/v6_hierarchical/`.
Both use the same 2×2 concatenation order: top-left, bottom-left, top-right,
bottom-right, followed by normalization and a linear projection.

| API                      | Layout and behavior                                                                            | Network                             |
| ------------------------ | ---------------------------------------------------------------------------------------------- | ----------------------------------- |
| `PatchMerging`           | `[B, H*W, C]`; optional prepended register row; even grids; configurable output width and norm | `ViT5HierarchicalClassificationNet` |
| `PatchMerging2D`         | `[B, H, W, C]`; pads odd dimensions; LayerNorm then C → 2C                                     | `ViT5HierarchicalNet`               |
| `PatchEmbedHierarchical` | Channels-last image → stride-four convolution → LayerNorm                                      | Stem for `ViT5HierarchicalNet`      |

The two networks represent different experiments. The token hierarchy has
learned absolute position embeddings, configurable stage widths, and optional
FiLM registers. The spatial hierarchy has a normalized stride-four stem, no
position/register tokens, a stage-wide stochastic-depth schedule, and padding
at odd merge boundaries. Their checkpoints are not interchangeable. Existing
parameter names within each API are retained. Pure even-grid merger weights
with matching normalization are interchangeable and tested for identical
outputs and gradients.

## Recipes

- `examples/vit5_imagenet/v6_hierarchical/`: original PR #122 ImageNet pure and
  FiLM register-row recipes; `cifar10/` contains the patch-size ablations.
- `examples/vit5_imagenet/v5_patchmerge/`: spatial-grid hierarchy, scalar or
  block-diagonal SIREN kernels, per-stage frequency scaling and GRN.
- `examples/vit5_imagenet/local_comparison/`: local ImageNet comparison recipe.
- `examples/patch_merging/cifar10/`: spatial hierarchy versus flat Hyena,
  retaining the original Hugging Face data and augmentation recipe.

The v6 CIFAR recipes use `experiments.datamodules.cifar10` (torchvision source).
The recovered spatial recipes explicitly use `experiments.datamodules.cifar10_hf`
(Hugging Face source). Their augmentation settings and data sources remain
separate so consolidation does not silently redefine experiments. Set
`CIFAR10_PATH` for the latter; its default is `.data/cifar10`.

ImageNet training configs retain their Apex/DALI requirements. CPU model tests
use `fft_backend="torch_fft"`, without downloading datasets or launching training.
The FFT backend on current main has a separate CPU bf16 shortcut-dtype limitation;
real Hyena/FiLM smoke tests therefore use float32, while merger and hierarchy
wiring tests cover CPU bf16 autocast independently.

## Loading pretrained weights with a new classifier

`ViT5HierarchicalNet` exposes `out_proj` for the shared classification wrapper,
but its saved classifier keys are `network.head.weight` and `network.head.bias`.
When changing the number of classes, drop `network.head` before loading weights:

```python
from experiments.default_cfg import StartFromCheckpointConfig
from experiments.utils.checkpointing import DropKeysFromCheckpoint, StripCompiledPrefix
from nvsubquadratic.lazy_config import LazyConfig

# Set config.net.num_classes and the dataset for the new classification task.
config.start_from_checkpoint = StartFromCheckpointConfig(
    load=True,
    run_path="entity/project/pretrained-run-id",
    strict=False,
    callbacks=[
        LazyConfig(StripCompiledPrefix)(),
        LazyConfig(DropKeysFromCheckpoint)(prefixes=("network.head",)),
    ],
)
```

`strict=False` permits the removed classifier keys; it does not permit loading
classifier tensors with incompatible shapes. The filter is therefore required.
For the token-based `ViT5HierarchicalClassificationNet` and the flat
`ViT5ClassificationNet`, the corresponding prefix is `network.out_proj`.
This keeps existing checkpoint names and avoids loading a pretrained classifier
into a task with a different class count.

## Reconciliation with PR #122

Compared original PR head `bda14549317b7724f833f9f78a6926d2ace8a1c4`, local
patch-merging head `74dd91c0c617bd051016913f9cee066ae8a7f3b6` (including
`ed0bcbc49617111d9143f18415442b0972ead4f3`), and upstream main
`51291129a5f8ae5a2cfb80c6776d265d8a3667a6`.

The integration preserves the PR's feature history and merges current main.
Conflicts in older documentation/runtime history resolve to current main:
those changes were integrated separately through rewritten/squashed history.
Patch-merging code, examples and tests are then reconciled explicitly.

- Both merger APIs share one concatenation helper, while retaining their
  layouts, normalization and parameter names.
- Register-row zero padding is cast to projected-token dtype/device under
  autocast. A float32 pad would otherwise promote concatenated outputs to
  float32. Tests cover both padded and unpadded register rows.
- The misleading `ViT5ClassificationNet(prepend_registers=True)` reference is
  corrected to the actual hierarchical register-row model.
- Spatial configs no longer pass the removed `Hyena(use_rope=False)` argument.
- Spatial-hierarchy FLOP traversal follows ceil-halving for odd grids;
  construction checks the required channel doubling between stages.
- The local benchmark scripts already exist on main at
  `benchmarks/benchmark_patch_size_2d.py` and
  `benchmarks/vit5_imagenet/benchmark_imagenet_throughput.py`; use those copies.
- Unrelated spatial-recall Gaussian experiments, global FFT/mask changes,
  generated plots and historical result trackers from the mixed local commits
  are not additional patch-merging implementation. They remain in their own
  histories/review streams. No old results are re-certified by this integration.

## Validation

```bash
OMP_NUM_THREADS=2 python -m pytest \
    tests/modules/test_patch_merging.py \
    tests/modules/test_patch_merging_2d.py \
    tests/networks/test_vit5_hierarchical.py \
    tests/networks/test_vit5_hierarchical_classification.py \
    tests/test_patch_merging_cifar10.py -q -o addopts=''
```

Tests cover merger equivalence and gradients, padding and autocast, both network
layouts, real Hyena/FiLM CPU forward/backward, spatial block-diagonal kernels,
and download-free CIFAR batch contracts. Full training and GPU validation remain
separate checks.
