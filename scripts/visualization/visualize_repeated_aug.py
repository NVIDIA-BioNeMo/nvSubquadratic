#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualize repeated augmentation: save a PNG grid showing the same source
images with different augmented views side by side.

Usage (inside SLURM container with GPU):
    PYTHONPATH=. python scripts/visualize_repeated_aug.py

Output: outputs/repeated_aug_grid.png
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.datamodules.dali_imagenet_fused import _RepeatedAugSource


TRAIN_ROOT = "/shared/data/image_datasets/imagenet_folder/train"
NUM_REPEATS = 3
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_BATCHES = 8
NUM_IMAGES_TO_SHOW = 6
MAX_VIEWS = 4
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "repeated_aug_grid.png"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def unnormalize(img_chw: np.ndarray) -> np.ndarray:
    """CHW float32 -> HWC uint8."""
    img = img_chw.transpose(1, 2, 0)
    img = img * STD + MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


@pipeline_def(enable_conditionals=True)
def vis_pipeline(ra_source):
    jpegs, labels = fn.external_source(
        source=ra_source,
        num_outputs=2,
        batch=False,
        parallel=True,
    )
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    images = fn.random_resized_crop(
        images,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        random_area=(0.08, 1.0),
        interp_type=types.INTERP_CUBIC,
    )
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[m * 255.0 for m in MEAN],
        std=[s * 255.0 for s in STD],
    )
    return images, labels


def main():
    print(f"Building _RepeatedAugSource from {TRAIN_ROOT} ...")
    src = _RepeatedAugSource(
        file_root=TRAIN_ROOT,
        num_repeats=NUM_REPEATS,
        shard_id=0,
        num_shards=1,
        seed=42,
    )
    print(f"  Total files: {len(src._files)}, num_selected: {src.num_selected}")

    # Build a reverse map: file_index -> list of positions in the epoch
    idx_positions: dict[int, list[int]] = defaultdict(list)
    for pos, file_idx in enumerate(src._indices):
        idx_positions[file_idx].append(pos)

    pipe = vis_pipeline(
        ra_source=src,
        batch_size=BATCH_SIZE,
        num_threads=4,
        device_id=0,
        seed=0,
    )
    pipe.build()

    it = DALIGenericIterator(
        pipe,
        output_map=["images", "labels"],
        size=src.num_selected,
        last_batch_policy=LastBatchPolicy.DROP,
        auto_reset=True,
    )

    # Collect augmented views keyed by original file index
    file_idx_to_views: dict[int, list[np.ndarray]] = defaultdict(list)
    samples_seen = 0

    print(f"Iterating {NUM_BATCHES} batches (bs={BATCH_SIZE}) ...")
    for batch_i, batch in enumerate(it):
        if batch_i >= NUM_BATCHES:
            break
        data = batch[0]
        imgs = data["images"].cpu().numpy()
        for j in range(imgs.shape[0]):
            global_pos = batch_i * BATCH_SIZE + j
            if global_pos >= len(src._indices):
                break
            file_idx = src._indices[global_pos]
            file_idx_to_views[file_idx].append(unnormalize(imgs[j]))
            samples_seen += 1

    print(f"  Collected {samples_seen} samples across {len(file_idx_to_views)} unique images")

    # Pick images that actually got multiple views in our collected batches
    candidates = [(fidx, views) for fidx, views in file_idx_to_views.items() if len(views) >= 2]
    candidates.sort(key=lambda x: -len(x[1]))
    selected = candidates[:NUM_IMAGES_TO_SHOW]

    if not selected:
        print("ERROR: No images with multiple views found. Try increasing NUM_BATCHES.")
        sys.exit(1)

    print(f"  Showing {len(selected)} images, each with up to {MAX_VIEWS} views")

    # Build the grid
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("ERROR: Pillow not installed")
        sys.exit(1)

    pad = 4
    label_h = 24
    cell_w = IMAGE_SIZE + pad
    cell_h = IMAGE_SIZE + pad + label_h
    cols = MAX_VIEWS
    rows = len(selected)
    grid_w = cols * cell_w + pad
    grid_h = rows * cell_h + pad
    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    for row_i, (file_idx, views) in enumerate(selected):
        class_name = Path(src._files[file_idx]).parent.name
        for col_i in range(min(len(views), MAX_VIEWS)):
            x = pad + col_i * cell_w
            y = pad + row_i * cell_h
            img_pil = Image.fromarray(views[col_i])
            grid.paste(img_pil, (x, y))
            label = f"idx={file_idx} | {class_name}" if col_i == 0 else f"view {col_i + 1}"
            draw.text((x + 2, y + IMAGE_SIZE + 2), label, fill=(220, 220, 220))

    grid.save(str(OUTPUT_PATH))
    print(f"\nSaved grid to {OUTPUT_PATH}")
    print(f"  Grid size: {grid_w} x {grid_h}")
    print(f"  Rows = {rows} source images, Cols = up to {MAX_VIEWS} augmented views each")


if __name__ == "__main__":
    main()
