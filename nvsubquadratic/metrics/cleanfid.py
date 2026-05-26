# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Thin wrapper around CleanFID image generation quality metrics.

Fréchet Inception Distance (FID) measures the distributional similarity between
a set of generated images and a reference dataset by comparing the mean and
covariance of Inception-v3 feature vectors (Heusel et al., "GANs Trained by a
Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", NeurIPS 2017).

CleanFID (Parmar et al., "On Aliased Resizing and Surprising Subtleties in GAN
Evaluation", CVPR 2022) corrects for common pre-processing inconsistencies
(e.g. JPEG re-compression, bilinear vs Lanczos resizing) that cause FID scores
to be non-reproducible across libraries.  This module delegates to the
``cleanfid`` package, which ships pre-computed reference statistics for standard
benchmarks (FFHQ, CIFAR-10, ImageNet, etc.).

Usage::

    score = compute_folder_fid(
        sample_dir="outputs/samples/",
        dataset_name="imagenet",
        dataset_resolution=256,
        dataset_split="train",
    )
    print(f"FID: {score:.2f}")
"""

from __future__ import annotations

from pathlib import Path

from cleanfid import fid


def compute_folder_fid(
    sample_dir: str | Path,
    *,
    dataset_name: str,
    dataset_resolution: int,
    dataset_split: str = "train",
) -> float:
    """Compute CleanFID between a folder of generated images and a reference dataset.

    Calls ``cleanfid.fid.compute_fid`` with pre-computed reference statistics
    for ``dataset_name`` at ``dataset_resolution``, so no reference images need
    to be stored locally.

    Args:
        sample_dir: Path to the directory containing generated images (PNG/JPEG).
            Expanded and resolved to an absolute path before use.
        dataset_name: Name of the CleanFID reference dataset, e.g.
            ``"imagenet"``, ``"ffhq"``, ``"cifar10"``.  Must match a dataset
            whose statistics are bundled with or downloaded by ``cleanfid``.
        dataset_resolution: Reference image resolution in pixels, e.g. ``256``
            for 256×256 ImageNet.
        dataset_split: Which split of the reference dataset to compare against.
            Default ``"train"``.

    Returns:
        FID score as a Python ``float``.  Lower is better; 0 means the
        generated and reference distributions are identical under the Inception
        feature extractor.

    Raises:
        FileNotFoundError: If ``sample_dir`` does not exist.
    """
    sample_path = Path(sample_dir).expanduser().resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_path}")

    score = fid.compute_fid(
        fdir1=sample_path.as_posix(),
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        dataset_split=dataset_split,
    )
    return float(score)
