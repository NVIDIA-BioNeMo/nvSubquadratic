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

"""Thin wrapper around CleanFID helpers."""

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
    """Compute FID between a folder of samples and CleanFID reference statistics."""
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
