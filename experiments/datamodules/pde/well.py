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

"""Lightning DataModule for WELL benchmark datasets.

Wraps ``the_well.data.WellDataModule`` with:

* **Persistent HDF5 handles** — avoids re-opening files on every sample
  (upstream opens and closes per ``__getitem__``).
* **Optional NVMe staging** — copies the dataset to fast local storage
  before training to reduce network I/O.
* **Optional RAM preloading** — loads the entire training split into
  numpy arrays in the parent process; workers then share the data via
  OS copy-on-write pages with zero I/O overhead.
* **Normalization fix** — upstream only normalizes the train split; this
  wrapper applies the same normalization to val/test splits.
"""

import shutil
import subprocess
import time
import types
from pathlib import Path
from typing import Optional

import h5py as h5
import numpy as np
import pytorch_lightning as pl
from the_well.data import WellDataModule as BaseWellDataModule
from the_well.data.utils import IO_PARAMS, maximum_stride_for_initial_index


# ---------------------------------------------------------------------------
# In-memory h5py mocks
#
# These lightweight classes implement the subset of the h5py API that the
# upstream ``WellDataset._reconstruct_*`` methods use (indexing, .attrs,
# group iteration, __contains__).  After RAM preloading, each HDF5 file
# is represented as an _InMemFile -> _InMemGroup -> _InMemDataset tree
# backed entirely by numpy arrays.
# ---------------------------------------------------------------------------


class _InMemDataset:
    """h5py.Dataset mock: supports ``__getitem__``, ``.shape``, and ``.attrs``."""

    __slots__ = ("_data", "attrs")

    def __init__(self, data: np.ndarray, attrs: dict):
        self._data = data
        self.attrs = attrs

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def shape(self):
        return self._data.shape


class _InMemGroup:
    """h5py.Group mock: supports ``['key']`` lookup, iteration, ``keys()``, and ``.attrs``."""

    __slots__ = ("_items", "attrs")

    def __init__(self):
        self._items: dict[str, "_InMemDataset | _InMemGroup"] = {}
        self.attrs: dict = {}

    def __getitem__(self, key: str):
        return self._items[key]

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __iter__(self):
        return iter(self._items)

    def keys(self):
        return self._items.keys()


class _InMemFile(_InMemGroup):
    """h5py.File mock — identical to ``_InMemGroup`` (a File is just a root Group)."""

    pass


def _preload_to_ram(dataset) -> None:
    """Preload all HDF5 data into numpy arrays stored on the dataset.

    Called in the parent process **before** DataLoader workers fork, so the
    numpy arrays live in shared (copy-on-write) memory pages.  Workers read
    from these arrays with zero I/O overhead.

    After preloading, ``_enable_h5_caching`` is replaced: the patched
    ``_load_one_sample`` uses in-memory mocks instead of h5py Handles.

    Memory cost: one copy of the training data in RAM.  For Gray-Scott this
    is ~126 GB (6 files × 2 fields × 160×1001×128×128 float32).
    """
    preloaded: dict[int, _InMemFile] = {}  # file_idx -> mock file

    for file_idx, path in enumerate(dataset.files_paths):
        t0 = time.monotonic()
        with h5.File(
            dataset.fs.open(path, "rb", **IO_PARAMS["fsspec_params"]),
            "r",
            **IO_PARAMS["h5py_params"],
        ) as f:
            mock = _InMemFile()
            mock.attrs = dict(f.attrs)

            for group_name in f:
                grp = f[group_name]
                mg = _InMemGroup()
                mg.attrs = dict(grp.attrs)

                for ds_name in grp:
                    item = grp[ds_name]
                    if isinstance(item, h5.Dataset):
                        data = item[()]  # read entire array into memory
                        attrs = dict(item.attrs)
                        mg._items[ds_name] = _InMemDataset(data, attrs)
                    elif isinstance(item, h5.Group):
                        # Nested group (e.g. boundary_conditions/x_periodic)
                        sub_mg = _InMemGroup()
                        sub_mg.attrs = dict(item.attrs)
                        for sub_name in item:
                            sub_item = item[sub_name]
                            if isinstance(sub_item, h5.Dataset):
                                sub_mg._items[sub_name] = _InMemDataset(sub_item[()], dict(sub_item.attrs))
                        mg._items[ds_name] = sub_mg

                mock._items[group_name] = mg

        elapsed = time.monotonic() - t0
        gb = (
            sum(
                ds._data.nbytes
                for g in mock._items.values()
                if isinstance(g, _InMemGroup)
                for ds in g._items.values()
                if isinstance(ds, _InMemDataset)
            )
            / 1e9
        )
        print(
            f"[ram-preload] File {file_idx}/{len(dataset.files_paths) - 1}: {gb:.1f} GB in {elapsed:.1f}s",
            flush=True,
        )

        preloaded[file_idx] = mock

    dataset._preloaded_files = preloaded


def _enable_h5_caching(dataset, *, use_ram_preload: bool = False) -> None:
    """Patch a WellDataset to cache HDF5 file handles across ``__getitem__`` calls.

    The upstream ``WellDataset._load_one_sample`` opens and closes an ``h5py.File``
    for **every** sample.  On large files the open/close overhead dominates data
    loading time (~175 ms per sample even on NVMe).

    This function replaces ``_load_one_sample`` with a version that keeps file
    handles alive in a per-closure dict.  Handles are opened **lazily** on first
    access, so they are always created inside the DataLoader worker process —
    avoiding the h5py/fork() corruption issue.

    When ``use_ram_preload=True``, the patched loader reads from in-memory
    numpy arrays (preloaded by ``_preload_to_ram``) instead of h5py.  This
    gives zero-I/O data loading at the cost of RAM.

    Based on ``the_well==1.0.1`` ``WellDataset._load_one_sample``.
    """
    _h5_cache: dict = {}  # file_idx -> h5py.File, populated lazily per worker

    def _load_one_sample(self, index):
        # --- index resolution (upstream WellDataset logic) ---
        if self.restriction_set is not None:
            index = self.restriction_set[index]
        file_idx = int(np.searchsorted(self.file_index_offsets, index, side="right") - 1)
        windows_per_trajectory = self.n_windows_per_trajectory[file_idx]
        local_idx = index - max(self.file_index_offsets[file_idx], 0)
        sample_idx = local_idx // windows_per_trajectory
        time_idx = local_idx % windows_per_trajectory

        # --- file handle: RAM preload or lazy h5py cache ---
        preloaded = getattr(self, "_preloaded_files", None)
        if preloaded is not None and file_idx in preloaded:
            file = preloaded[file_idx]
        else:
            if file_idx not in _h5_cache:
                _h5_cache[file_idx] = h5.File(
                    self.fs.open(self.files_paths[file_idx], "rb", **IO_PARAMS["fsspec_params"]),
                    "r",
                    **IO_PARAMS["h5py_params"],
                )
            file = _h5_cache[file_idx]

        # --- data loading (upstream WellDataset logic) ---
        dt = self.min_dt_stride
        if self.max_dt_stride > self.min_dt_stride:
            effective_max_dt = maximum_stride_for_initial_index(
                time_idx,
                self.n_steps_per_trajectory[file_idx],
                self.n_steps_input,
                self.n_steps_output,
            )
            effective_max_dt = min(effective_max_dt, self.max_dt_stride)
            if effective_max_dt > self.min_dt_stride:
                dt = np.random.randint(self.min_dt_stride, effective_max_dt + 1)

        data = {}
        output_steps = min(self.n_steps_output, self.max_rollout_steps)
        if self.full_trajectory_mode and self.start_output_steps_at_t >= 0:
            time_idx = self.start_output_steps_at_t - (self.n_steps_input) * dt

        data["variable_fields"], data["constant_fields"] = self._reconstruct_fields(
            file,
            self.caches[file_idx],
            sample_idx,
            time_idx,
            self.n_steps_input + output_steps,
            dt,
        )
        data["variable_scalars"], data["constant_scalars"] = self._reconstruct_scalars(
            file,
            self.caches[file_idx],
            sample_idx,
            time_idx,
            self.n_steps_input + output_steps,
            dt,
        )

        if self.boundary_return_type is not None:
            data["boundary_conditions"] = self._reconstruct_bcs(
                file,
                self.caches[file_idx],
                sample_idx,
                time_idx,
                self.n_steps_input + output_steps,
                dt,
            )

        if self.return_grid:
            data["space_grid"], data["time_grid"] = self._reconstruct_grids(
                file,
                self.caches[file_idx],
                sample_idx,
                time_idx,
                self.n_steps_input + output_steps,
                dt,
            )

        return data, file_idx, sample_idx, time_idx, dt

    dataset._load_one_sample = types.MethodType(_load_one_sample, dataset)


class WellDataModule(pl.LightningDataModule):
    """Lightning DataModule for WELL benchmark datasets.

    Wraps ``the_well.data.WellDataModule`` to provide persistent HDF5 handles,
    optional NVMe staging, optional RAM preloading, and a normalization bugfix
    for val/test splits.  See the module docstring for the full feature list.

    Args:
        well_base_path: Path to the WELL datasets directory
        well_dataset_name: Name of the dataset (e.g., 'active_matter')
        batch_size: Batch size for training
        num_workers: Number of data loading workers (maps to ``data_workers`` in BaseWellDataModule)
        use_normalization: Whether to use normalization
        n_steps_input: Number of input timesteps
        n_steps_output: Number of output timesteps (for training)
        max_rollout_steps: Maximum number of rollout steps for validation
        min_dt_stride: Minimum time stride
        max_dt_stride: Maximum time stride (for data augmentation during training)
        local_staging_dir: Optional path to fast local storage (e.g. NVMe).
            When set, the dataset is copied there before training for faster I/O.
        prefetch_factor: Number of batches each worker pre-loads.  Higher values
            keep the GPU better fed at the cost of more CPU memory.  Only used
            when ``num_workers > 0``.
        persistent_workers: Keep worker processes alive between epochs to avoid
            re-opening HDF5 files and rebuilding worker state every epoch.
        preload_to_ram: If True, load all HDF5 data into RAM before forking
            DataLoader workers.  Workers share the data via copy-on-write pages,
            eliminating all h5py/fsspec I/O overhead.  Requires sufficient RAM
            (e.g. ~126 GB for Gray-Scott training split).  Default is False.
    """

    def __init__(
        self,
        well_base_path: str,
        well_dataset_name: str,
        batch_size: int = 64,
        num_workers: int = 4,
        use_normalization: bool = True,
        n_steps_input: int = 4,
        n_steps_output: int = 1,
        max_rollout_steps: int = 32,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        local_staging_dir: Optional[str] = None,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        preload_to_ram: bool = False,
    ):
        """Initialize the WELL DataModule with dataset and loader parameters."""
        super().__init__()
        self.well_base_path = well_base_path
        self._local_staging_dir = Path(local_staging_dir) if local_staging_dir is not None else None
        self.well_dataset_name = well_dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_normalization = use_normalization
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.max_rollout_steps = max_rollout_steps
        self.min_dt_stride = min_dt_stride
        self.max_dt_stride = max_dt_stride
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.preload_to_ram = preload_to_ram

        self._well_datamodule = None

    # ------------------------------------------------------------------
    # Local NVMe staging
    # ------------------------------------------------------------------

    def _stage_to_local(self) -> None:
        """Copy dataset to fast local storage (e.g. NVMe).

        Idempotent via a ``.staging_complete`` sentinel.  After staging,
        ``self.well_base_path`` is redirected to the local directory.

        Directory layout preserved:
            {well_base_path}/{dataset_name}/  ->  {local_staging_dir}/{dataset_name}/
        """
        src = Path(self.well_base_path) / self.well_dataset_name
        dst = self._local_staging_dir / self.well_dataset_name
        sentinel = dst / ".staging_complete"

        if not src.is_dir():
            raise FileNotFoundError(
                f"[data-staging] Source dataset not found: {src}. "
                f"Download it first with: bash scripts/download_well.sh {self.well_dataset_name}"
            )

        # Fast path: previous staging completed successfully.
        if sentinel.is_file():
            print(f"[data-staging] {self.well_dataset_name} already staged.", flush=True)
            self.well_base_path = str(self._local_staging_dir)
            return

        # Ensure destination is accessible before starting the copy.
        try:
            dst.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"[data-staging] Cannot access {self._local_staging_dir}: {exc}") from exc

        free_gb = shutil.disk_usage(self._local_staging_dir).free / (1024**3)
        print(f"[data-staging] Copying {self.well_dataset_name} -> {dst}  ({free_gb:.1f} GB free)", flush=True)

        # cp -a preserves permissions/timestamps, -r for recursive.
        # --no-clobber prevents conflicts when parallel jobs stage the same dataset.
        result = subprocess.run(
            ["cp", "-a", "--no-clobber", "-r", f"{src}/.", str(dst)],
            check=False,
            timeout=7200,
        )
        # cp --no-clobber exits 1 when it skips existing files (e.g. parallel
        # jobs staging to the same directory).  Only fail on unexpected codes.
        if result.returncode not in (0, 1):
            raise RuntimeError(f"[data-staging] cp failed with exit code {result.returncode}")

        sentinel.write_text("ok\n")
        self.well_base_path = str(self._local_staging_dir)
        print("[data-staging] Done.", flush=True)

    def _resolve_staged_path(self) -> None:
        """Point ``well_base_path`` to local staging dir if staging completed.

        Called from ``setup()`` (runs on **all** DDP ranks) so every rank
        sees the fast local path.  Only checks the sentinel — no I/O.
        """
        if self._local_staging_dir is None:
            return
        sentinel = self._local_staging_dir / self.well_dataset_name / ".staging_complete"
        if sentinel.is_file():
            self.well_base_path = str(self._local_staging_dir)

    def prepare_data(self):
        """Stage data to local storage if configured (rank 0 only in DDP)."""
        if self._local_staging_dir is not None:
            self._stage_to_local()

        dataset_dir = Path(self.well_base_path) / self.well_dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir}. "
                f"Download it first with: bash scripts/download_well.sh {self.well_dataset_name}"
            )

    def setup(self, stage=None):
        """Setup datasets (called on each process in DDP)."""
        self._resolve_staged_path()
        print(f"[WellDataModule] Loading {self.well_dataset_name} from {self.well_base_path} …", flush=True)
        from the_well.data.normalization import ZScoreNormalization

        # Create the WELL datamodule
        self._well_datamodule = BaseWellDataModule(
            well_base_path=self.well_base_path,
            well_dataset_name=self.well_dataset_name,
            batch_size=self.batch_size,
            use_normalization=self.use_normalization,
            normalization_type=ZScoreNormalization if self.use_normalization else None,
            n_steps_input=self.n_steps_input,
            n_steps_output=self.n_steps_output,
            max_rollout_steps=self.max_rollout_steps,
            min_dt_stride=self.min_dt_stride,
            max_dt_stride=self.max_dt_stride,
            data_workers=self.num_workers,
        )

        # BUGFIX: The WELL library doesn't apply normalization to val/test datasets
        # We need to manually set it using the train dataset's normalization
        if self.use_normalization and hasattr(self._well_datamodule.train_dataset, "norm"):
            train_normalization = self._well_datamodule.train_dataset.norm

            # Apply same normalization to val dataset
            self._well_datamodule.val_dataset.use_normalization = True
            self._well_datamodule.val_dataset.norm = train_normalization

            # Apply same normalization to rollout val dataset
            self._well_datamodule.rollout_val_dataset.use_normalization = True
            self._well_datamodule.rollout_val_dataset.norm = train_normalization

            # Apply same normalization to test dataset
            self._well_datamodule.test_dataset.use_normalization = True
            self._well_datamodule.test_dataset.norm = train_normalization

            # Apply same normalization to rollout test dataset
            self._well_datamodule.rollout_test_dataset.use_normalization = True
            self._well_datamodule.rollout_test_dataset.norm = train_normalization

        # Preload training data into RAM if requested (before workers fork).
        # Only train split is preloaded — val/test are infrequent enough
        # that h5py caching is fine.
        all_splits = (
            self._well_datamodule.train_dataset,
            self._well_datamodule.val_dataset,
            self._well_datamodule.rollout_val_dataset,
            self._well_datamodule.test_dataset,
            self._well_datamodule.rollout_test_dataset,
        )
        if self.preload_to_ram:
            print("[WellDataModule] Preloading training data into RAM …", flush=True)
            _preload_to_ram(self._well_datamodule.train_dataset)
            total_gb = (
                sum(
                    ds._data.nbytes
                    for mock in self._well_datamodule.train_dataset._preloaded_files.values()
                    for g in mock._items.values()
                    if isinstance(g, _InMemGroup)
                    for ds in g._items.values()
                    if isinstance(ds, _InMemDataset)
                )
                / 1e9
            )
            print(f"[WellDataModule] Preloaded {total_gb:.1f} GB into RAM.", flush=True)

        # Patch all dataset splits to cache HDF5 file handles (each split
        # gets its own independent cache via a fresh closure).
        for ds in all_splits:
            _enable_h5_caching(ds, use_ram_preload=self.preload_to_ram)

        # Get metadata from the training dataset
        metadata = self._well_datamodule.train_dataset.metadata

        # Calculate input and output channels
        # Input: n_steps_input timesteps × n_fields + constant fields
        # Output: n_fields (single timestep prediction)
        self._input_channels = self.n_steps_input * metadata.n_fields + metadata.n_constant_fields
        self._output_channels = metadata.n_fields

        # Store metadata for use in model/wrapper
        self.metadata = metadata

    def _loader_kwargs(self) -> dict:
        """Common DataLoader kwargs shared across all splits."""
        use_workers = self.num_workers > 0
        return {
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.persistent_workers and use_workers,
            "prefetch_factor": self.prefetch_factor if use_workers else None,
        }

    def train_dataloader(self):
        """Return the training dataloader with persistent workers and prefetching."""
        from torch.utils.data import DataLoader

        return DataLoader(
            self._well_datamodule.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            **self._loader_kwargs(),
        )

    def val_dataloader(self):
        """Return the validation dataloader with persistent workers and prefetching."""
        from torch.utils.data import DataLoader

        return DataLoader(
            self._well_datamodule.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self):
        """Return the test dataloader with persistent workers and prefetching."""
        from torch.utils.data import DataLoader

        return DataLoader(
            self._well_datamodule.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            **self._loader_kwargs(),
        )

    @property
    def input_channels(self):
        """Number of input channels for the model."""
        if self._well_datamodule is None:
            raise RuntimeError("DataModule not setup yet. Call setup() first.")
        return self._input_channels

    @property
    def output_channels(self):
        """Number of output channels for the model."""
        if self._well_datamodule is None:
            raise RuntimeError("DataModule not setup yet. Call setup() first.")
        return self._output_channels

    @property
    def normalization(self):
        """Normalization object (ZScoreNormalization) from the training split, or None."""
        if self._well_datamodule is None:
            raise RuntimeError("DataModule not setup yet. Call setup() first.")
        if self.use_normalization and hasattr(self._well_datamodule.train_dataset, "norm"):
            return self._well_datamodule.train_dataset.norm
        return None

    @property
    def rollout_val_dataloader(self):
        """Return long rollout validation dataloader."""
        from torch.utils.data import DataLoader

        return DataLoader(
            self._well_datamodule.rollout_val_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            **self._loader_kwargs(),
        )

    @property
    def rollout_test_dataloader(self):
        """Return long rollout test dataloader."""
        from torch.utils.data import DataLoader

        return DataLoader(
            self._well_datamodule.rollout_test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True,
            **self._loader_kwargs(),
        )
