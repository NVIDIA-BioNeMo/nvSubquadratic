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


"""UCF101 datamodule with in-code download in prepare_data."""

import os
import random
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms._transforms_video as TV
from einops import rearrange
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


_BASE_SEED = 0


def set_base_seed(seed):
    """Set the base seed for worker initialization."""
    global _BASE_SEED
    _BASE_SEED = seed


def deterministic_worker_init_fn(worker_id: int):
    """Initialize the worker with a deterministic seed derived from base_seed and worker_id.

    Each worker gets a unique but deterministic seed: base_seed + worker_id
    """
    global _BASE_SEED
    seed = _BASE_SEED + worker_id
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class UCF101DataModule(pl.LightningDataModule):
    """UCF101 Lightning data module.

    Expects videos under data_dir and annotation split files under annotation_dir.
    prepare_data() will attempt to download/extract both if missing.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        data_type: Literal["sequence", "video"],
        num_workers: int,
        pin_memory: bool,
        use_deterministic_worker_init: bool,
        seed: int,
        frames_per_clip: int = 16,
        step_between_clips: int = 1,
        frame_size: Optional[tuple[int, int]] = (256, 256),
        val_split_fraction: float = 0.1,
        split_fold: int = 1,
    ):
        """Initialize the UCF101DataModule.

        Args:
            data_dir: Directory to save the data
            batch_size: Batch size
            data_type: Type of data. Can be "sequence" or "video".
            num_workers: Number of workers
            pin_memory: Whether to pin memory
            use_deterministic_worker_init: Whether to use deterministic worker initialization
            seed: Seed for the data
            frames_per_clip: Number of frames per clip
            step_between_clips: Step between clips
            frame_size: Size of the frames
            val_split_fraction: Fraction of the data to use for validation
            split_fold: Fold to use for the data. Can be 1, 2 or 3.
        """
        assert data_type in ["video"], f"data_type must be 'video', got {data_type}"
        assert split_fold in [1, 2, 3], f"split_fold must be 1, 2 or 3, got {split_fold}"

        super().__init__()

        self.data_dir = data_dir
        self.annotation_dir = Path(data_dir) / "annotations"
        self.videos_dir = Path(data_dir) / "videos"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.frame_size = frame_size
        self.val_split_fraction = val_split_fraction
        self.split_fold = split_fold

        self.generator = torch.Generator().manual_seed(seed)
        self.worker_init_fn = deterministic_worker_init_fn if use_deterministic_worker_init else None

        self.input_channels = 3
        self.output_channels = 101

        self.data_type = data_type

        if self.frame_size is not None:
            self.frame_transform = transforms.Compose(
                [
                    transforms.Resize(self.frame_size),
                ]
            )
        else:
            self.frame_transform = None

        self.video_transform = transforms.Compose(
            [
                TV.ToTensorVideo(),
                TV.NormalizeVideo(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download and prepare UCF101 videos and annotation splits in-place if missing."""
        import shutil
        import tempfile
        import urllib.request
        import zipfile
        from pathlib import Path

        try:
            from tqdm import tqdm as _tqdm  # type: ignore
        except Exception:
            _tqdm = None

        def _download_with_progress(urls, filename: str, desc: str):
            import ssl

            if isinstance(urls, str):
                urls = [urls]

            def _stream(url: str, verify: bool):
                ctx = None if verify else ssl._create_unverified_context()
                opener = urllib.request.build_opener(
                    urllib.request.HTTPSHandler(context=ctx) if ctx is not None else urllib.request.HTTPSHandler()
                )
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with opener.open(req) as resp, open(filename, "wb") as out_f:
                    total = resp.length
                    if total is None:
                        total = int(resp.headers.get("Content-Length", 0)) or None
                    if _tqdm is None:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            out_f.write(chunk)
                    else:
                        bar = _tqdm(total=total, unit="B", unit_scale=True, desc=desc)
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            out_f.write(chunk)
                            bar.update(len(chunk))
                        bar.close()

            last_error = None
            for url in urls:
                try:
                    _stream(url, verify=True)
                    return
                except Exception as e1:
                    last_error = e1
                    try:
                        _stream(url, verify=False)
                        return
                    except Exception as e2:
                        last_error = e2
                        continue
            raise RuntimeError(f"Failed downloading from provided URLs. Last error: {last_error}")

        def _extract_zip_with_progress(zip_path: Path, dest_dir: Path, desc: str):
            if _tqdm is None:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(dest_dir)
                return
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.infolist()
                bar = _tqdm(total=len(members), unit="file", desc=desc)
                for m in members:
                    zf.extract(m, dest_dir)
                    bar.update(1)
                bar.close()

        videos_root = Path(self.videos_dir)
        ann_root = Path(self.annotation_dir)
        videos_root.mkdir(parents=True, exist_ok=True)
        ann_root.mkdir(parents=True, exist_ok=True)

        def has_videos(root: Path) -> bool:
            video_exts = (".avi", ".mp4", ".webm", ".mkv")
            # Any video file anywhere under root
            for ext in video_exts:
                if next(root.rglob(f"*{ext}"), None) is not None:
                    return True
            return False

        def organize_ucf101_structure(root: Path):
            import re

            video_exts = (".avi", ".mp4", ".webm", ".mkv")
            # Move top-level videos like v_ApplyEyeMakeup_g01_c01.avi into class dirs ApplyEyeMakeup/
            top_level_files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in video_exts]
            iterator = (
                _tqdm(top_level_files, desc="Organizing videos by class") if _tqdm is not None else top_level_files
            )
            for p in iterator:
                m = re.match(r"^v_([^_]+)_", p.name)
                if not m:
                    # Skip files that don't follow the expected naming
                    continue
                class_name = m.group(1)
                class_dir = root / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(p.as_posix(), (class_dir / p.name).as_posix())

        def _pair_exists_for_fold(root: Path, fold: int) -> bool:
            tfile = f"trainlist{fold:02d}.txt"
            vfile = f"testlist{fold:02d}.txt"
            return (root / tfile).is_file() and (root / vfile).is_file()

        def _detect_available_fold(root: Path) -> Optional[int]:
            for f in (1, 2, 3):
                if _pair_exists_for_fold(root, f):
                    return f
            return None

        def has_annotations(root: Path) -> bool:
            # Need classInd and at least one fold pair
            if not (root / "classInd.txt").is_file():
                return False
            return _detect_available_fold(root) is not None

        if not has_videos(videos_root):
            mirror_zip = "https://storage.googleapis.com/thumos14_files/UCF101_videos.zip"
            with tempfile.TemporaryDirectory() as td:
                tmp_zip = Path(td) / "UCF101_videos.zip"
                _download_with_progress(mirror_zip, tmp_zip.as_posix(), desc="Downloading UCF101 videos")
                _extract_zip_with_progress(tmp_zip, videos_root, desc=f"Extracting videos in {videos_root}")
            # Flatten common nested folders created by the archive
            for nested in [videos_root / "UCF-101", videos_root / "UCF101"]:
                if nested.exists() and nested.is_dir():
                    items = list(nested.iterdir())
                    iterator = _tqdm(items, desc=f"Flattening {nested.name}") if _tqdm is not None else items
                    for p in iterator:
                        shutil.move(p.as_posix(), (videos_root / p.name).as_posix())
                    try:
                        nested.rmdir()
                    except OSError:
                        pass
            # Ensure class-separated structure
            organize_ucf101_structure(videos_root)
            if not has_videos(videos_root):
                raise RuntimeError("UCF101 videos not found after extraction.")

        if not has_annotations(ann_root):
            splits_zip = "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
            with tempfile.TemporaryDirectory() as td:
                tmp_zip = Path(td) / "UCF101TrainTestSplits-RecognitionTask.zip"
                _download_with_progress(splits_zip, tmp_zip.as_posix(), desc="Downloading UCF101 splits")
                _extract_zip_with_progress(tmp_zip, ann_root, desc=f"Extracting annotations in {ann_root}")

            nested = ann_root / "ucfTrainTestlist"
            if nested.exists() and nested.is_dir():
                items = list(nested.iterdir())
                iterator = _tqdm(items, desc="Organizing annotations") if _tqdm is not None else items
                for p in iterator:
                    if p.is_file():
                        shutil.move(p.as_posix(), (ann_root / p.name).as_posix())
                try:
                    nested.rmdir()
                except OSError:
                    pass
            if not has_annotations(ann_root):
                raise RuntimeError("UCF101 annotation files not found after extraction.")

        # Decide on fold
        if self.split_fold is None:
            detected = _detect_available_fold(ann_root)
            if detected is None:
                raise RuntimeError(
                    "Could not detect any available fold (trainlistXX/testlistXX). Please provide fold files."
                )
            self.split_fold = detected
        else:
            # Validate explicit fold exists
            if not _pair_exists_for_fold(ann_root, self.split_fold):
                raise RuntimeError(
                    f"Requested split_fold={self.split_fold} but corresponding files are missing in {ann_root}."
                )

        self.annotation_dir = str(ann_root)

    def setup(self, stage=None):
        """Function to setup the datamodule."""
        # we set up only relevant datamodules when stage is specified
        if stage == "fit" or stage is None:
            full_train = datasets.UCF101(
                root=self.videos_dir,
                annotation_path=self.annotation_dir,
                frames_per_clip=self.frames_per_clip,
                step_between_clips=self.step_between_clips,
                fold=self.split_fold,
                train=True,
                num_workers=self.num_workers,
                transform=self.video_transform,
                frame_rate=None,
            )
            num_full = len(full_train)
            num_val = max(1, int(num_full * self.val_split_fraction))
            num_train = num_full - num_val
            self.train_dataset, self.val_dataset = random_split(
                full_train, [num_train, num_val], generator=self.generator
            )

        if stage == "test" or stage is None:
            self.test_dataset = datasets.UCF101(
                root=self.videos_dir,
                annotation_path=self.annotation_dir,
                frames_per_clip=self.frames_per_clip,
                step_between_clips=self.step_between_clips,
                fold=self.split_fold,
                train=False,
                num_workers=self.num_workers,
                transform=self.video_transform,
                frame_rate=None,
            )

    def _build_loader(self, dataset, shuffle: bool, drop_last: bool = False):
        """Function to create dataloaders given a dataset and a few arguments.

        Reused for train, val and test dataloaders.

        Args:
            dataset: Dataset to create a dataloader for.
            shuffle: Whether to shuffle the dataset.
            drop_last: Whether to drop the last batch if it's not complete.

        Returns:
            DataLoader: DataLoader for the dataset.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            # worker_init_fn=self.worker_init_fn,  # No longer needed with pl.seed_everything(workers=True)
            generator=self.generator,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._ignore_audio_collate_fn,
        )

    def train_dataloader(self):
        """Function to create the train dataloader."""
        return self._build_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self):
        """Function to create the validation dataloader."""
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        """Function to create the test dataloader."""
        return self._build_loader(self.test_dataset, shuffle=False)

    def _ignore_audio_collate_fn(self, samples):
        """Collate that ignores audio and stacks videos and labels.

        Samples are shaped as (video, audio, label).

        Returns (videos [batch_size, ...], labels [batch_size]).
        """
        # Keep only valid (video, label)
        videos, _audios, labels = zip(*samples)
        return torch.stack(list(videos), dim=0), torch.as_tensor(labels, dtype=torch.long)

    @torch.no_grad()
    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Function to reshape (if needed) the frames before batch transfer."""
        # Must reshape the frames
        if self.frame_transform is not None:
            video, label = batch
            print(f"video: {video.shape}, label: {label}")
            batch_size, channels, temporal_length, height, width = video.shape
            video = rearrange(video, "b c t h w -> (b c t) h w")
            # Apply frame transform
            video = self.frame_transform(video)
            # Reconstruct to original shape
            video = rearrange(video, "(b c t) h w -> b t h w c", b=batch_size, t=temporal_length, c=channels)
            print(f"reshaped video: {video.shape}, label: {label}")
            return (video, label)
        else:
            return batch
