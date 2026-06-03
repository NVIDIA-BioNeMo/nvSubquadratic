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

"""Generate samples from a trained diffusion model and compute CleanFID."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image

from experiments.utils.checkpointing import download_checkpoint, load_checkpoint_state_dict
from experiments.utils.cli import (
    apply_config_overrides,
    load_config_from_file,
    verify_no_interpolator_overwrites,
)
from nvsubquadratic.lazy_config import instantiate
from nvsubquadratic.metrics.cleanfid import compute_folder_fid


# Set high precision for matrix multiplication (tensor cores)
torch.set_float32_matmul_precision("high")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline CleanFID evaluation helper.")
    parser.add_argument("--config", required=True, help="Path to the experiment config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .ckpt file.")
    parser.add_argument(
        "--wandb-run-path",
        type=str,
        default=None,
        help="Optional W&B run path (entity/project/run_id) to download a checkpoint from.",
    )
    parser.add_argument(
        "--wandb-alias",
        type=str,
        default="best",
        help="Checkpoint alias to download when --wandb-run-path is provided.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50_000,
        help="Total number of samples to generate before computing CleanFID.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=250,
        help="Number of samples to draw per diffusion pass.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the scheduler inference steps used while sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated PNG samples will be written.",
    )
    parser.add_argument(
        "--fid-dataset-name",
        type=str,
        default="imagenet2012",
        help="CleanFID reference dataset name.",
    )
    parser.add_argument(
        "--fid-dataset-res",
        type=int,
        default=64,
        help="Resolution of the CleanFID reference statistics.",
    )
    parser.add_argument(
        "--fid-dataset-split",
        type=str,
        default="train",
        help="Dataset split to use for CleanFID reference statistics.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Force EMA weights for sampling if the checkpoint contains them.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional config overrides in key=value format.",
    )
    return parser.parse_args()


def _extract_example_shape(datamodule: Any) -> tuple[int, int, int]:
    if hasattr(datamodule, "num_workers"):
        setattr(datamodule, "num_workers", 0)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))
    if isinstance(batch, dict):
        example = batch["input"]
    elif isinstance(batch, (list, tuple)):
        example = batch[0]
    else:
        raise ValueError("Unsupported batch structure while deriving example shape.")
    if not torch.is_tensor(example):
        raise TypeError("Datamodule inputs must be tensors.")
    shape = tuple(example.shape[1:])
    if len(shape) != 3:
        raise ValueError(f"Expected image-like inputs with 3 dims; got {shape}.")
    return shape  # type: ignore[return-value]


def _resolve_checkpoint_path(args: argparse.Namespace) -> str:
    if bool(args.checkpoint) == bool(args.wandb_run_path):
        raise ValueError("Provide exactly one of --checkpoint or --wandb-run-path.")
    if args.checkpoint:
        return args.checkpoint
    return download_checkpoint(run_path=args.wandb_run_path, alias=args.wandb_alias)


def _set_example_shape(model: torch.nn.Module, shape: tuple[int, int, int]) -> None:
    if not hasattr(model, "example_input_shape"):
        raise AttributeError("Model does not expose example_input_shape.")
    model.example_input_shape = torch.Size(shape)  # type: ignore[attr-defined]


def _save_sample_batch(samples: torch.Tensor, output_dir: Path, start_idx: int) -> None:
    samples = samples.detach().cpu()
    samples = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)
    samples = samples.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
    if samples.shape[1] == 1:
        samples = samples.repeat(1, 3, 1, 1)
    for offset, tensor in enumerate(samples):
        save_image(tensor, output_dir / f"{start_idx + offset:06d}.png")


def main() -> None:
    args = _parse_args()
    ckpt_path = _resolve_checkpoint_path(args)

    config = load_config_from_file(args.config)
    verify_no_interpolator_overwrites(config, args.overrides)
    config = apply_config_overrides(config, args.overrides)
    if getattr(config, "diffusion", None) is None:
        raise ValueError("Selected config does not define diffusion settings.")

    # Override batch size to match the requested sample batch size for efficiency
    if hasattr(config.dataset, "batch_size"):
        config.dataset.batch_size = args.sample_batch_size

    datamodule = instantiate(config.dataset)
    example_shape = _extract_example_shape(datamodule)
    if hasattr(datamodule, "teardown"):
        datamodule.teardown(stage="fit")

    network = instantiate(
        config.net,
        in_channels=getattr(datamodule, "input_channels", None),
        out_channels=getattr(datamodule, "output_channels", None),
    )
    model = instantiate(config.lightning_wrapper_class, network=network, cfg=config)
    _set_example_shape(model, example_shape)

    # Optional compilation
    if getattr(config, "compile", False):
        print("[fid] Compiling model with torch.compile...")
        model.network = torch.compile(model.network)

    state_dict = load_checkpoint_state_dict(ckpt_path)
    load_msg = model.load_state_dict(state_dict, strict=False)
    if load_msg.missing_keys:
        print(f"[load] Missing keys: {len(load_msg.missing_keys)}")
    if load_msg.unexpected_keys:
        print(f"[load] Unexpected keys: {len(load_msg.unexpected_keys)}")
    if args.use_ema and getattr(model, "ema_enabled", False) and getattr(model, "_ema_model", None) is not None:
        model._ema_has_been_updated = True  # type: ignore[attr-defined]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    inference_steps = args.num_inference_steps or getattr(config.diffusion, "num_inference_steps", None)
    if inference_steps is None:
        raise ValueError("Unable to determine num_inference_steps for sampling.")

    print(f"[fid] Generating {args.num_samples} samples into {output_dir} ...")

    # Determine precision for autocast
    precision_str = getattr(config.train, "precision", "32-true")
    if precision_str in ["bf16-mixed", "bf16"]:
        autocast_dtype = torch.bfloat16
    elif precision_str in ["16-mixed", "fp16"]:
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    # Use the training dataloader to ensure we sample labels from the correct distribution
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()
    iterator = iter(loader)

    with torch.inference_mode(), torch.autocast("cuda", dtype=autocast_dtype, enabled=autocast_dtype != torch.float32):
        while total < args.num_samples:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            # Handle different batch structures
            if isinstance(batch, dict):
                # We consume as many labels as we need to fill the batch or finish the quota
                current_labels = batch.get("label")
                if current_labels is not None:
                    current_labels = current_labels.to(device)
            elif isinstance(batch, (list, tuple)) and len(batch) > 1:
                # Assumption: (data, label) tuple
                current_labels = batch[1].to(device)
            else:
                current_labels = None

            # Determine how many samples to generate in this pass
            remaining = args.num_samples - total

            # If we have labels, use their batch size (capped by remaining)
            # If no labels, use sample-batch-size (capped by remaining)
            if current_labels is not None:
                current = min(current_labels.shape[0], remaining)
                # Slice labels to match the number of samples we are generating
                batch_labels = current_labels[:current]
            else:
                current = min(args.sample_batch_size, remaining)
                batch_labels = None

            samples = model.sample(num_samples=current, num_inference_steps=inference_steps, labels=batch_labels)
            _save_sample_batch(samples, output_dir, total)
            total += current
            print(f"\r[sampling] {total}/{args.num_samples} images complete", end="")
    print("\n[sampling] Done.")

    score = compute_folder_fid(
        output_dir,
        dataset_name=args.fid_dataset_name,
        dataset_resolution=args.fid_dataset_res,
        dataset_split=args.fid_dataset_split,
    )
    print(f"[fid] CleanFID score: {score:.4f}")


if __name__ == "__main__":
    main()
