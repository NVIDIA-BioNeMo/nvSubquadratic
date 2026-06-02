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

"""Extract SIREN kernel + mask data from a trained checkpoint for visualization.

Loads a config + checkpoint, evaluates the continuous kernels and masks on the
correct grid, and writes a compressed ``.npz`` file with raw numpy arrays.

Output format (``np.load(path)``):
    grid:                  [K_h, K_w, data_dim]   — shared grid coordinates
    channel_indices:       [n_sampled]             — which channels were sampled
    block_{i}_kernel:      [K_h, K_w, n_sampled]   — raw SIREN kernel
    block_{i}_mask:        [K_h, K_w, n_sampled]   — mask values (1 = pass-through)
    block_{i}_masked:      [K_h, K_w, n_sampled]   — kernel * mask
    num_blocks, hidden_dim, kernel_size_h, kernel_size_w  — scalar metadata

Usage (inside container or with nv-subq env):
    PYTHONPATH=. python scripts/extract_kernel_data.py \\
        --config examples/well/euler_multi_quadrants_periodicBC/cfg_hyena_gaussian_mask.py \\
        --checkpoint runs/<run_dir>/checkpoints/last.ckpt \\
        --output tmp/kernel_data/gmask.npz \\
        --spatial-dims 32 32
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from experiments.utils.cli import apply_config_overrides, load_config_from_file
from nvsubquadratic.lazy_config import instantiate


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract kernel data from checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config .py file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--num-channels", type=int, default=16, help="Number of channels to sample per block")
    parser.add_argument(
        "--spatial-dims",
        type=int,
        nargs="+",
        required=True,
        help="Spatial dimensions of the input (after patching), e.g. 32 32 for 2D",
    )
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides (key=value)")
    return parser.parse_args()


@torch.no_grad()
def extract_npz(
    config, checkpoint_path: str, spatial_dims: tuple[int, ...], num_sample_channels: int
) -> dict[str, np.ndarray]:
    """Load model from config + checkpoint and extract kernel/mask arrays.

    Returns a dict of numpy arrays suitable for ``np.savez_compressed``.
    """
    from nvsubquadratic.modules.ckconv_nd import CKConvND

    network = instantiate(config.net)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", {})

    # Strip Lightning wrapper prefix and torch.compile prefix
    cleaned_sd = {}
    for k, v in sd.items():
        clean_k = k
        for prefix in ["network._orig_mod.", "network."]:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix) :]
                break
        cleaned_sd[clean_k] = v

    missing, unexpected = network.load_state_dict(cleaned_sd, strict=False)
    if missing:
        print(f"[warn] Missing keys: {len(missing)} (first 5: {missing[:5]})")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")

    network.eval()

    arrays: dict[str, np.ndarray] = {}
    block_ids = []

    for name, module in network.named_modules():
        if not isinstance(module, CKConvND):
            continue

        block_match = re.search(r"blocks\.(\d+)", name)
        block_id = int(block_match.group(1)) if block_match else -1
        block_ids.append(block_id)
        ckconv = module

        # Compute grid_lens (same logic as CKConvND.forward)
        if ckconv.grid_type == "single":
            grid_lens = tuple((s + 1) // 2 for s in spatial_dims)
        else:
            grid_lens = tuple(spatial_dims)

        kernel_size = tuple(2 * gl - 1 for gl in grid_lens)
        print(f"Block {block_id} ({name}): grid_lens={grid_lens}, kernel_size={kernel_size}")

        # Evaluate SIREN kernel
        conv_kernel, grid = ckconv.kernel(grid_lens)
        # conv_kernel: [1, *kernel_size, hidden_dim], grid: [1, *kernel_size, data_dim]

        # Evaluate mask on ones to get mask-only values
        ones = torch.ones_like(conv_kernel)
        if not isinstance(ckconv.mask, torch.nn.Identity):
            mask_values = ckconv.mask(grid=grid, x=ones)
            masked_kernel = ckconv.mask(grid=grid, x=conv_kernel)
        else:
            mask_values = ones
            masked_kernel = conv_kernel

        # Squeeze batch dim → [*kernel_size, hidden_dim]
        conv_kernel_np = conv_kernel.squeeze(0).float().numpy()
        mask_np = mask_values.squeeze(0).float().numpy()
        masked_np = masked_kernel.squeeze(0).float().numpy()

        if "grid" not in arrays:
            arrays["grid"] = grid.squeeze(0).float().numpy()

        hidden_dim = conv_kernel_np.shape[-1]
        if num_sample_channels >= hidden_dim:
            ch_idx = np.arange(hidden_dim)
        else:
            ch_idx = np.linspace(0, hidden_dim - 1, num_sample_channels, dtype=int)

        # Store sampled channels: [K_h, K_w, n_sampled]
        arrays[f"block_{block_id}_kernel"] = conv_kernel_np[..., ch_idx]
        arrays[f"block_{block_id}_mask"] = mask_np[..., ch_idx]
        arrays[f"block_{block_id}_masked"] = masked_np[..., ch_idx]

    # Metadata
    arrays["channel_indices"] = ch_idx
    arrays["block_ids"] = np.array(sorted(block_ids))
    arrays["kernel_size"] = np.array(kernel_size)
    arrays["spatial_dims"] = np.array(spatial_dims)
    arrays["hidden_dim"] = np.array(hidden_dim)

    return arrays


def main():
    args = parse_args()

    config = load_config_from_file(args.config)
    if args.overrides:
        config = apply_config_overrides(config, args.overrides)

    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")

    arrays = extract_npz(config, args.checkpoint, tuple(args.spatial_dims), args.num_channels)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **arrays)

    size_mb = out_path.stat().st_size / 1024 / 1024
    n_blocks = len(arrays["block_ids"])
    print(f"Wrote {out_path} ({size_mb:.1f} MB, {n_blocks} blocks)")


if __name__ == "__main__":
    main()
