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

"""Cross-check the VRMSE metric implementation against manual computation.

Loads a checkpoint, runs inference on test data, and computes VRMSE
via multiple independent paths (Lightning callback, manual numpy,
per-channel reduction) to detect implementation drift.  Run after any
change to the WELL regression wrapper or its loss / metric code.

Targets: H100 SXM 80GB (or any Ampere+ GPU); needs the matching WELL
dataset on disk.

Usage:
    PYTHONPATH=. conda run -n nv-subq python \\
        benchmarks/well/verify_vrmse.py --checkpoint <path>

Output: stdout (per-method VRMSE values + per-channel diffs).
"""

import argparse
import sys
from pathlib import Path

import torch
from einops import rearrange


# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    """Load a checkpoint and compute per-batch VRMSE against the test split."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-batches", type=int, default=5)
    parser.add_argument("--overrides", nargs="*", default=[])
    args = parser.parse_args()

    torch.multiprocessing.set_sharing_strategy("file_system")

    # 1. Load config and build model
    from experiments.utils.cli import apply_config_overrides, load_config_from_file

    config = load_config_from_file(args.config)
    if args.overrides:
        config = apply_config_overrides(config, args.overrides)

    # 2. Build datamodule
    from nvsubquadratic.lazy_config import instantiate

    datamodule = instantiate(config.dataset)
    datamodule.prepare_data()
    datamodule.setup("test")

    metadata = datamodule.metadata
    normalization = datamodule.normalization

    print(f"Dataset: {datamodule.well_dataset_name}")
    print(f"n_fields: {metadata.n_fields}")
    print(f"n_spatial_dims: {metadata.n_spatial_dims}")
    print(f"Normalization: {normalization is not None}")
    print()

    # 3. Build and load model
    net = instantiate(config.net)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    # Strip "network." prefix from Lightning state dict
    cleaned = {}
    for k, v in state.items():
        if k.startswith("network."):
            cleaned[k[len("network.") :]] = v
        else:
            cleaned[k] = v
    # Handle compiled model prefix
    cleaned2 = {}
    for k, v in cleaned.items():
        if k.startswith("_orig_mod."):
            cleaned2[k[len("_orig_mod.") :]] = v
        else:
            cleaned2[k] = v
    net.load_state_dict(cleaned2)
    net = net.cuda().eval()

    # 4. Get test dataloader
    test_loader = datamodule.test_dataloader()

    # 5. Import WELL metrics directly
    from the_well.benchmark.metrics.spatial import MSE as MSE_metric
    from the_well.benchmark.metrics.spatial import VRMSE as VRMSE_metric

    vrmse_fn = VRMSE_metric()
    mse_fn = MSE_metric()

    # 6. Run inference and compute metrics
    all_vrmse_denorm = []
    all_vrmse_norm = []
    all_vrmse_manual = []
    all_mse_denorm = []

    print(f"Running on {args.num_batches} test batches...")
    print()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= args.num_batches:
                break

            input_fields = batch["input_fields"].cuda()  # [B, T, H, W, C]
            output_fields = batch["output_fields"].cuda()  # [B, 1, H, W, C]
            target = output_fields[:, 0]  # [B, H, W, C]

            # Build model input (same as our wrapper)
            model_input = rearrange(input_fields, "b t h w c -> b h w (t c)")
            if "constant_fields" in batch:
                model_input = torch.cat([model_input, batch["constant_fields"].cuda()], dim=-1)

            # Forward pass — network expects dict input like our Lightning wrapper
            pred = net({"input": model_input, "condition": None})
            if isinstance(pred, dict):
                pred = pred["logits"]

            # pred and target are in NORMALIZED space
            pred_5d = pred.unsqueeze(1)  # [B, 1, H, W, C]
            target_5d = target.unsqueeze(1)  # [B, 1, H, W, C]

            # --- Method A: VRMSE on NORMALIZED data ---
            vrmse_norm = vrmse_fn(pred_5d, target_5d, metadata)
            vrmse_norm_mean = vrmse_norm.mean().item()

            # --- Method B: VRMSE on DENORMALIZED data ---
            if normalization is not None:
                pred_denorm = normalization.denormalize_flattened(pred_5d, "variable")
                target_denorm = normalization.denormalize_flattened(target_5d, "variable")
            else:
                pred_denorm = pred_5d
                target_denorm = target_5d

            vrmse_denorm = vrmse_fn(pred_denorm, target_denorm, metadata)
            vrmse_denorm_mean = vrmse_denorm.mean().item()

            # --- Method C: Manual VRMSE computation ---
            # VRMSE = sqrt(MSE / var(target)) where MSE and var are over spatial dims
            spatial_dims = tuple(range(-metadata.n_spatial_dims - 1, -1))  # (-3, -2) for 2D
            mse_manual = torch.mean((pred_denorm - target_denorm) ** 2, dim=spatial_dims)
            var_manual = torch.var(target_denorm, dim=spatial_dims)
            vrmse_manual = torch.sqrt(mse_manual / (var_manual + 1e-7))
            vrmse_manual_mean = vrmse_manual.mean().item()

            # --- MSE for sanity ---
            mse_denorm = mse_fn(pred_denorm, target_denorm, metadata)
            mse_denorm_mean = mse_denorm.mean().item()

            all_vrmse_norm.append(vrmse_norm_mean)
            all_vrmse_denorm.append(vrmse_denorm_mean)
            all_vrmse_manual.append(vrmse_manual_mean)
            all_mse_denorm.append(mse_denorm_mean)

            print(
                f"Batch {batch_idx}: "
                f"VRMSE(norm)={vrmse_norm_mean:.6f}  "
                f"VRMSE(denorm)={vrmse_denorm_mean:.6f}  "
                f"VRMSE(manual)={vrmse_manual_mean:.6f}  "
                f"MSE(denorm)={mse_denorm_mean:.6f}"
            )

            # Per-field breakdown
            if batch_idx == 0:
                print("\n  Per-field VRMSE (denorm), batch 0:")
                field_names = [f"field_{i}" for i in range(vrmse_denorm.shape[-1])]
                for fi in range(vrmse_denorm.shape[-1]):
                    v = vrmse_denorm[:, :, fi].mean().item()
                    print(f"    {field_names[fi]}: {v:.6f}")

                print("\n  Per-field target std (denorm):")
                for fi in range(target_denorm.shape[-1]):
                    std = target_denorm[..., fi].std().item()
                    print(f"    {field_names[fi]}: std={std:.6f}")
                print()

    # Summary
    import numpy as np

    print("\n" + "=" * 70)
    print("SUMMARY (averaged over batches)")
    print("=" * 70)
    print(f"  VRMSE (normalized space):   {np.mean(all_vrmse_norm):.6f}")
    print(f"  VRMSE (denormalized space): {np.mean(all_vrmse_denorm):.6f}")
    print(f"  VRMSE (manual denorm):      {np.mean(all_vrmse_manual):.6f}")
    print(f"  MSE (denormalized):         {np.mean(all_mse_denorm):.6f}")
    print()
    print(f"  norm vs denorm diff: {abs(np.mean(all_vrmse_norm) - np.mean(all_vrmse_denorm)):.8f}")
    print(f"  denorm vs manual diff: {abs(np.mean(all_vrmse_denorm) - np.mean(all_vrmse_manual)):.8f}")
    print()
    print("  Logged test/VRMSE from wandb: 0.04036 (for CNextU Euler)")
    print("  Paper Table 2 CNextU Euler:   0.1531")


if __name__ == "__main__":
    main()
