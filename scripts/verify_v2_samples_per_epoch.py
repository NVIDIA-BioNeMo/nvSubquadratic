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

"""Verify that SAMPLES_PER_EPOCH and channel counts in v2 _base.py configs
match what the_well.data.datasets.WellDataset actually reports.

CPU-only — no GPU required. Run with:
    conda run -n nv-subq python scripts/verify_v2_samples_per_epoch.py
"""

import importlib
import os
import sys
import traceback


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASETS = [
    "acoustic_scattering_maze",
    "active_matter",
    "euler_multi_quadrants_periodicBC",
    "gray_scott_reaction_diffusion",
    "helmholtz_staircase",
    "MHD_64",
    "rayleigh_benard",
    "rayleigh_taylor_instability",
    "shear_flow",
    "supernova_explosion_64",
    "turbulence_gravity_cooling",
    "turbulent_radiative_layer_2D",
    "turbulent_radiative_layer_3D",
    "viscoelastic_instability",
]

WELL_BASE_PATH = "/shared/data/image_datasets/the_well/datasets"


def main():
    from the_well.data.datasets import WellDataset

    passed = 0
    failed = 0
    errors = 0

    for ds_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"  {ds_name}")
        print(f"{'=' * 60}")

        # Import the _base module
        mod = importlib.import_module(f"examples.well.v2.{ds_name}._base")
        expected_spe = mod.SAMPLES_PER_EPOCH
        n_steps_input = mod.N_STEPS_INPUT
        n_steps_output = mod.N_STEPS_OUTPUT
        in_channels = mod.IN_CHANNELS
        out_channels = mod.OUT_CHANNELS
        well_dataset_name = mod.WELL_DATASET_NAME
        n_fields = mod.N_FIELDS
        n_constant_fields = mod.N_CONSTANT_FIELDS

        try:
            train_dataset = WellDataset(
                well_base_path=WELL_BASE_PATH,
                well_dataset_name=well_dataset_name,
                well_split_name="train",
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                use_normalization=False,
                return_grid=False,
                cache_small=False,
            )
        except Exception:
            print("  ERROR: Could not instantiate WellDataset")
            traceback.print_exc()
            errors += 1
            continue

        actual_len = len(train_dataset)
        spe_ok = actual_len == expected_spe

        # Check metadata field counts
        meta = train_dataset.metadata
        actual_n_fields = meta.n_fields
        actual_n_constant = meta.n_constant_fields

        fields_ok = actual_n_fields == n_fields
        constants_ok = actual_n_constant == n_constant_fields

        # Derived channel check
        expected_in = n_steps_input * actual_n_fields + actual_n_constant
        expected_out = actual_n_fields
        in_ok = in_channels == expected_in
        out_ok = out_channels == expected_out

        status = "PASS" if (spe_ok and fields_ok and constants_ok and in_ok and out_ok) else "FAIL"

        print(
            f"  SAMPLES_PER_EPOCH: config={expected_spe:>10,}  actual={actual_len:>10,}  {'OK' if spe_ok else 'MISMATCH!'}"
        )
        print(
            f"  N_FIELDS:          config={n_fields:>3}         actual={actual_n_fields:>3}         {'OK' if fields_ok else 'MISMATCH!'}"
        )
        print(
            f"  N_CONSTANT_FIELDS: config={n_constant_fields:>3}         actual={actual_n_constant:>3}         {'OK' if constants_ok else 'MISMATCH!'}"
        )
        print(
            f"  IN_CHANNELS:       config={in_channels:>3}         expected={expected_in:>3}         {'OK' if in_ok else 'MISMATCH!'}"
        )
        print(
            f"  OUT_CHANNELS:      config={out_channels:>3}         expected={expected_out:>3}         {'OK' if out_ok else 'MISMATCH!'}"
        )
        print(f"  → {status}")

        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY: {passed} passed, {failed} failed, {errors} errors (out of {len(DATASETS)})")
    print(f"{'=' * 60}")

    sys.exit(1 if (failed > 0 or errors > 0) else 0)


if __name__ == "__main__":
    main()
