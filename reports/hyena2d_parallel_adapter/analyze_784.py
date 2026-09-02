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

"""Displacement-bucketed evaluation of the 784-token frozen-backbone arms.

What: for each arm (a / c0 / c1 / c2s / c2f) and each seed, load the best
checkpoint and evaluate on the random-placement test set, bucketing samples by
the L2 distance from the digit's top-left corner to the readout corner.  Also
extracts per-layer ``norm(W_up)`` from the adapter arms.

The decision rule under test (from the experiment brief): an inductive-bias gain
shows up as a margin that GROWS with displacement.  A uniform margin across all
buckets is extra capacity, not inductive bias.

Hardware: one GPU.

Invoke::

    PYTHONPATH=. python reports/hyena2d_parallel_adapter/analyze_784.py

Output: ``buckets_784.json`` + ``wup_norms_784.json`` next to this script.
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch

from examples.hyena2d_parallel_adapter import _base_lg as B
from experiments.datamodules.mnist import MNISTDataModule
from experiments.datamodules.spatial_recall_classification import (
    SpatialRecallClassificationDataModule,
)
from nvsubquadratic.lazy_config import LazyConfig, instantiate


ARMS = ["a", "c0", "c1", "c2s", "c2f"]
SEEDS = [0, 1, 2]
N_BUCKETS = 4


def find_ckpt(arm: str, seed: int) -> str | None:
    """Locate the best checkpoint for one (arm, seed) run.

    Args:
        arm: Arm identifier.
        seed: Seed used for the run.

    Returns:
        Path to the checkpoint, or ``None`` if the run directory is missing.
    """
    # The run_path override contains slashes, so Lightning nests the run dir:
    #   runs/FA_..._lg_arm_<arm>_run_path_<entity>/<project>/<id>_seed_<n>_<ts>/checkpoints/
    pat = f"runs/*lg_arm_{arm}_run_path_*/**/*seed_{seed}_*/checkpoints/*.ckpt"
    hits = sorted(glob.glob(pat, recursive=True))
    if not hits:
        return None
    # Prefer an epoch-tagged (best) checkpoint over last.ckpt.
    best = [h for h in hits if "epoch=" in Path(h).name]
    return best[-1] if best else hits[-1]


def load_net(arm: str, ckpt: str) -> torch.nn.Module:
    """Instantiate an arm's network and load its trained weights.

    Args:
        arm: Arm identifier, selecting the architecture.
        ckpt: Path to the Lightning checkpoint.

    Returns:
        The network in eval mode on CUDA.
    """
    net = instantiate(B.make_net_cfg(arm)).cuda().eval()
    sd = torch.load(ckpt, map_location="cuda", weights_only=False)["state_dict"]
    sd = {k.replace("network.", "", 1): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"    [warn] {arm}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return net


def build_test_loader(seed: int = 42):
    """Return the random-placement test dataloader and its datamodule."""
    dm = SpatialRecallClassificationDataModule(
        base_datamodule_cfg=LazyConfig(MNISTDataModule)(
            data_dir=".data/mnist",
            batch_size=256,
            data_type="image",
            num_workers=4,
            pin_memory=True,
            use_deterministic_worker_init=False,
            seed=seed,
            task="classification",
        ),
        target_size=B.TARGET_SIZE,
        canvas_size=B.CANVAS_SIZE,
        placement="random",
        with_mask=False,
        readout_value=B.READOUT_VALUE,
        batch_size=256,
        num_workers=4,
        pin_memory=True,
        seed=seed,
    )
    dm.setup("test")
    return dm


def digit_offsets(canvas: torch.Tensor) -> torch.Tensor:
    r"""Recover each sample's digit top-left ``(y0, x0)`` from the canvas.

    The canvas is zero-filled; the pasted MNIST crop carries its own normalised
    background (``(0 - 0.1307)/0.3081 = -0.424``), so the digit's bounding box is
    exactly the set of non-zero pixels once the constant readout region is
    excluded.

    Args:
        canvas: Batch of canvases ``[B, 1, H, W]``.

    Returns:
        Long tensor ``[B, 2]`` of ``(y0, x0)`` offsets.
    """
    a = canvas[:, 0].clone()
    a[:, B.CANVAS_SIZE - B.TARGET_SIZE :, B.CANVAS_SIZE - B.TARGET_SIZE :] = 0.0
    nz = a.abs() > 1e-6
    ys = torch.where(nz.any(dim=2), torch.arange(a.shape[1], device=a.device), a.shape[1])
    xs = torch.where(nz.any(dim=1), torch.arange(a.shape[2], device=a.device), a.shape[2])
    return torch.stack([ys.min(dim=1).values, xs.min(dim=1).values], dim=1)


@torch.no_grad()
def evaluate(net: torch.nn.Module, dm) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over the test set, returning correctness and displacement.

    Args:
        net: Network in eval mode.
        dm: Datamodule providing ``test_dataloader`` and the batch transform.

    Returns:
        ``(correct [N], displacement [N])`` as numpy arrays.
    """
    corner = float(B.CANVAS_SIZE - B.TARGET_SIZE)
    correct, dist = [], []
    for batch in dm.test_dataloader():
        canvas, label = batch
        off = digit_offsets(canvas.cuda()).float()
        d = torch.sqrt((corner - off[:, 0]) ** 2 + (corner - off[:, 1]) ** 2)
        b = dm.on_before_batch_transfer(batch, 0)
        logits = net({"input": b["input"].cuda(), "condition": None})["logits"]
        correct.append((logits.argmax(1) == b["label"].cuda()).cpu().numpy())
        dist.append(d.cpu().numpy())
    return np.concatenate(correct), np.concatenate(dist)


def wup_norms(net: torch.nn.Module) -> dict[int, float]:
    """Extract per-block Frobenius norm of ``W_up``.

    Args:
        net: Network with parallel adapters attached.

    Returns:
        Mapping from block index to ``norm(W_up)``; empty for non-adapter arms.
    """
    out = {}
    for name, p in net.named_parameters():
        if name.endswith("w_up.weight"):
            m = re.search(r"blocks\.(\d+)", name)
            if m:
                out[int(m.group(1))] = float(p.data.norm())
    return out


def main(output_dir: Path) -> None:
    """Evaluate every arm/seed and write bucket + W_up JSON summaries."""
    dm = build_test_loader()

    # Fixed edges in PATCH units, chosen to bracket the local-conv receptive
    # field.  Six blocks of 3x3 depthwise conv reach ~1 + 6*2 = 13 patches, so a
    # percentile split (whose first edge lands near 14 patches) would put every
    # sample at or beyond C1's reach and could not show a crossover.  These edges
    # put two buckets strictly inside that radius.
    P = B.PATCH_SIZE
    edges = [0.0, 6 * P, 12 * P, 20 * P, 1e9]  # 0-6, 6-12, 12-20, 20+ patches
    print(f"bucket edges (patches): 0-6, 6-12, 12-20, 20+   (px: {[round(e) for e in edges[:-1]]}..inf)")
    print(f"C1 receptive field after {B.NUM_BLOCKS} blocks of 3x3: ~{1 + 2 * B.NUM_BLOCKS} patches\n")

    buckets: dict = {}
    wups: dict = {}
    raw: dict = {}

    for arm in ARMS:
        buckets[arm] = {"overall": [], "per_bucket": []}
        for seed in SEEDS:
            ck = find_ckpt(arm, seed)
            if ck is None:
                print(f"  {arm} seed{seed}: NO CHECKPOINT")
                continue
            net = load_net(arm, ck)
            corr, dist = evaluate(net, dm)
            per = []
            for i in range(N_BUCKETS):
                m = (dist >= edges[i]) & (dist < edges[i + 1])
                per.append(float(corr[m].mean()) if m.sum() else float("nan"))
            buckets[arm]["overall"].append(float(corr.mean()))
            buckets[arm]["per_bucket"].append(per)
            # Persist raw per-sample data so bucket edges can be revisited
            # without re-running inference.
            raw.setdefault(arm, {})[str(seed)] = {
                "correct": corr.astype(np.uint8).tolist(),
                "dist_px": np.round(dist, 2).tolist(),
            }
            if seed == 0:
                w = wup_norms(net)
                if w:
                    wups[arm] = w
            print(f"  {arm:<4} seed{seed}  overall={corr.mean():.4f}  buckets={['%.3f' % p for p in per]}")
            del net
            torch.cuda.empty_cache()

    (output_dir / "buckets_784.json").write_text(
        json.dumps({"edges": edges, "n_buckets": N_BUCKETS, "arms": buckets}, indent=2)
    )
    (output_dir / "wup_norms_784.json").write_text(json.dumps(wups, indent=2))
    (output_dir / "raw_784.json").write_text(json.dumps(raw))
    print(f"\nwrote {output_dir / 'buckets_784.json'} and {output_dir / 'wup_norms_784.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    main(a.output_dir)
