#!/usr/bin/env python
"""Standalone Test-Time Training (TTT) evaluation for ARC-AGI.

Matches the VARC reference protocol:
  - Full model fine-tuning (all parameters, not just the task token)
  - Per-task deepcopy so no cross-task weight contamination
  - Random translation + resolution augmentation on every TTT gradient step
  - Multiple augmented inference passes with pixel-wise majority voting
  - Cosine LR schedule with linear warmup
  - Gradient clipping (max_norm=1.0)

Usage
-----
# Full evaluation (400 tasks, 100 TTT steps, 2 attempts, 10 inference passes):
  PYTHONPATH=. python scripts/evaluation/eval_arc_ttt.py \\
      --config examples/arc/cfg_vit_rearc.py \\
      --checkpoint /path/to/epoch=469-step=757640.ckpt

# Smoke test (3 tasks, 5 TTT steps — quick sanity check):
  PYTHONPATH=. python scripts/evaluation/eval_arc_ttt.py \\
      --config examples/arc/cfg_vit_rearc.py \\
      --checkpoint /path/to/epoch=469-step=757640.ckpt \\
      --smoke-test

Canvas / task-embedding notes
------------------------------
We use a 32×32 discrete integer canvas; each ARC cell is a learned color embedding
rather than an RGB pixel.  Both produce the same 256-patch sequence so the TTT
protocol is equivalent.

The model is trained with num_tasks=400 (training tasks only).  This script appends
one extra slot (ID 400) to the task_embed table.  Each eval task reinitialises that
slot, runs TTT, then the fine-tuned copy is discarded — the original model is never
modified.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from experiments.datamodules.arc import IGNORE_INDEX, PAD_INDEX, _place_on_canvas, _scale_grid
from experiments.utils.checkpointing import load_checkpoint_state_dict
from experiments.utils.cli import load_config_from_file
from nvsubquadratic.lazy_config import instantiate


torch.set_float32_matmul_precision("high")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TTT evaluation for ARC-AGI.")
    p.add_argument("--config", required=True, help="Path to experiment config (.py).")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file.")
    p.add_argument("--data-dir", default="data/arc/data", help="ARC data root.")
    p.add_argument("--ttt-steps", type=int, default=100, help="Gradient steps per TTT attempt.")
    p.add_argument("--ttt-lr", type=float, default=3e-4, help="AdamW lr for TTT.")
    p.add_argument(
        "--ttt-attempts", type=int, default=2, help="Independent TTT runs per task (best majority-voted result kept)."
    )
    p.add_argument(
        "--num-inference-attempts",
        type=int,
        default=10,
        help="Augmented inference passes per TTT attempt for majority voting.",
    )
    p.add_argument("--max-size", type=int, default=32, help="Canvas size — must match training config.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke-test", action="store_true", help="3 tasks / 5 steps / 1 attempt / 2 inference passes.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------


def _strip_compiled_prefix(state_dict: dict) -> dict:
    return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


def _get_task_embed(network: torch.nn.Module) -> torch.nn.Embedding:
    net = getattr(network, "_orig_mod", network)
    return net.embedding.task_embed


def _add_eval_slot(task_embed: torch.nn.Embedding) -> int:
    """Append one zero-initialised row; return its index."""
    with torch.no_grad():
        old_w = task_embed.weight.data
        new_row = torch.zeros(1, old_w.shape[1], dtype=old_w.dtype, device=old_w.device)
        task_embed.weight = torch.nn.Parameter(torch.cat([old_w, new_row], dim=0))
        task_embed.num_embeddings += 1
    return task_embed.num_embeddings - 1


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def _make_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay, matching VARC's lr_scheduler.py."""

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)


# ---------------------------------------------------------------------------
# Batch construction  (augmented or fixed)
# ---------------------------------------------------------------------------


def _build_batch(
    examples: list[dict],
    task_id: int,
    device: torch.device,
    max_size: int,
    augment: bool = False,
) -> dict | None:
    """Build a batch from *examples*, optionally with random scale + translation.

    Returns a dict with keys:
        input, label, condition (task_id, attention_mask),
        scales (list[int]), offsets (list[tuple[int,int]])
    or None if every example was filtered out.
    """
    inputs, labels, scales, offsets = [], [], [], []

    for ex in examples:
        inp = np.array(ex["input"], dtype=np.int64)
        out = np.array(ex.get("output", [[0]]), dtype=np.int64)
        max_dim = max(inp.shape[0], inp.shape[1], out.shape[0], out.shape[1])
        if max_dim > max_size - 2:
            continue

        # --- resolution augmentation ---
        if augment:
            max_scale = max(1, (max_size - 2) // max_dim)
            scale = random.randint(1, max_scale)
        else:
            scale = 1

        inp = _scale_grid(inp, scale)
        out = _scale_grid(out, scale)

        in_h, in_w = inp.shape
        out_h, out_w = out.shape
        scaled_max = max(in_h, in_w, out_h, out_w)
        avail = max(0, max_size - scaled_max - 2)

        # --- translation augmentation ---
        if augment:
            y_off = random.randint(1, 1 + avail)
            x_off = random.randint(1, 1 + avail)
        else:
            y_off = x_off = 1

        inp_canvas = _place_on_canvas(inp, max_size, y_off, x_off, IGNORE_INDEX)
        out_canvas = _place_on_canvas(out, max_size, y_off, x_off, IGNORE_INDEX)

        # Output-shape border markers
        if x_off + out_w < max_size:
            out_canvas[y_off : y_off + out_h, x_off + out_w] = PAD_INDEX
        if y_off + out_h < max_size:
            out_canvas[y_off + out_h, x_off : x_off + out_w + 1] = PAD_INDEX

        inputs.append(torch.from_numpy(inp_canvas))
        labels.append(torch.from_numpy(out_canvas))
        scales.append(scale)
        offsets.append((y_off, x_off))

    if not inputs:
        return None

    input_t = torch.stack(inputs).to(device)
    label_t = torch.stack(labels).to(device)
    task_ids = torch.full((len(inputs),), task_id, dtype=torch.long, device=device)
    attn_mask = (input_t != IGNORE_INDEX).long()

    return {
        "input": input_t,
        "label": label_t,
        "condition": {"task_id": task_ids, "attention_mask": attn_mask},
        "scales": scales,
        "offsets": offsets,
    }


# ---------------------------------------------------------------------------
# Prediction extraction and majority voting
# ---------------------------------------------------------------------------


def _extract_prediction(
    pred_canvas: np.ndarray,
    scale: int,
    y_off: int,
    x_off: int,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """Extract the predicted output grid from an augmented canvas.

    If scale > 1, each original cell maps to a (scale × scale) block in the
    canvas; we downsample by taking the majority-vote color within each block,
    counting only valid ARC colors (0–9).
    """
    region = pred_canvas[y_off : y_off + orig_h * scale, x_off : x_off + orig_w * scale]
    if scale == 1:
        return region.copy()
    # Reshape so that blocks[i, j] contains the scale×scale pixels for cell (i, j)
    blocks = region.reshape(orig_h, scale, orig_w, scale).transpose(0, 2, 1, 3)  # (H, W, s, s)
    blocks = blocks.reshape(orig_h, orig_w, scale * scale)  # (H, W, s²)
    result = np.array(
        [[np.bincount(blocks[i, j].clip(0, 9), minlength=10).argmax() for j in range(orig_w)] for i in range(orig_h)],
        dtype=np.int64,
    )
    return result


def _majority_vote(predictions: list[np.ndarray]) -> np.ndarray:
    """Pixel-wise majority vote over a list of (H, W) predicted grids.

    Only ARC colors 0–9 are counted; any out-of-range values are ignored.
    """
    stacked = np.stack(predictions, axis=0).clip(0, 9)  # [N, H, W]
    N, H, W = stacked.shape
    pixels = stacked.reshape(N, H * W).T  # [H*W, N]
    voted_flat = np.array([np.bincount(row, minlength=10).argmax() for row in pixels], dtype=np.int64)
    return voted_flat.reshape(H, W)


# ---------------------------------------------------------------------------
# Per-task TTT
# ---------------------------------------------------------------------------


def _run_ttt_task(
    network_original: torch.nn.Module,
    eval_slot: int,
    task_path: Path,
    device: torch.device,
    max_size: int,
    ttt_steps: int,
    ttt_lr: float,
    ttt_attempts: int,
    num_inference_attempts: int,
) -> dict | None:
    """Run full-model TTT for one eval task and return exact_match / pixel_acc.

    For each of *ttt_attempts* independent runs:
      1. Deepcopy the original model and reinitialise the eval task embedding.
      2. Fine-tune ALL parameters on the task's training demonstrations with
         random augmentation, cosine LR schedule, and gradient clipping.
      3. Run *num_inference_attempts* augmented forward passes on the first test
         query, extract the predicted output grid from each, and collect them.
    Majority-vote across all predictions (attempts × inference passes) to produce
    the final prediction, then compute exact match against the ground-truth output.
    """
    task_json = json.loads(task_path.read_text())
    train_examples = task_json.get("train", [])
    test_examples = task_json.get("test", [])
    if not train_examples or not test_examples:
        return None

    gt_output = np.array(test_examples[0]["output"], dtype=np.int64)
    gt_h, gt_w = gt_output.shape

    use_amp = device.type == "cuda"
    all_preds: list[np.ndarray] = []

    for _ in range(ttt_attempts):
        # --- Fresh model copy for this attempt ---
        network = deepcopy(network_original)

        # Reinitialise the eval slot with fresh random weights
        task_embed = _get_task_embed(network)
        with torch.no_grad():
            torch.nn.init.trunc_normal_(task_embed.weight[eval_slot : eval_slot + 1], std=0.02)

        # Fine-tune ALL parameters
        network.train()
        optimizer = torch.optim.AdamW(network.parameters(), lr=ttt_lr, weight_decay=0.0)
        warmup_steps = max(1, ttt_steps // 10)
        scheduler = _make_lr_schedule(optimizer, warmup_steps, ttt_steps)

        for _ in range(ttt_steps):
            batch = _build_batch(train_examples, eval_slot, device, max_size, augment=True)
            if batch is None:
                break
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = network({"input": batch["input"], "condition": batch["condition"]})
                loss_labels = batch["label"].clone()
                loss_labels[loss_labels == PAD_INDEX] = IGNORE_INDEX
                loss = F.cross_entropy(output["logits"], loss_labels, ignore_index=IGNORE_INDEX)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # --- Augmented inference passes ---
        network.eval()
        for _ in range(num_inference_attempts):
            test_batch = _build_batch(test_examples[:1], eval_slot, device, max_size, augment=True)
            if test_batch is None:
                break
            scale = test_batch["scales"][0]
            y_off, x_off = test_batch["offsets"][0]
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    output = network({"input": test_batch["input"], "condition": test_batch["condition"]})
                pred = output["logits"].argmax(dim=1)[0].cpu().numpy()
            extracted = _extract_prediction(pred, scale, y_off, x_off, gt_h, gt_w)
            all_preds.append(extracted)

        del network  # free GPU memory before next attempt

    if not all_preds:
        return None

    voted = _majority_vote(all_preds)
    exact_match = float(np.array_equal(voted, gt_output))
    pixel_acc = float((voted == gt_output).mean())
    return {"exact_match": exact_match, "pixel_acc": pixel_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    # Build model (no compile for TTT)
    config = load_config_from_file(args.config)
    if hasattr(config, "compile"):
        config.compile = False

    print(f"[ttt] Building model from config: {args.config}")
    network = instantiate(config.net)

    print(f"[ttt] Loading checkpoint: {args.checkpoint}")
    state_dict = load_checkpoint_state_dict(args.checkpoint)
    state_dict = _strip_compiled_prefix(state_dict)
    state_dict = {k.removeprefix("network."): v for k, v in state_dict.items() if k.startswith("network.")}

    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load] WARNING — missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[load] WARNING — unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    if not missing and not unexpected:
        print("[load] All weights loaded cleanly.")

    network.to(device)
    network.eval()

    # Append the shared eval slot (deepcopies will inherit it and reinitialise it)
    task_embed = _get_task_embed(network)
    eval_slot = _add_eval_slot(task_embed)
    print(f"[ttt] task_embed expanded to {task_embed.num_embeddings} entries; eval slot = {eval_slot}")

    eval_dir = Path(args.data_dir) / "evaluation"
    task_paths = sorted(eval_dir.glob("*.json"))
    if not task_paths:
        raise FileNotFoundError(f"No eval tasks found in {eval_dir}")

    ttt_steps = args.ttt_steps
    ttt_attempts = args.ttt_attempts
    num_inference_attempts = args.num_inference_attempts

    if args.smoke_test:
        task_paths = task_paths[:3]
        ttt_steps = 5
        ttt_attempts = 1
        num_inference_attempts = 2
        print(
            f"[ttt] Smoke test: {len(task_paths)} tasks × {ttt_steps} TTT steps"
            f" × {ttt_attempts} attempt × {num_inference_attempts} inference passes"
        )
    else:
        print(
            f"[ttt] Full eval: {len(task_paths)} tasks"
            f" × {ttt_steps} TTT steps × {ttt_attempts} attempts × {num_inference_attempts} inference passes"
        )

    exact_matches: list[float] = []
    pixel_accs: list[float] = []

    for task_path in tqdm(task_paths, desc="TTT eval"):
        result = _run_ttt_task(
            network_original=network,
            eval_slot=eval_slot,
            task_path=task_path,
            device=device,
            max_size=args.max_size,
            ttt_steps=ttt_steps,
            ttt_lr=args.ttt_lr,
            ttt_attempts=ttt_attempts,
            num_inference_attempts=num_inference_attempts,
        )
        if result is not None:
            exact_matches.append(result["exact_match"])
            pixel_accs.append(result["pixel_acc"])

    n = len(exact_matches)
    total = len(task_paths)
    mean_em = sum(exact_matches) / n if n else 0.0
    mean_pa = sum(pixel_accs) / n if n else 0.0

    print(f"\n{'=' * 50}")
    print(f"TTT Results  ({n}/{total} tasks evaluated)")
    print(f"  exact_match : {mean_em:.4f}  ({mean_em * 100:.2f}%)")
    print(f"  pixel_acc   : {mean_pa:.4f}  ({mean_pa * 100:.2f}%)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
