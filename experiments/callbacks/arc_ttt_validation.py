"""Test-Time Training validation callback for ARC-AGI.

Runs mini-TTT on a sampled subset of evaluation tasks every N training epochs.
For each task the task token is reinitialised, fine-tuned for ``ttt_steps``
gradient steps on the task's "train" demonstrations (all weights frozen except
the task token), then evaluated on the task's "test" query.  Exact-match and
pixel accuracy are logged under the ``val_ttt/`` prefix.

This matches the TTT protocol used in the VARC paper:
  - 100 gradient steps per task (``ttt_steps=100``)
  - No augmentation (scale=1, offset=(1,1))
  - Only the task token is updated; all other weights stay frozen
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback

from experiments.datamodules.arc import IGNORE_INDEX, PAD_INDEX, _place_on_canvas


class ARCTTTValidationCallback(Callback):
    """Periodic TTT validation: fine-tune task tokens on eval tasks and measure exact match.

    Args:
        ttt_val_tasks: Number of evaluation tasks to sample per validation run.
        ttt_val_every_n_epochs: How often (training epochs) to run TTT validation.
        ttt_steps: Gradient steps per task — 100 matches the VARC paper / test protocol.
        ttt_lr: Learning rate for the task-token AdamW optimizer.
        seed: RNG seed for task sampling; the same fixed subset is used every validation run.
    """

    def __init__(
        self,
        ttt_val_tasks: int = 20,
        ttt_val_every_n_epochs: int = 5,
        ttt_steps: int = 100,
        ttt_lr: float = 3e-4,
        seed: int = 0,
    ) -> None:
        """Init function."""
        super().__init__()
        self.ttt_val_tasks = ttt_val_tasks
        self.ttt_val_every_n_epochs = ttt_val_every_n_epochs
        self.ttt_steps = ttt_steps
        self.ttt_lr = ttt_lr
        self.seed = seed

    # ------------------------------------------------------------------
    # Callback entry point
    # ------------------------------------------------------------------

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Run TTT validation on rank 0 every ``ttt_val_every_n_epochs`` epochs."""
        if (trainer.current_epoch + 1) % self.ttt_val_every_n_epochs != 0:
            return
        if trainer.global_rank != 0:
            return

        datamodule = trainer.datamodule
        eval_task_files: list[tuple[int, Path]] | None = getattr(datamodule, "_eval_task_files", None)
        if not eval_task_files:
            return

        device = pl_module.device
        network = pl_module.network

        # Sample tasks reproducibly — same fixed subset every validation run
        rng = random.Random(self.seed)
        sampled = rng.sample(eval_task_files, min(self.ttt_val_tasks, len(eval_task_files)))

        exact_matches: list[float] = []
        pixel_accs: list[float] = []

        # Freeze all weights except task_token_embed; switch to eval mode
        grad_states = {n: p.requires_grad for n, p in network.named_parameters()}
        for n, p in network.named_parameters():
            p.requires_grad_(n == "task_token_embed.weight")
        network.eval()

        try:
            for task_id, task_path in sampled:
                result = self._run_ttt_task(network, task_id, task_path, device)
                if result is not None:
                    exact_matches.append(result["exact_match"])
                    pixel_accs.append(result["pixel_acc"])
        finally:
            # Always restore model state regardless of errors
            for n, p in network.named_parameters():
                p.requires_grad_(grad_states[n])
            network.train()

        if not exact_matches:
            return

        avg_exact_match = sum(exact_matches) / len(exact_matches)
        avg_pixel_acc = sum(pixel_accs) / len(pixel_accs)

        pl_module.log("val_ttt/exact_match", avg_exact_match, prog_bar=True, rank_zero_only=True)
        pl_module.log("val_ttt/pixel_acc", avg_pixel_acc, rank_zero_only=True)
        pl_module.log("val_ttt/num_tasks", float(len(exact_matches)), rank_zero_only=True)

    # ------------------------------------------------------------------
    # Per-task TTT
    # ------------------------------------------------------------------

    def _run_ttt_task(
        self,
        network: torch.nn.Module,
        task_id: int,
        task_path: Path,
        device: torch.device,
    ) -> Optional[dict[str, float]]:
        """Reinit, fine-tune, and evaluate the task token for a single ARC task.

        Returns a dict with ``exact_match`` and ``pixel_acc``, or ``None`` if the
        task has no usable examples.
        """
        task_json = json.loads(task_path.read_text())
        train_examples = task_json.get("train", [])
        test_examples = task_json.get("test", [])
        if not train_examples or not test_examples:
            return None

        # Save original token and reinitialise with fresh random weights
        with torch.no_grad():
            original_token = network.task_token_embed.weight[task_id].clone()
            torch.nn.init.trunc_normal_(network.task_token_embed.weight[task_id : task_id + 1], std=0.02)

        try:
            train_batch = self._build_batch(train_examples, task_id, device)
            if train_batch is None:
                return None

            # AdamW on the full embedding weight; only the task_id row receives
            # gradients (sparse embedding updates) so other tasks are unaffected.
            optimizer = torch.optim.AdamW([network.task_token_embed.weight], lr=self.ttt_lr, weight_decay=0.0)

            for _ in range(self.ttt_steps):
                optimizer.zero_grad()
                output = network({"input": train_batch["input"], "condition": train_batch["condition"]})
                loss_labels = train_batch["label"].clone()
                loss_labels[loss_labels == PAD_INDEX] = IGNORE_INDEX
                loss = F.cross_entropy(output["logits"], loss_labels, ignore_index=IGNORE_INDEX)
                loss.backward()
                optimizer.step()

            # Evaluate on first test query (no augmentation)
            test_batch = self._build_batch(test_examples[:1], task_id, device)
            if test_batch is None:
                return None

            with torch.no_grad():
                output = network({"input": test_batch["input"], "condition": test_batch["condition"]})
                logits = output["logits"]
                labels = test_batch["label"]
                preds = logits.argmax(dim=1)
                valid = (labels != IGNORE_INDEX) & (labels != PAD_INDEX)
                pixel_acc = (preds[valid] == labels[valid]).float().mean().item() if valid.any() else 0.0
                exact_match = ((preds == labels) | ~valid).all(dim=(-2, -1)).float().mean().item()

            return {"exact_match": exact_match, "pixel_acc": pixel_acc}

        finally:
            # Restore the original task token so training is unaffected
            with torch.no_grad():
                network.task_token_embed.weight[task_id] = original_token

    # ------------------------------------------------------------------
    # Batch construction (no augmentation — matches VARC TTT protocol)
    # ------------------------------------------------------------------

    def _build_batch(
        self,
        examples: list[dict],
        task_id: int,
        device: torch.device,
        max_size: int = 32,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Convert raw ARC examples into a model-ready batch with no augmentation.

        Grids are placed at offset (1, 1) with scale=1, matching VARC's TTT
        evaluation settings (``--disable-translation``, ``--fix-scale-factor 1``).
        """
        inputs, labels = [], []

        for ex in examples:
            inp = np.array(ex["input"], dtype=np.int64)
            out = np.array(ex.get("output", [[0]]), dtype=np.int64)

            # Skip grids too large to fit with a 1-cell border
            if max(inp.shape[0], inp.shape[1], out.shape[0], out.shape[1]) > max_size - 2:
                continue

            inp_canvas = _place_on_canvas(inp, max_size, 1, 1, IGNORE_INDEX)
            out_canvas = _place_on_canvas(out, max_size, 1, 1, IGNORE_INDEX)

            # Add output-shape border markers
            out_h, out_w = out.shape
            if 1 + out_w < max_size:
                out_canvas[1 : 1 + out_h, 1 + out_w] = PAD_INDEX
            if 1 + out_h < max_size:
                out_canvas[1 + out_h, 1 : 1 + out_w + 1] = PAD_INDEX

            inputs.append(torch.from_numpy(inp_canvas))
            labels.append(torch.from_numpy(out_canvas))

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
        }
