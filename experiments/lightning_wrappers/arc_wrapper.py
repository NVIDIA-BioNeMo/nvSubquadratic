from collections import defaultdict
from typing import Any, Dict

import torch
import torch.nn.functional as F

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class ARCWrapper(LightningWrapperBase):
    """Lightning wrapper for ARC-AGI: pixel-level cross-entropy + exact-match metric."""

    def __init__(self, network: torch.nn.Module, cfg: ExperimentConfig, *args, **kwargs):
        """Initialize wrapper with best-metric tracking dicts."""
        super().__init__(network=network, cfg=cfg, *args, **kwargs)
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)

        self.best_metrics = {
            "val/exact_match": -1.0,
            "val/pixel_acc": -1.0,
        }

    def _step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        labels = batch.pop("label")
        output = self(batch)
        logits = output["logits"]

        loss_labels = labels.clone()
        loss_labels[loss_labels == 11] = 10
        loss = F.cross_entropy(logits, loss_labels, ignore_index=10)

        preds = logits.argmax(dim=1)
        valid = (labels != 10) & (labels != 11)

        if valid.any():
            pixel_acc = (preds[valid] == labels[valid]).float().mean()
        else:
            pixel_acc = torch.tensor(0.0, device=logits.device)

        per_example = ((preds == labels) | ~valid).all(dim=(-2, -1))
        exact_match = per_example.float().mean()

        return {
            "loss": loss,
            "pixel_acc": pixel_acc,
            "exact_match": exact_match,
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute loss and log train metrics for one batch."""
        out = self._step(batch)
        self.log("train/loss", out["loss"], sync_dist=True, prog_bar=True)
        self.log("train/pixel_acc", out["pixel_acc"], sync_dist=True)
        self.log("train/exact_match", out["exact_match"], sync_dist=True)

        return out["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Accumulate val metrics for epoch-end aggregation."""
        out = self._step(batch)
        self.validation_step_outputs["val"].append(
            {"loss": out["loss"], "pixel_acc": out["pixel_acc"], "exact_match": out["exact_match"]}
        )

    def on_validation_epoch_end(self) -> None:
        """Aggregate and log val metrics; track per-epoch bests."""
        for prefix, outputs in self.validation_step_outputs.items():
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            avg_pixel_acc = torch.stack([x["pixel_acc"] for x in outputs]).mean()
            avg_exact_match = torch.stack([x["exact_match"] for x in outputs]).mean()

            self.log(f"{prefix}/loss", avg_loss, sync_dist=True)
            self.log(f"{prefix}/pixel_acc", avg_pixel_acc, sync_dist=True, prog_bar=True)
            self.log(f"{prefix}/exact_match", avg_exact_match, sync_dist=True, prog_bar=True)

            if prefix == "val":
                if avg_exact_match > self.best_metrics["val/exact_match"]:
                    self.best_metrics["val/exact_match"] = avg_exact_match.item()
                if avg_pixel_acc > self.best_metrics["val/pixel_acc"]:
                    self.best_metrics["val/pixel_acc"] = avg_pixel_acc.item()

                self.log("val/exact_match_best", self.best_metrics["val/exact_match"], sync_dist=True)
                self.log("val/pixel_acc_best", self.best_metrics["val/pixel_acc"], sync_dist=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        """Accumulate test metrics for epoch-end aggregation."""
        out = self._step(batch)
        self.test_step_outputs[dataloader_idx].append(out)

    def on_test_epoch_end(self) -> None:
        """Aggregate and log test metrics across all dataloaders."""
        for dl_idx, outputs in self.test_step_outputs.items():
            avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
            avg_pixel_acc = torch.stack([x["pixel_acc"] for x in outputs]).mean()
            avg_exact_match = torch.stack([x["exact_match"] for x in outputs]).mean()

            self.log(f"test_{dl_idx}/loss", avg_loss, sync_dist=True)
            self.log(f"test_{dl_idx}/pixel_acc", avg_pixel_acc, sync_dist=True)
            self.log(f"test_{dl_idx}/exact_match", avg_exact_match, sync_dist=True)

        self.test_step_outputs.clear()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Persist best-metric dict alongside model weights."""
        super().on_save_checkpoint(checkpoint)
        checkpoint["best_metrics"] = self.best_metrics

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore best-metric dict from checkpoint."""
        super().on_load_checkpoint(checkpoint)
        if "best_metrics" in checkpoint:
            self.best_metrics = checkpoint["best_metrics"]
