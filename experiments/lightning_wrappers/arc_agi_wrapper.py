"""Lightning wrapper for ARC-AGI grid-to-grid prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchmetrics

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class ArcAGIWrapper(LightningWrapperBase):
    """Masked cross-entropy training for ARC-AGI grid prediction."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
        ignore_index: int = -100,
    ) -> None:
        super().__init__(network=network, cfg=cfg)
        self.ignore_index = ignore_index

        self.train_pixel_acc = torchmetrics.MeanMetric()
        self.val_pixel_acc = torchmetrics.MeanMetric()
        self.test_pixel_acc = torchmetrics.MeanMetric()

        self.train_exact = torchmetrics.MeanMetric()
        self.val_exact = torchmetrics.MeanMetric()
        self.test_exact = torchmetrics.MeanMetric()

    def _compute_masked_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (pixel_acc, exact_match) using the provided mask."""
        preds = logits.argmax(dim=-1)
        valid = mask & (labels != self.ignore_index)

        valid_count = valid.sum()
        correct = ((preds == labels) & valid).sum()
        pixel_acc = correct.float() / valid_count.clamp(min=1).float()

        per_sample_valid = valid.view(valid.shape[0], -1).sum(dim=1)
        per_sample_correct = ((preds == labels) & valid).view(valid.shape[0], -1).sum(dim=1)
        eligible = per_sample_valid > 0
        successes = (per_sample_correct == per_sample_valid) & eligible
        exact_match = successes.float().sum() / eligible.sum().clamp(min=1).float()

        return pixel_acc, exact_match

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        pixel_acc_metric: torchmetrics.Metric,
        exact_metric: torchmetrics.Metric,
    ) -> torch.Tensor:
        assert isinstance(batch, dict), "Batch must be a dictionary"
        assert "label" in batch and "input" in batch and "condition" in batch, "Missing required batch keys."

        labels = batch.pop("label")
        condition = batch["condition"]
        mask = condition.get("label_mask")
        if mask is None:
            raise ValueError("condition['label_mask'] is required for masked loss/metrics.")

        output = self(input_and_condition=batch)
        if not isinstance(output, dict) or "logits" not in output:
            raise ValueError("Model must return a dict with a 'logits' key.")

        logits = output["logits"].contiguous()  # (B, H, W, C)

        # Loss is channel-first for PyTorch CE.
        loss = F.cross_entropy(
            logits.permute(0, 3, 1, 2),
            labels,
            ignore_index=self.ignore_index,
        )

        pixel_acc, exact_match = self._compute_masked_metrics(logits, labels, mask)
        pixel_acc_metric(pixel_acc)
        exact_metric(exact_match)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.train_pixel_acc, self.train_exact)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log(
            "train/pixel_acc",
            self.train_pixel_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "train/exact_match",
            self.train_exact,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, self.val_pixel_acc, self.val_exact)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log(
            "val/pixel_acc",
            self.val_pixel_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "val/exact_match",
            self.val_exact,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, self.test_pixel_acc, self.test_exact)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log(
            "test/pixel_acc",
            self.test_pixel_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        self.log(
            "test/exact_match",
            self.test_exact,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )
        return loss
