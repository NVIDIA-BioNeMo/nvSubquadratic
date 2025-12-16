"""Lightning wrapper that mimics the VARC training recipe (canvas + augmentations)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import torchmetrics

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class ArcAGIVARCWrapper(LightningWrapperBase):
    """Implements the key training tricks from `ARC Is a Vision Problem!`.

    Highlights:
        * Canvas-based translation & scale augmentation.
        * Optional geometric flips, rotations, and color permutations.
        * Multi-view inference (aggregate predictions from multiple random views).
    """

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
        *,
        ignore_index: int = -100,
        num_colors: int = 10,
        canvas_size: int = 64,
        min_scale: int = 1,
        max_scale: int = 4,
        background_value: float = 0.0,
        enable_flip: bool = True,
        enable_rotation: bool = True,
        enable_color_permutation: bool = True,
        color_permutation_prob: float = 0.3,
        train_views: int = 1,
        val_views: int = 8,
        test_views: int = 16,
    ) -> None:
        super().__init__(network=network, cfg=cfg)
        self.ignore_index = ignore_index
        self.num_colors = num_colors
        self.canvas_size = canvas_size
        self.min_scale = max(1, min_scale)
        self.max_scale = max(self.min_scale, max_scale)
        self.background_value = background_value
        self.enable_flip = enable_flip
        self.enable_rotation = enable_rotation
        self.enable_color_permutation = enable_color_permutation
        self.color_permutation_prob = color_permutation_prob
        self.train_views = max(1, train_views)
        self.val_views = max(1, val_views)
        self.test_views = max(1, test_views)

        self.train_pixel_acc = torchmetrics.MeanMetric()
        self.val_pixel_acc = torchmetrics.MeanMetric()
        self.test_pixel_acc = torchmetrics.MeanMetric()

        self.train_exact = torchmetrics.MeanMetric()
        self.val_exact = torchmetrics.MeanMetric()
        self.test_exact = torchmetrics.MeanMetric()

    # ------------------------------------------------------------------
    # Metric helpers copied from ArcAGIWrapper
    # ------------------------------------------------------------------
    def _compute_masked_metrics(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
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

    # ---------------------------------------------------------------------
    # Augmentation helpers
    # ---------------------------------------------------------------------
    def _maybe_color_permute(
        self,
        input_grid: torch.Tensor,
        label_grid: torch.Tensor,
        mask: torch.Tensor,
        apply_perm: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly permute colors across the entire grid."""
        if not (self.enable_color_permutation and apply_perm):
            return input_grid, label_grid

        perm = torch.randperm(self.num_colors, device=input_grid.device)
        # Inputs are normalized to [0, 1]; convert to discrete ids for permutation.
        scaled = torch.round(input_grid * (self.num_colors - 1)).clamp(0, self.num_colors - 1).long()
        permuted = perm[scaled]
        input_grid = permuted.float() / float(self.num_colors - 1)

        if label_grid is not None:
            valid = mask & (label_grid != self.ignore_index)
            new_labels = label_grid.clone()
            new_labels[valid] = perm[new_labels[valid].clamp(0, self.num_colors - 1)]
            label_grid = new_labels

        return input_grid, label_grid

    def _apply_geometric_transforms(
        self,
        input_grid: torch.Tensor,
        label_grid: torch.Tensor,
        mask: torch.Tensor,
        condition_grid: Optional[torch.Tensor],
        apply_random: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply optional random flips/rotations."""
        if apply_random and self.enable_rotation:
            k = torch.randint(0, 4, ()).item()
            if k:
                input_grid = torch.rot90(input_grid, k, [0, 1])
                label_grid = torch.rot90(label_grid, k, [0, 1])
                mask = torch.rot90(mask, k, [0, 1])
                if condition_grid is not None:
                    condition_grid = torch.rot90(condition_grid, k, [0, 1])

        if apply_random and self.enable_flip:
            if torch.rand(()) < 0.5:
                input_grid = torch.flip(input_grid, [0])
                label_grid = torch.flip(label_grid, [0])
                mask = torch.flip(mask, [0])
                if condition_grid is not None:
                    condition_grid = torch.flip(condition_grid, [0])

            if torch.rand(()) < 0.5:
                input_grid = torch.flip(input_grid, [1])
                label_grid = torch.flip(label_grid, [1])
                mask = torch.flip(mask, [1])
                if condition_grid is not None:
                    condition_grid = torch.flip(condition_grid, [1])

        return input_grid, label_grid, mask, condition_grid

    def _scale_tensor(self, tensor: torch.Tensor, scale: int, mode: str) -> torch.Tensor:
        """Nearest-neighbor scaling for inputs/labels/masks."""
        if scale == 1:
            return tensor

        if tensor.ndim == 3:
            # (H, W, C) -> (C, H, W) for interpolation.
            tensor_chw = tensor.permute(2, 0, 1).unsqueeze(0)
            scaled = F.interpolate(tensor_chw, scale_factor=scale, mode="nearest").squeeze(0)
            return scaled.permute(1, 2, 0)

        # (H, W) tensors (labels, masks)
        tensor_hw = tensor.unsqueeze(0).unsqueeze(0)
        scaled = F.interpolate(tensor_hw.float(), scale_factor=scale, mode=mode).squeeze(0).squeeze(0)
        if tensor.dtype == torch.long:
            return scaled.round().long()
        if tensor.dtype == torch.bool:
            return scaled > 0.5
        return scaled

    def _place_on_canvas(
        self,
        input_grid: torch.Tensor,
        label_grid: torch.Tensor,
        mask: torch.Tensor,
        condition_grid: Optional[torch.Tensor],
        scale: int,
        apply_translation: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Scale and translate grids onto the shared canvas."""
        input_scaled = self._scale_tensor(input_grid, scale, mode="nearest")
        label_scaled = self._scale_tensor(label_grid, scale, mode="nearest")
        mask_scaled = self._scale_tensor(mask, scale, mode="nearest")
        condition_scaled = None
        if condition_grid is not None:
            condition_scaled = self._scale_tensor(condition_grid, scale, mode="nearest")

        h, w = input_scaled.shape[:2]
        max_h = self.canvas_size - h
        max_w = self.canvas_size - w
        offset_y = torch.randint(0, max(1, max_h + 1), ()).item() if apply_translation and max_h > 0 else 0
        offset_x = torch.randint(0, max(1, max_w + 1), ()).item() if apply_translation and max_w > 0 else 0

        canvas = torch.full(
            (self.canvas_size, self.canvas_size, input_scaled.shape[-1]),
            self.background_value,
            device=input_grid.device,
            dtype=input_grid.dtype,
        )
        label_canvas = torch.full(
            (self.canvas_size, self.canvas_size),
            self.ignore_index,
            device=label_grid.device,
            dtype=label_grid.dtype,
        )
        mask_canvas = torch.zeros(
            (self.canvas_size, self.canvas_size),
            device=mask.device,
            dtype=mask.dtype,
        )
        canvas[offset_y : offset_y + h, offset_x : offset_x + w] = input_scaled
        label_canvas[offset_y : offset_y + h, offset_x : offset_x + w] = label_scaled
        mask_canvas[offset_y : offset_y + h, offset_x : offset_x + w] = mask_scaled

        condition_canvas = None
        if condition_scaled is not None:
            condition_canvas = torch.zeros(
                (self.canvas_size, self.canvas_size, condition_scaled.shape[-1]),
                device=condition_scaled.device,
                dtype=condition_scaled.dtype,
            )
            condition_canvas[offset_y : offset_y + h, offset_x : offset_x + w] = condition_scaled

        return canvas, label_canvas, mask_canvas, condition_canvas

    def _prepare_augmented_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        training: bool,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Apply VARC-style augmentations and return model-ready tensors + labels/masks."""
        inputs = batch["input"]
        labels = batch["label"]
        masks = batch["condition"]["label_mask"]
        model_condition = batch.get("model_condition")

        augmented_inputs = []
        augmented_labels = []
        augmented_masks = []
        augmented_conditions: list[torch.Tensor] | None = [] if model_condition is not None else None

        for idx in range(inputs.shape[0]):
            inp = inputs[idx].clone()
            lbl = labels[idx].clone()
            msk = masks[idx].clone()
            cond = model_condition[idx].clone() if model_condition is not None else None

            apply_perm = training and (torch.rand(()) < self.color_permutation_prob)
            inp, lbl = self._maybe_color_permute(inp, lbl, msk, apply_perm)
            inp, lbl, msk, cond = self._apply_geometric_transforms(inp, lbl, msk, cond, training)

            # Ensure scale fits the canvas for the current sample.
            max_allowed_scale = min(self.max_scale, self.canvas_size // max(lbl.shape[0], lbl.shape[1]))
            max_allowed_scale = max(self.min_scale, max_allowed_scale)
            scale = torch.randint(self.min_scale, max_allowed_scale + 1, ()).item() if max_allowed_scale > 0 else 1
            inp, lbl, msk, cond = self._place_on_canvas(inp, lbl, msk, cond, scale, apply_translation=True)

            augmented_inputs.append(inp)
            augmented_labels.append(lbl)
            augmented_masks.append(msk)
            if augmented_conditions is not None:
                if cond is None:
                    raise RuntimeError("Model condition is enabled but no conditioning tensor was provided.")
                augmented_conditions.append(cond)

        inputs_tensor = torch.stack(augmented_inputs)
        labels_tensor = torch.stack(augmented_labels)
        masks_tensor = torch.stack(augmented_masks)

        model_inputs = {"input": inputs_tensor, "condition": None}
        if augmented_conditions is not None:
            model_inputs["condition"] = torch.stack(augmented_conditions)

        return model_inputs, labels_tensor, masks_tensor

    # ---------------------------------------------------------------------
    # Forward helpers
    # ---------------------------------------------------------------------
    def _forward_views(
        self,
        batch: dict[str, torch.Tensor],
        *,
        training: bool,
        num_views: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the network across multiple random views and aggregate logits."""
        logits_by_view = []
        labels_aug = None
        masks_aug = None
        for _ in range(num_views):
            model_inputs, labels_aug, masks_aug = self._prepare_augmented_batch(batch, training=training)
            outputs = self.network(model_inputs)
            logits_by_view.append(outputs["logits"])

        stacked = torch.stack(logits_by_view, dim=0)  # (V, B, H, W, C)
        mean_logits = stacked.mean(dim=0)
        return mean_logits, labels_aug, masks_aug

    # ---------------------------------------------------------------------
    # Training / Evaluation
    # ---------------------------------------------------------------------
    def _step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        training: bool,
        pixel_metric: torchmetrics.Metric,
        exact_metric: torchmetrics.Metric,
        num_views: int,
    ) -> torch.Tensor:
        # Multi-view inference is only necessary when not training
        views = 1 if training else num_views
        logits, labels, masks = self._forward_views(batch, training=training, num_views=views)
        loss = F.cross_entropy(
            logits.permute(0, 3, 1, 2),
            labels,
            ignore_index=self.ignore_index,
        )

        pixel_acc, exact_match = self._compute_masked_metrics(logits, labels, masks)
        pixel_metric(pixel_acc)
        exact_metric(exact_match)
        return loss

    def training_step(self, batch, batch_idx):
        batch_size = batch["input"].shape[0]
        loss = self._step(
            batch,
            training=True,
            pixel_metric=self.train_pixel_acc,
            exact_metric=self.train_exact,
            num_views=self.train_views,
        )
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        self.log(
            "train/pixel_acc",
            self.train_pixel_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        self.log(
            "train/exact_match",
            self.train_exact,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["input"].shape[0]
        loss = self._step(
            batch,
            training=False,
            pixel_metric=self.val_pixel_acc,
            exact_metric=self.val_exact,
            num_views=self.val_views,
        )
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        self.log(
            "val/pixel_acc",
            self.val_pixel_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        self.log(
            "val/exact_match",
            self.val_exact,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx):
        batch_size = batch["input"].shape[0]
        loss = self._step(
            batch,
            training=False,
            pixel_metric=self.test_pixel_acc,
            exact_metric=self.test_exact,
            num_views=self.test_views,
        )
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        self.log(
            "test/pixel_acc",
            self.test_pixel_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        self.log(
            "test/exact_match",
            self.test_exact,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        return loss
