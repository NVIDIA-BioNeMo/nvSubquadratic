# David W. Romero, 2025-01-19

"""Lightning wrapper for autoregressive (next-token prediction) tasks.

Supports:
- Discrete tokens (cross-entropy loss, perplexity metric)
- Continuous values (MSE/MAE loss)
- Teacher forcing during training
- Autoregressive generation during inference

For causal language modeling style tasks where:
- Input: x[:, :-1] (all tokens except last)
- Target: x[:, 1:] (all tokens except first)
"""

from typing import Literal, Optional

import torch
import torch.nn.functional as F
import torchmetrics

from experiments.default_cfg import ExperimentConfig
from experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class AutoregressiveWrapper(LightningWrapperBase):
    """Lightning wrapper for autoregressive (next-token prediction) tasks.

    Args:
        network: Network to wrap. Should output logits of shape [B, L, vocab_size] for
            discrete tokens or [B, L, C] for continuous values.
        cfg: Experiment configuration.
        mode: "discrete" for token prediction (cross-entropy), "continuous" for
            value prediction (MSE/MAE).
        vocab_size: Vocabulary size (required for discrete mode).
        loss_type: Loss type for continuous mode ("mse" or "mae"). Ignored for discrete.
        ignore_index: Index to ignore in loss computation (e.g., padding token).
            Default -100 (PyTorch convention).
    """

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: ExperimentConfig,
        mode: Literal["discrete", "continuous"] = "discrete",
        vocab_size: Optional[int] = None,
        loss_type: Literal["mse", "mae"] = "mse",
        ignore_index: int = -100,
    ):
        """Initialize the AutoregressiveWrapper."""
        super().__init__(
            network=network,
            cfg=cfg,
        )
        self.mode = mode
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

        if mode == "discrete":
            if vocab_size is None:
                raise ValueError("vocab_size must be provided for discrete mode")
            # Cross-entropy loss for discrete tokens
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
            # Metrics
            self.train_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=vocab_size, ignore_index=ignore_index
            )
            self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=ignore_index)
            self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, ignore_index=ignore_index)
        else:
            # MSE/MAE loss for continuous values
            if loss_type == "mse":
                self.loss_fn = torch.nn.MSELoss()
                MetricClass = torchmetrics.MeanSquaredError
            else:
                self.loss_fn = torch.nn.L1Loss()
                MetricClass = torchmetrics.MeanAbsoluteError
            self.train_metric = MetricClass()
            self.val_metric = MetricClass()
            self.test_metric = MetricClass()

        # Best loss tracking
        self.best_train_loss = float("inf")
        self.best_val_loss = float("inf")

    def _prepare_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare input and target tensors for autoregressive training.

        Supports two modes:
        1. Standard autoregressive shift: when label == input (or label is None),
           input is x[:, :-1] and target is x[:, 1:].
        2. Custom labels: when label is a separate tensor (e.g., MQAR with -100 masking),
           input is still x[:, :-1] but target comes from the provided labels.

        Args:
            batch: Dictionary with "input" key and optional "label" key.

        Returns:
            Tuple of (input_tensor, target_tensor).
        """
        x = batch["input"]  # [B, L, C] or [B, L]
        label = batch.get("label")

        # Check if custom labels are provided (different from input)
        has_custom_labels = label is not None and not (label.shape == x.shape and torch.equal(label, x))

        if has_custom_labels:
            # Custom labels mode (e.g., MQAR): input and label are already aligned
            # input_seq: [B, L] or [B, L, C], target_seq: [B, L] (with -100 for ignored positions)
            input_seq = x
            target_seq = label
        else:
            # Standard autoregressive shift: input[:, :-1], target[:, 1:]
            if x.ndim == 3:
                # [B, L, C] -> input: [B, L-1, C], target: [B, L-1, C]
                input_seq = x[:, :-1, :]
                target_seq = x[:, 1:, :]
            else:
                # [B, L] -> input: [B, L-1], target: [B, L-1]
                input_seq = x[:, :-1]
                target_seq = x[:, 1:]

        return input_seq, target_seq

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and predictions.

        Args:
            logits: Model output, shape [B, L, vocab_size] for discrete or [B, L, C] for continuous.
            targets: Target tensor, shape [B, L] for discrete or [B, L, C] for continuous.

        Returns:
            Tuple of (loss, predictions).
        """
        if self.mode == "discrete":
            # Reshape for cross-entropy: [B*L, vocab_size] and [B*L]
            B, L, V = logits.shape
            logits_flat = logits.reshape(B * L, V)
            targets_flat = targets.reshape(B * L)
            loss = self.loss_fn(logits_flat, targets_flat)
            predictions = logits.argmax(dim=-1)  # [B, L]
        else:
            # Continuous: direct MSE/MAE
            loss = self.loss_fn(logits, targets)
            predictions = logits

        return loss, predictions

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        metric_calculator: torchmetrics.Metric,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Perform a training/validation/test step.

        Args:
            batch: Input batch dictionary.
            metric_calculator: Metric to update.

        Returns:
            Tuple of (predictions, loss, other_outputs).
        """
        # Validate the structure of the batch
        assert isinstance(batch, dict), "Batch must be a dictionary"
        assert len(batch) == 3, "Batch must contain exactly 3 keys: 'input', 'label' and 'condition'"
        assert "input" in batch, "Batch must contain 'input' key"
        assert "label" in batch, "Batch must contain 'label' key"
        assert "condition" in batch, "Batch must contain 'condition' key"

        # Prepare shifted sequences
        input_seq, target_seq = self._prepare_batch(batch)

        # Forward pass (teacher forcing)
        output = self({"input": input_seq, "condition": batch.get("condition")})
        logits = output["logits"]  # [B, L-1, vocab_size] or [B, L-1, C]

        # Compute loss
        loss, predictions = self._compute_loss(logits, target_seq)

        # Update metrics
        if self.mode == "discrete":
            metric_calculator(predictions.reshape(-1), target_seq.reshape(-1))
        else:
            metric_calculator(predictions.reshape(-1), target_seq.reshape(-1))

        other_outputs = {}
        return predictions, loss, other_outputs

    def training_step(self, batch, batch_idx):
        """Perform training step."""
        # Start timing (CUDA events)
        self._start_timing()
        # Perform step
        if self.mode == "discrete":
            predictions, loss, other_outputs = self._step(batch, self.train_acc)
        else:  # Continuous mode
            predictions, loss, other_outputs = self._step(batch, self.train_metric)
        # Log loss
        self.log("train/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        # Log perplexity for discrete mode
        if self.mode == "discrete":
            perplexity = torch.exp(loss)
            self.log("train/perplexity", perplexity, on_epoch=True, prog_bar=False, sync_dist=self.distributed)
        # Add other outputs to the list of other outputs. This is used for end of epoch logging.
        self.other_outputs_train.append(other_outputs)
        # Return loss
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform validation step."""
        # Perform step
        if self.mode == "discrete":
            predictions, loss, other_outputs = self._step(batch, self.val_acc)
        else:  # Continuous mode
            predictions, loss, other_outputs = self._step(batch, self.val_metric)
        # Log loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        # Log perplexity for discrete mode
        if self.mode == "discrete":
            perplexity = torch.exp(loss)
            self.log(
                "val/perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.distributed
            )
        # Add other outputs to the list of other outputs. This is used for end of epoch logging.
        self.other_outputs_validation.append(other_outputs)
        # Return loss
        return loss

    def test_step(self, batch, batch_idx):
        """Perform test step."""
        # Perform step
        if self.mode == "discrete":
            predictions, loss, _ = self._step(batch, self.test_acc)
        else:  # Continuous mode
            predictions, loss, _ = self._step(batch, self.test_metric)
        # Log loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        # Log perplexity for discrete mode
        if self.mode == "discrete":
            perplexity = torch.exp(loss)
            self.log(
                "test/perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=False, sync_dist=self.distributed
            )

    def on_train_epoch_end(self):
        """Log metrics at end of training epoch."""
        train_step_outputs = self.other_outputs_train
        if len(train_step_outputs) == 0:
            # When autoresuming, the first epoch step outputs is empty, which would otherwise raise an error.
            # We add this here to avoid that error.
            return

        self.other_outputs_train.clear()

        # Log best training loss
        train_loss = self.trainer.callback_metrics.get("train/loss_epoch")
        if train_loss is not None and train_loss < self.best_train_loss:
            self.best_train_loss = train_loss.item()
            if self.logger is not None:
                self.logger.experiment.log({"train/best_loss": self.best_train_loss, "global_step": self.global_step})

        # Log accuracy for discrete mode
        if self.mode == "discrete":
            train_acc = self.train_acc.compute()
            self.log("train/accuracy", train_acc, prog_bar=False, sync_dist=self.distributed)
            self.train_acc.reset()

    def on_validation_epoch_end(self):
        """Log metrics at end of validation epoch."""
        validation_step_outputs = self.other_outputs_validation
        if len(validation_step_outputs) == 0:
            # When autoresuming, the first epoch step outputs is empty, which would otherwise raise an error.
            # We add this here to avoid that error.
            return

        self.other_outputs_validation.clear()

        # Log best validation loss
        val_loss = self.trainer.callback_metrics.get("val/loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            if self.logger is not None:
                self.logger.experiment.log({"val/best_loss": self.best_val_loss, "global_step": self.global_step})

        # Log accuracy for discrete mode
        if self.mode == "discrete":
            val_acc = self.val_acc.compute()
            self.log("val/accuracy", val_acc, prog_bar=True, sync_dist=self.distributed)
            self.val_acc.reset()

    # =========================================================================
    # Generation utilities
    # =========================================================================

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            prompt: Initial sequence, shape [B, L] for discrete or [B, L, C] for continuous.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (1.0 = no change, <1.0 = more deterministic).
            top_k: If set, only sample from top-k most likely tokens.
            top_p: If set, use nucleus sampling with this probability mass.
            condition: Optional conditioning tensor.

        Returns:
            Generated sequence including prompt, shape [B, L + max_new_tokens, ...].
        """
        self.eval()
        device = next(self.parameters()).device
        prompt = prompt.to(device)
        if condition is not None:
            condition = condition.to(device)

        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # Get logits for next token
            output = self({"input": generated, "condition": condition})
            logits = output["logits"]  # [B, L, vocab_size] or [B, L, C]

            # Take logits for last position
            next_logits = logits[:, -1, :]  # [B, vocab_size] or [B, C]

            if self.mode == "discrete":
                # Apply temperature
                next_logits = next_logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, [-1]]] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

                # Append to generated
                generated = torch.cat([generated, next_token], dim=1)
            else:
                # For continuous mode, just use the prediction directly
                if next_logits.ndim == 2:
                    next_logits = next_logits.unsqueeze(1)  # [B, 1, C]
                generated = torch.cat([generated, next_logits], dim=1)

        return generated

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate tokens using greedy decoding (argmax).

        Args:
            prompt: Initial sequence.
            max_new_tokens: Maximum number of new tokens to generate.
            condition: Optional conditioning tensor.

        Returns:
            Generated sequence including prompt.
        """
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,  # Greedy = top-1
            condition=condition,
        )
