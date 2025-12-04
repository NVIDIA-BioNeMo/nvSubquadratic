"""Lightning wrapper for text pretraining (causal language modeling)."""

import torch
import torchmetrics

import wandb
from experiments.default_cfg import TextPretrainingExperimentConfig
from experiments.lightning_wrappers.base_lightning_wrapper import LightningWrapperBase


class TextPretrainingWrapper(LightningWrapperBase):
    """Lightning wrapper for text pretraining (next token prediction)."""

    def __init__(
        self,
        network: torch.nn.Module,
        cfg: TextPretrainingExperimentConfig,
        vocab_size: int = 50257,
    ):
        """Initialize the TextPretrainingWrapper.

        Args:
            network: Network to wrap.
            cfg: Configuration.
            vocab_size: Vocabulary size.
        """
        super().__init__(
            network=network,
            cfg=cfg,
        )
        self.cfg = cfg

        # Metrics
        # Perplexity is exp(loss), so we can just track loss and compute ppl on the fly or log it.
        # Accuracy: Next token prediction accuracy.
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)

        # Top-5 Accuracy
        self.train_acc_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, top_k=5)
        self.val_acc_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, top_k=5)
        self.test_acc_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, top_k=5)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        # Best metrics
        self.best_val_loss = float("inf")
        self.best_val_ppl = float("inf")

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass."""
        # We allow passing args directly or as a dict in input_and_condition via base class
        # But base class calls self.network(input_and_condition).
        # So we override forward to handle the specific call structure if needed.
        # Actually, let's override the base class forward if we want specific signature,
        # but base class forward is just self.network(input).
        # We'll handle the input unpacking in _step.
        return self.network(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def _step(self, batch):
        """Perform a step."""
        # Batch from ZydaDataModule is dict: {'input_ids': ..., 'attention_mask': ..., 'labels': ...}
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # Forward pass
        out = self.network({"input": input_ids, "condition": None})
        logits = out["logits"]

        # Shift logits and labels for next token prediction
        # Logits: [B, T, V] -> [B, T-1, V]
        # Labels: [B, T] -> [B, T-1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Loss
        loss = self.loss_fn(shift_logits, shift_labels)

        return loss, shift_logits, shift_labels

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss, logits, labels = self._step(batch)

        # Metrics
        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, labels)
        self.train_acc_top5(logits, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log("train/ppl", torch.exp(loss), on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log(
            "train/acc_top5",
            self.train_acc_top5,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.distributed,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss, logits, labels = self._step(batch)

        # Metrics
        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, labels)
        self.val_acc_top5(logits, labels)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log("val/ppl", torch.exp(loss), on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log("val/acc_top5", self.val_acc_top5, on_epoch=True, prog_bar=True, sync_dist=self.distributed)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        loss, logits, labels = self._step(batch)

        # Metrics
        preds = torch.argmax(logits, dim=-1)
        self.test_acc(preds, labels)
        self.test_acc_top5(logits, labels)

        self.log("test/loss", loss, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log("test/ppl", torch.exp(loss), on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log("test/acc", self.test_acc, on_epoch=True, prog_bar=True, sync_dist=self.distributed)
        self.log("test/acc_top5", self.test_acc_top5, on_epoch=True, prog_bar=True, sync_dist=self.distributed)

        return loss

    def on_validation_epoch_end(self):
        """Log best validation metrics and generate text samples."""
        val_loss = self.trainer.callback_metrics.get("val/loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss.item()
            self.best_val_ppl = torch.exp(val_loss).item()
            if self.logger:
                self.logger.experiment.log(
                    {
                        "val/best_loss": self.best_val_loss,
                        "val/best_ppl": self.best_val_ppl,
                        "global_step": self.global_step,
                    }
                )

        # Text Generation
        if (
            self.cfg.text_generation.enabled
            and (self.current_epoch + 1) % self.cfg.text_generation.every_n_epochs == 0
        ):
            self.generate_text_samples()

    def generate_text_samples(self):
        """Generate text samples and log to WandB."""
        if not hasattr(self, "tokenizer"):
            # Lazy load tokenizer to avoid overhead if not needed or on non-main processes
            from transformers import AutoTokenizer

            try:
                # Try to get tokenizer name from dataset config if available
                tokenizer_name = self.cfg.dataset.tokenizer_name
            except AttributeError:
                # Fallback or default
                tokenizer_name = "nvidia/Mistral-NeMo-Minitron-8B-Base"

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        device = self.device
        num_samples = self.cfg.text_generation.num_samples
        max_new_tokens = self.cfg.text_generation.max_new_tokens
        temperature = self.cfg.text_generation.temperature
        top_k = self.cfg.text_generation.top_k

        # Prompts (you can customize these)
        prompts = [
            "The quick brown fox",
            "Once upon a time",
            "In a galaxy far, far away",
            "The future of AI is",
        ][:num_samples]

        # If fewer prompts than num_samples, repeat or add more generic ones
        while len(prompts) < num_samples:
            prompts.append("The")

        generated_texts = []

        self.network.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                # Simple autoregressive generation loop
                curr_ids = input_ids.clone()

                for _ in range(max_new_tokens):
                    # Forward pass
                    # Network expects dict
                    out = self.network({"input": curr_ids, "condition": None})
                    logits = out["logits"]

                    # Get last token logits: [B, T, V] -> [B, 1, V]
                    next_token_logits = logits[:, -1, :]

                    # Temperature
                    next_token_logits = next_token_logits / temperature

                    # Top-k filtering
                    if top_k > 0:
                        v, _ = torch.topk(next_token_logits, top_k)
                        next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

                    # Sample
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Append
                    curr_ids = torch.cat([curr_ids, next_token], dim=1)

                    # Stop if EOS (optional, but good for chat models)
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                decoded_text = self.tokenizer.decode(curr_ids[0], skip_special_tokens=True)
                generated_texts.append([prompt, decoded_text])

        self.network.train()

        # Log to WandB
        if self.logger:
            columns = ["Prompt", "Generated Text"]
            self.logger.experiment.log(
                {
                    "val/generated_samples": wandb.Table(columns=columns, data=generated_texts),
                    "global_step": self.global_step,
                }
            )
