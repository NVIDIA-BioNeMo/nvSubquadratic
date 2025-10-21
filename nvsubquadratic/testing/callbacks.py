# TODO: Add license header here

"""Testing callbacks for gradient logging and validation."""

import json
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl


class GradientLoggingCallback(pl.Callback):
    """Callback to log gradients for CP equivalence testing.

    Logs gradient statistics AFTER DDP all-reduce (gradients are averaged).
    Use `on_before_optimizer_step` to capture gradients after DDP sync but
    before optimizer modifies them.

    Args:
        save_dir: Directory to save gradient logs
        log_every_n_steps: Log gradients every N steps
        max_steps: Stop after logging this many steps
    """

    def __init__(
        self,
        save_dir: Path,
        log_every_n_steps: int = 1,
        max_steps: Optional[int] = None,
    ):
        """Initialize the callback."""
        super().__init__()
        self.save_dir = Path(save_dir)
        self.log_every_n_steps = log_every_n_steps
        self.max_steps = max_steps
        self.logged_steps = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Save gradients AFTER DDP all-reduce, BEFORE optimizer step."""
        current_step = trainer.global_step + 1

        if current_step % self.log_every_n_steps != 0:
            return

        if self.max_steps and self.logged_steps >= self.max_steps:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Get unwrapped model for clean parameter names
        model = pl_module.module if hasattr(pl_module, "module") else pl_module

        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = {
                    "shape": list(param.grad.shape),
                    "norm": param.grad.norm().item(),
                    "mean": param.grad.mean().item(),
                    "std": param.grad.std().item(),
                    "min": param.grad.min().item(),
                    "max": param.grad.max().item(),
                }

        # Save to file
        rank = trainer.global_rank
        save_path = self.save_dir / f"gradients_rank{rank}_step{current_step}.json"

        with open(save_path, "w") as f:
            json.dump(gradients, f, indent=2)

        self.logged_steps += 1

        if self.max_steps and self.logged_steps >= self.max_steps:
            trainer.should_stop = True
