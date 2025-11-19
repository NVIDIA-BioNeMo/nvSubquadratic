"""Callback to mark when training is complete for job chaining."""

from pathlib import Path

from pytorch_lightning import Callback, LightningModule, Trainer


class TrainingCompletionCallback(Callback):
    """Creates a completion marker file when training finishes successfully.

    This is useful for SLURM job chaining where we need to know if training
    completed or if it was interrupted and needs to continue.

    Args:
        checkpoint_dir: Directory where checkpoints are saved. The completion
            marker will be created here.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        """Initialize the callback.

        Args:
            checkpoint_dir: Directory where checkpoints are saved.
        """
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.completion_marker = self.checkpoint_dir / ".training_complete"

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when training ends.

        Creates a completion marker file if training reached max_steps/max_epochs.
        """
        # Only create completion marker if we actually finished training
        # (not if training was interrupted or stopped early)
        if trainer.max_steps is not None and trainer.global_step >= trainer.max_steps:
            print(f"[TrainingCompletionCallback] Training reached max_steps={trainer.max_steps}")
            print(f"[TrainingCompletionCallback] Creating completion marker: {self.completion_marker}")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.completion_marker.touch()
        elif trainer.max_epochs is not None and trainer.current_epoch >= trainer.max_epochs:
            print(f"[TrainingCompletionCallback] Training reached max_epochs={trainer.max_epochs}")
            print(f"[TrainingCompletionCallback] Creating completion marker: {self.completion_marker}")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.completion_marker.touch()
        else:
            print(
                f"[TrainingCompletionCallback] Training ended early (step={trainer.global_step}/{trainer.max_steps})"
            )
            print("[TrainingCompletionCallback] Not creating completion marker (may need to resume)")
