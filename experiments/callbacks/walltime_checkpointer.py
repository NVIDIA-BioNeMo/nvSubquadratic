import datetime
import time
from pathlib import Path
from typing import Union

import pytorch_lightning as pl


class WalltimeCheckpointer(pl.Callback):
    """Checkpoints and stops the training when a walltime limit is reached.

    This is useful for Slurm jobs that have a hard time limit. We want to stop
    short of the limit to save a checkpoint and exit gracefully, so that the
    job can be automatically resumed.
    """

    def __init__(
        self,
        start_time: float,
        checkpoint_dir: Union[str, Path],
        time_limit_hours: float = 4.0,
        buffer_minutes: float = 2.0,
        checkpoint_filename: str = "last.ckpt",
        stop_trainer_after_time_limit: bool = True,
    ):
        """WalltimeCheckpointer constructor.

        Args:
            start_time: Timestamp when the job started (e.g. from time.time() or `date +%s`).
            checkpoint_dir: Directory where to save the checkpoint.
            time_limit_hours: The allocation time limit in hours.
            buffer_minutes: How many minutes before the limit to stop.
            checkpoint_filename: Name of the checkpoint file to save.
            stop_trainer_after_time_limit: Whether to stop the trainer after the time limit.

        """
        super().__init__()
        self.start_time = start_time
        # Convert limit to seconds
        self.time_limit_seconds = time_limit_hours * 3600
        self.buffer_seconds = buffer_minutes * 60
        self.checkpoint_filename = checkpoint_filename
        self.stopped = False
        self.stop_trainer_after_time_limit = stop_trainer_after_time_limit
        self.checkpoint_dir = checkpoint_dir

        # Calculate the absolute deadline timestamp
        self.deadline = self.start_time + self.time_limit_seconds - self.buffer_seconds

        # Log the config
        deadline_str = datetime.datetime.fromtimestamp(self.deadline).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[WalltimeCheckpointer] Checkpoint deadline set to {deadline_str} "
            f"(Start: {start_time}, Limit: {time_limit_hours}h, Buffer: {buffer_minutes}m)"
        )

    def on_val_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """Check time after every batch."""
        self.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """Check time after every batch."""
        if self.stopped:
            return

        current_time = time.time()
        if current_time >= self.deadline:
            print(
                f"[WalltimeCheckpointer] Deadline reached ({current_time:.2f} >= {self.deadline:.2f}). "
                f"Stopping training and saving checkpoint..."
            )

            # Determine checkpoint path
            # We want to save to the standard checkpoint directory so autoresume finds it.
            ckpt_dir = Path(self.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / self.checkpoint_filename

            print(f"[WalltimeCheckpointer] Saving checkpoint to {ckpt_path}...")
            trainer.save_checkpoint(ckpt_path)

            if self.stop_trainer_after_time_limit:
                print("[WalltimeCheckpointer] Checkpoint saved. Signaling trainer to stop.")
                trainer.should_stop = True
                self.stopped = True
