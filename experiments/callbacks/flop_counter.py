"""Callback to measure per-step FLOPs at fit start.

Runs one real ``training_step`` (forward + backward) on a single batch under
``torch.utils.flop_counter.FlopCounterMode``, and logs the counts to the
attached logger as a one-shot metric:

    ``flops/fwd``    – forward FLOPs for one batch (per rank)
    ``flops/bwd``    – backward FLOPs for one batch (per rank)
    ``flops/step``   – total per-step FLOPs (fwd + bwd, per rank)
    ``flops/step_global`` – ``flops/step * world_size``

``FlopCounterMode`` is a ``TorchDispatchMode`` and natively counts ``aten``
ops including ``fft_*`` / ``conv_*`` / ``matmul`` / SDPA, which matters here
because Hyena/CKConv use FFT-based long convolutions that ``fvcore`` /
``torchinfo`` do not see.

Notes:
-----
* If ``pl_module.network`` was wrapped by ``torch.compile`` (i.e. it has a
  ``_orig_mod`` attribute), the un-compiled module is swapped in for the
  measurement.  Dispatch modes do not always observe ops inside compiled
  graphs, so this gives a reliable count.
* The measurement runs on every rank (so the DDP gradient all-reduce stays
  synchronized during ``loss.backward()``) but only rank 0 logs.
* Gradients are zeroed and the original ``pl_module.network`` is restored
  before training starts, so this callback is non-invasive.
* With ``gradient_checkpointing=True`` the backward also re-runs the
  checkpointed forward chunks; those re-runs are counted in ``flops/bwd``,
  which is the correct accounting for true per-step training cost.
"""

from __future__ import annotations

import pytorch_lightning as pl
from torch.utils.flop_counter import FlopCounterMode


class FlopCounterCallback(pl.Callback):
    """Measure FLOPs for one fwd+bwd training step at fit start."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self._done = False

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: D102
        if self._done:
            return
        self._done = True

        # Fetch one training batch.  Prefer the trainer's already-built
        # dataloader; fall back to the datamodule.
        batch = self._fetch_batch(trainer)
        if batch is None:
            if trainer.is_global_zero:
                print("[flop_counter] could not fetch a training batch; skipping FLOP measurement.")
            return

        batch = pl_module._apply_batch_transfer_handler(batch, pl_module.device, dataloader_idx=0)

        # If torch.compile wrapped the network, run the count against the
        # un-compiled module so the dispatch mode sees every op.
        compiled_net = getattr(pl_module, "network", None)
        original_net = None
        if compiled_net is not None and hasattr(compiled_net, "_orig_mod"):
            original_net = compiled_net
            pl_module.network = compiled_net._orig_mod

        # Suppress ``self.log`` calls that wrappers normally make inside
        # ``training_step`` — we are not inside the actual training loop hook.
        original_log = pl_module.log
        pl_module.log = lambda *a, **k: None

        try:
            with FlopCounterMode(display=False) as fc:
                with trainer.precision_plugin.forward_context():
                    output = pl_module.training_step(batch, 0)
                fwd_flops = fc.get_total_flops()

                loss = output["loss"] if isinstance(output, dict) else output
                loss.backward()
                total_flops = fc.get_total_flops()
        except Exception as e:
            pl_module.log = original_log
            if original_net is not None:
                pl_module.network = original_net
            pl_module.zero_grad(set_to_none=True)
            if trainer.is_global_zero:
                print(f"[flop_counter] measurement failed ({type(e).__name__}: {e}); skipping.")
            return

        pl_module.log = original_log
        if original_net is not None:
            pl_module.network = original_net
        pl_module.zero_grad(set_to_none=True)

        bwd_flops = total_flops - fwd_flops
        world_size = trainer.world_size if trainer.world_size else 1

        if trainer.is_global_zero:
            payload = {
                "flops/fwd": float(fwd_flops),
                "flops/bwd": float(bwd_flops),
                "flops/step": float(total_flops),
                "flops/step_global": float(total_flops * world_size),
            }
            bs = getattr(getattr(trainer, "datamodule", None), "batch_size", "?")
            print(
                f"[flop_counter] per-rank batch_size={bs}  "
                f"fwd={fwd_flops / 1e12:.3f} TFLOPs  "
                f"bwd={bwd_flops / 1e12:.3f} TFLOPs  "
                f"step={total_flops / 1e12:.3f} TFLOPs  "
                f"(global step ≈ {total_flops * world_size / 1e12:.3f} TFLOPs across {world_size} rank(s))"
            )
            if trainer.logger is not None:
                trainer.logger.log_metrics(payload, step=0)

    @staticmethod
    def _fetch_batch(trainer: pl.Trainer):
        dl = getattr(trainer, "train_dataloader", None)
        if dl is None and getattr(trainer, "datamodule", None) is not None:
            try:
                dl = trainer.datamodule.train_dataloader()
            except Exception:
                dl = None
        if dl is None:
            return None
        try:
            return next(iter(dl))
        except Exception:
            return None
