"""Tests for IterationSpeedCallback."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from experiments.callbacks.iteration_speed import IterationSpeedCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(global_step: int = 0, world_size: int = 1) -> MagicMock:
    trainer = MagicMock()
    trainer.global_step = global_step
    trainer.is_global_zero = True
    trainer.world_size = world_size
    trainer.datamodule = MagicMock()
    trainer.datamodule.batch_size = 16
    return trainer


def _make_pl_module() -> MagicMock:
    return MagicMock()


def _run_warmup(cb: IterationSpeedCallback, trainer: MagicMock, pl_module: MagicMock) -> None:
    """Advance the callback past its warmup phase."""
    for i in range(cb._warmup_batches + 2):
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
        cb.on_before_backward(trainer, pl_module, loss=MagicMock())
        cb.on_after_backward(trainer, pl_module)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        cb = IterationSpeedCallback()
        assert cb.log_every_n_steps == 10
        assert cb.window_size == 10
        assert cb.batch_size_per_gpu is None

    def test_custom_window(self):
        cb = IterationSpeedCallback(log_every_n_steps=5, window_size=20)
        assert cb.log_every_n_steps == 5
        assert cb.window_size == 20

    def test_window_defaults_to_log_every(self):
        cb = IterationSpeedCallback(log_every_n_steps=7)
        assert cb.window_size == 7


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:
    def test_no_logging_during_warmup(self):
        cb = IterationSpeedCallback(log_every_n_steps=1)
        trainer = _make_trainer(global_step=0)
        pl_module = _make_pl_module()

        for i in range(cb._warmup_batches + 1):
            trainer.global_step = i
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            cb.on_after_backward(trainer, pl_module)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)

        pl_module.log_dict.assert_not_called()

    def test_no_times_recorded_during_warmup(self):
        cb = IterationSpeedCallback()
        trainer = _make_trainer()
        pl_module = _make_pl_module()

        for i in range(cb._warmup_batches):
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)

        assert len(cb._iter_times) == 0
        assert len(cb._fwd_times) == 0
        assert len(cb._bwd_times) == 0


# ---------------------------------------------------------------------------
# Start-to-start timing
# ---------------------------------------------------------------------------


class TestStartToStartTiming:
    def test_iter_time_measures_start_to_start(self):
        """Total iteration time should be the gap between batch_start calls."""
        cb = IterationSpeedCallback(log_every_n_steps=1, window_size=5)
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        cb._iter_times.clear()
        cb._fwd_times.clear()
        cb._bwd_times.clear()

        sleep_sec = 0.05
        for i in range(5):
            trainer.global_step = 100 + i
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            time.sleep(sleep_sec)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            time.sleep(sleep_sec)
            cb.on_after_backward(trainer, pl_module)
            # Simulate other callback overhead between after_backward and next start
            time.sleep(sleep_sec)
        # One more start to close the last iteration
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=5)

        assert len(cb._iter_times) == 5
        for t in cb._iter_times:
            # Each iteration should be ~3 * sleep_sec (fwd + bwd + other)
            assert t >= sleep_sec * 2.5, f"iter time {t:.4f} too short"

    def test_other_callback_work_not_in_fwd_bwd(self):
        """Work after after_backward should appear in other_ms, not fwd/bwd."""
        cb = IterationSpeedCallback(log_every_n_steps=1, window_size=5)
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        cb._iter_times.clear()
        cb._fwd_times.clear()
        cb._bwd_times.clear()

        compute_sleep = 0.02
        overhead_sleep = 0.08

        for i in range(5):
            trainer.global_step = 100 + i
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            time.sleep(compute_sleep)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            time.sleep(compute_sleep)
            cb.on_after_backward(trainer, pl_module)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)
            # Simulate overhead between iterations (data loading, other callbacks)
            time.sleep(overhead_sleep)

        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=5)

        avg_fwd = sum(cb._fwd_times) / len(cb._fwd_times)
        avg_bwd = sum(cb._bwd_times) / len(cb._bwd_times)
        avg_iter = sum(cb._iter_times) / len(cb._iter_times)
        avg_other = avg_iter - avg_fwd - avg_bwd

        # fwd and bwd should each be ~compute_sleep
        assert avg_fwd < compute_sleep * 2, f"fwd {avg_fwd:.4f} too large"
        assert avg_bwd < compute_sleep * 2, f"bwd {avg_bwd:.4f} too large"
        # other should capture the overhead
        assert avg_other > overhead_sleep * 0.5, f"other {avg_other:.4f} too small"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestLogging:
    def test_logs_at_correct_interval(self):
        cb = IterationSpeedCallback(log_every_n_steps=5)
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        # Run 10 more steps; should log at global_step 5 and 10
        for i in range(10):
            trainer.global_step = i + 1
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            cb.on_after_backward(trainer, pl_module)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)

        logged_calls = pl_module.log_dict.call_args_list
        assert len(logged_calls) == 2

    def test_logged_keys(self):
        cb = IterationSpeedCallback(log_every_n_steps=1)
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        for i in range(5):
            trainer.global_step = 100 + i
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            time.sleep(0.01)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            time.sleep(0.01)
            cb.on_after_backward(trainer, pl_module)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)
        # Final start to record last iter time
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=5)
        trainer.global_step = 105
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=5)

        logged = pl_module.log_dict.call_args_list[-1][0][0]
        expected_keys = {
            # Windowed breakdown
            "perf/iter_per_sec",
            "perf/samples_per_sec",
            "perf/total_ms",
            "perf/fwd_ms",
            "perf/bwd_ms",
            "perf/other_ms",
            # Wall-clock cumulative
            "perf/wc_iter_per_sec",
            "perf/wc_samples_per_sec",
            "perf/wc_total_sec",
            # GPU memory
            "perf/peak_gpu_mb",
            "perf/current_gpu_mb",
        }
        assert set(logged.keys()) == expected_keys

    def test_logged_values_are_positive(self):
        cb = IterationSpeedCallback(log_every_n_steps=1)
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        for i in range(5):
            trainer.global_step = 100 + i
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            time.sleep(0.01)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            time.sleep(0.01)
            cb.on_after_backward(trainer, pl_module)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=5)
        trainer.global_step = 105
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=5)

        logged = pl_module.log_dict.call_args_list[-1][0][0]
        for key, val in logged.items():
            assert val > 0, f"{key} = {val} should be positive"

    def test_samples_per_sec_uses_batch_size(self):
        cb = IterationSpeedCallback(log_every_n_steps=1, batch_size_per_gpu=32)
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        for i in range(5):
            trainer.global_step = 100 + i
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            time.sleep(0.005)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            time.sleep(0.005)
            cb.on_after_backward(trainer, pl_module)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=5)
        trainer.global_step = 105
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=5)

        logged = pl_module.log_dict.call_args_list[-1][0][0]
        assert logged["perf/samples_per_sec"] == pytest.approx(logged["perf/iter_per_sec"] * 32, rel=1e-5)

    def test_no_log_for_non_global_zero(self):
        cb = IterationSpeedCallback(log_every_n_steps=1)
        trainer = _make_trainer()
        trainer.is_global_zero = False
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        for i in range(5):
            trainer.global_step = 100 + i
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            cb.on_after_backward(trainer, pl_module)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)

        pl_module.log_dict.assert_not_called()


# ---------------------------------------------------------------------------
# Validation reset
# ---------------------------------------------------------------------------


class TestValidationReset:
    def test_validation_resets_timing_state(self):
        cb = IterationSpeedCallback()
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=0)
        assert cb._prev_start is not None

        cb.on_validation_start(trainer, pl_module)
        assert cb._prev_start is None
        assert cb._batch_start is None
        assert cb._bwd_start is None

        cb.on_validation_end(trainer, pl_module)
        assert cb._prev_start is None

    def test_no_spurious_time_after_validation(self):
        """First batch after validation should not record an iter_time."""
        cb = IterationSpeedCallback(log_every_n_steps=1, window_size=5)
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        n_before = len(cb._iter_times)

        cb.on_validation_start(trainer, pl_module)
        time.sleep(0.05)  # simulate long validation
        cb.on_validation_end(trainer, pl_module)

        # First batch after validation
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=99)

        # _prev_start was None, so no iter_time should be appended
        assert len(cb._iter_times) == n_before


# ---------------------------------------------------------------------------
# Window size
# ---------------------------------------------------------------------------


class TestWindowSize:
    def test_deque_respects_maxlen(self):
        window = 3
        cb = IterationSpeedCallback(log_every_n_steps=1, window_size=window)
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        for i in range(20):
            trainer.global_step = 100 + i
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            cb.on_after_backward(trainer, pl_module)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=20)

        assert len(cb._iter_times) == window
        assert len(cb._fwd_times) == window
        assert len(cb._bwd_times) == window


# ---------------------------------------------------------------------------
# Timing consistency: fwd + bwd + other ≈ total
# ---------------------------------------------------------------------------


class TestTimingConsistency:
    def test_fwd_bwd_other_sum_to_total(self):
        cb = IterationSpeedCallback(log_every_n_steps=1, window_size=10)
        trainer = _make_trainer()
        pl_module = _make_pl_module()
        _run_warmup(cb, trainer, pl_module)

        cb._iter_times.clear()
        cb._fwd_times.clear()
        cb._bwd_times.clear()

        for i in range(10):
            trainer.global_step = 200 + i
            cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=i)
            time.sleep(0.01)
            cb.on_before_backward(trainer, pl_module, loss=MagicMock())
            time.sleep(0.015)
            cb.on_after_backward(trainer, pl_module)
            cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=i)
            time.sleep(0.005)  # between-iteration overhead
        cb.on_train_batch_start(trainer, pl_module, batch=None, batch_idx=10)

        avg_iter = sum(cb._iter_times) / len(cb._iter_times)
        avg_fwd = sum(cb._fwd_times) / len(cb._fwd_times)
        avg_bwd = sum(cb._bwd_times) / len(cb._bwd_times)

        # fwd + bwd should be <= total (other fills the gap)
        assert avg_fwd + avg_bwd <= avg_iter * 1.05  # 5% tolerance for timing jitter
        # other = total - fwd - bwd, should be >= 0
        assert avg_iter - avg_fwd - avg_bwd >= -0.001
