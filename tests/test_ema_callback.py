"""Unit tests for the shared EMACallback."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from experiments.callbacks.ema import EMACallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleNetwork(torch.nn.Module):
    """Minimal network for testing."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim, bias=False)
        # Use a known weight for reproducibility
        torch.nn.init.ones_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FakeLightningModule:
    """Minimal stand-in for a pl.LightningModule with a .network attribute."""

    def __init__(self, network: torch.nn.Module):
        self.network = network
        self.device = torch.device("cpu")


class _FakeTrainer:
    """Minimal stand-in for a pl.Trainer."""

    def __init__(self, global_step: int = 0):
        self.global_step = global_step


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEMACallbackInit:
    """Test callback initialization and shadow model creation."""

    def test_on_fit_start_creates_shadow_model(self):
        cb = EMACallback(decay=0.99)
        net = _SimpleNetwork()
        pl_module = _FakeLightningModule(net)

        cb.on_fit_start(trainer=None, pl_module=pl_module)

        assert cb._ema_model is not None
        assert cb._ema_model is not net  # must be a separate copy
        # Weights should be equal initially
        assert torch.equal(
            cb._ema_model.linear.weight.data,
            net.linear.weight.data,
        )

    def test_shadow_model_is_detached_and_frozen(self):
        cb = EMACallback()
        net = _SimpleNetwork()
        pl_module = _FakeLightningModule(net)

        cb.on_fit_start(trainer=None, pl_module=pl_module)

        for p in cb._ema_model.parameters():
            assert not p.requires_grad

    def test_shadow_model_is_independent(self):
        """Mutating the training model should not affect the EMA shadow."""
        cb = EMACallback()
        net = _SimpleNetwork()
        pl_module = _FakeLightningModule(net)
        cb.on_fit_start(trainer=None, pl_module=pl_module)

        original_ema_weight = cb._ema_model.linear.weight.data.clone()
        net.linear.weight.data.fill_(99.0)

        assert torch.equal(cb._ema_model.linear.weight.data, original_ema_weight)


class TestEMAUpdate:
    """Test EMA parameter update logic."""

    def _setup(self, decay=0.9, warmup_steps=0):
        cb = EMACallback(decay=decay, warmup_steps=warmup_steps)
        net = _SimpleNetwork()
        pl_module = _FakeLightningModule(net)
        cb.on_fit_start(trainer=None, pl_module=pl_module)
        return cb, net, pl_module

    def test_ema_update_blends_weights(self):
        decay = 0.9
        cb, net, pl_module = self._setup(decay=decay)

        original_ema = cb._ema_model.linear.weight.data.clone()
        # Change training weights
        net.linear.weight.data.fill_(2.0)

        trainer = _FakeTrainer(global_step=1)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        expected = original_ema * decay + 2.0 * (1 - decay)
        assert torch.allclose(cb._ema_model.linear.weight.data, expected)

    def test_warmup_skips_update(self):
        cb, net, pl_module = self._setup(warmup_steps=100)

        original_ema = cb._ema_model.linear.weight.data.clone()
        net.linear.weight.data.fill_(99.0)

        # global_step < warmup_steps → should not update
        trainer = _FakeTrainer(global_step=50)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        assert torch.equal(cb._ema_model.linear.weight.data, original_ema)

    def test_update_after_warmup(self):
        cb, net, pl_module = self._setup(decay=0.5, warmup_steps=10)

        original_ema = cb._ema_model.linear.weight.data.clone()
        net.linear.weight.data.fill_(3.0)

        trainer = _FakeTrainer(global_step=10)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        expected = original_ema * 0.5 + 3.0 * 0.5
        assert torch.allclose(cb._ema_model.linear.weight.data, expected)

    def test_update_every_skips_intermediate_steps(self):
        cb = EMACallback(decay=0.9, warmup_steps=0, update_every=3)
        net = _SimpleNetwork()
        pl_module = _FakeLightningModule(net)
        cb.on_fit_start(trainer=None, pl_module=pl_module)

        original_ema = cb._ema_model.linear.weight.data.clone()
        net.linear.weight.data.fill_(5.0)

        # Step 1: not divisible by 3 → skip
        trainer = _FakeTrainer(global_step=1)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
        assert torch.equal(cb._ema_model.linear.weight.data, original_ema)

        # Step 3: divisible by 3 → update
        trainer = _FakeTrainer(global_step=3)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
        assert not torch.equal(cb._ema_model.linear.weight.data, original_ema)


class TestEMASwap:
    """Test validation/test network pointer swapping."""

    def _setup(self):
        cb = EMACallback(decay=0.9, warmup_steps=0)
        net = _SimpleNetwork()
        pl_module = _FakeLightningModule(net)
        cb.on_fit_start(trainer=None, pl_module=pl_module)
        # Diverge the EMA from training weights
        net.linear.weight.data.fill_(10.0)
        trainer = _FakeTrainer(global_step=1)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)
        return cb, net, pl_module

    def test_validation_swaps_to_ema(self):
        cb, net, pl_module = self._setup()
        ema_model = cb._ema_model

        cb.on_validation_start(trainer=None, pl_module=pl_module)
        assert pl_module.network is ema_model

    def test_validation_restores_training(self):
        cb, net, pl_module = self._setup()

        cb.on_validation_start(trainer=None, pl_module=pl_module)
        cb.on_validation_end(trainer=None, pl_module=pl_module)
        assert pl_module.network is net

    def test_test_swaps_to_ema(self):
        cb, net, pl_module = self._setup()
        ema_model = cb._ema_model

        cb.on_test_start(trainer=None, pl_module=pl_module)
        assert pl_module.network is ema_model

    def test_test_restores_training(self):
        cb, net, pl_module = self._setup()

        cb.on_test_start(trainer=None, pl_module=pl_module)
        cb.on_test_end(trainer=None, pl_module=pl_module)
        assert pl_module.network is net

    def test_no_swap_before_first_update(self):
        """If EMA has never been updated, validation should use training weights."""
        cb = EMACallback(decay=0.9, warmup_steps=0)
        net = _SimpleNetwork()
        pl_module = _FakeLightningModule(net)
        cb.on_fit_start(trainer=None, pl_module=pl_module)
        # No on_train_batch_end called → _has_been_updated is False

        cb.on_validation_start(trainer=None, pl_module=pl_module)
        assert pl_module.network is net  # should NOT swap


class TestEMACheckpointing:
    """Test state_dict save/load for checkpoint resumption."""

    def test_roundtrip_state_dict(self):
        cb = EMACallback(decay=0.5, warmup_steps=0)
        net = _SimpleNetwork()
        pl_module = _FakeLightningModule(net)
        cb.on_fit_start(trainer=None, pl_module=pl_module)

        # Do a few updates to diverge EMA from init
        net.linear.weight.data.fill_(5.0)
        trainer = _FakeTrainer(global_step=1)
        cb.on_train_batch_end(trainer, pl_module, outputs=None, batch=None, batch_idx=0)

        ema_weight_before = cb._ema_model.linear.weight.data.clone()
        saved = cb.state_dict()

        # Create a fresh callback and load the state
        cb2 = EMACallback(decay=0.5, warmup_steps=0)
        net2 = _SimpleNetwork()
        pl_module2 = _FakeLightningModule(net2)
        cb2.on_fit_start(trainer=None, pl_module=pl_module2)
        cb2.load_state_dict(saved)

        assert torch.equal(cb2._ema_model.linear.weight.data, ema_weight_before)
        assert cb2._has_been_updated is True

    def test_load_state_dict_without_ema_model_is_noop(self):
        """Loading into a callback that hasn't called on_fit_start should not crash."""
        cb = EMACallback()
        cb.load_state_dict({"ema_model_state_dict": None, "has_been_updated": False})
        assert cb._ema_model is None
