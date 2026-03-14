"""Tests for the MuonAdamW composite optimizer.

Verifies:
1. Parameter splitting: 2D hidden weights -> Muon, everything else -> AdamW
2. Weight-decay groups: _no_weight_decay, _weight_decay attributes are respected
3. Optimizer step: both sub-optimizers update parameters
4. Gradient flow: zero_grad clears all gradients
5. State dict round-trip: save and restore produces identical state
6. Scheduler compatibility: LR scheduler updates all param groups uniformly
7. Edge cases: models with no Muon-eligible params, single param, etc.
8. Exclusion patterns: embeddings, heads, classifier layers use AdamW

Run:
    PYTHONPATH=. conda run -n nv-subq pytest tests/test_muon_adamw.py -v
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.optimizers.muon_adamw import MuonAdamW, _is_muon_eligible


# ─── Fixtures ────────────────────────────────────────────────────────────────────


class ToyModel(nn.Module):
    """Small model with a mix of 2D weights, biases, norms, and embeddings."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.embed.weight._exclude_from_muon = True  # embedding -> AdamW
        self.linear1 = nn.Linear(32, 64)  # weight: Muon (2D), bias: AdamW (1D)
        self.norm = nn.LayerNorm(64)  # AdamW (1D)
        self.linear2 = nn.Linear(64, 32)  # weight: Muon (2D), bias: AdamW (1D)
        self.head = nn.Linear(32, 10)
        self.head.weight._exclude_from_muon = True  # classifier head -> AdamW

    def forward(self, x):
        x = self.embed(x)
        x = self.linear1(x)
        x = self.norm(x)
        x = self.linear2(x)
        return self.head(x)


class ToyModelWithCustomWD(nn.Module):
    """Model with _weight_decay and _no_weight_decay attributes."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 32)
        self.head = nn.Linear(32, 10)
        self.head.weight._exclude_from_muon = True  # classifier head -> AdamW

        # Mark linear2 weight with custom weight decay
        self.linear2.weight._weight_decay = 1e-3
        self.linear2.bias._weight_decay = 1e-3

        # Mark head as no weight decay
        self.head.weight._no_weight_decay = True
        self.head.bias._no_weight_decay = True

    def forward(self, x):
        return self.head(self.linear2(self.linear1(x)))


@pytest.fixture
def toy_model():
    return ToyModel()


@pytest.fixture
def toy_model_custom_wd():
    return ToyModelWithCustomWD()


def _fake_grads(model):
    """Assign random gradients to all parameters."""
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn_like(p)


# ─── 1. Eligibility Tests ───────────────────────────────────────────────────────


class TestMuonEligibility:
    """Test _is_muon_eligible correctly classifies parameters."""

    def test_2d_weight_eligible(self):
        p = nn.Parameter(torch.randn(64, 32))
        assert _is_muon_eligible("layer.weight", p) is True

    def test_1d_bias_not_eligible(self):
        p = nn.Parameter(torch.randn(64))
        assert _is_muon_eligible("layer.bias", p) is False

    def test_scalar_not_eligible(self):
        p = nn.Parameter(torch.tensor(1.0))
        assert _is_muon_eligible("scale", p) is False

    def test_3d_not_eligible(self):
        p = nn.Parameter(torch.randn(64, 32, 3))
        assert _is_muon_eligible("conv.weight", p) is False

    def test_4d_not_eligible(self):
        p = nn.Parameter(torch.randn(64, 32, 3, 3))
        assert _is_muon_eligible("conv.weight", p) is False

    def test_exclude_from_muon_attribute(self):
        p = nn.Parameter(torch.randn(64, 32))
        assert _is_muon_eligible("any.name", p) is True
        p._exclude_from_muon = True
        assert _is_muon_eligible("any.name", p) is False

    def test_exclude_from_muon_false_still_eligible(self):
        p = nn.Parameter(torch.randn(64, 32))
        p._exclude_from_muon = False
        assert _is_muon_eligible("any.name", p) is True

    def test_name_is_irrelevant(self):
        """With attribute-based exclusion, names don't affect eligibility."""
        p = nn.Parameter(torch.randn(64, 32))
        assert _is_muon_eligible("embed.weight", p) is True
        assert _is_muon_eligible("head.weight", p) is True
        assert _is_muon_eligible("film_generator.mlp.0.weight", p) is True


# ─── 2. Parameter Splitting Tests ───────────────────────────────────────────────


class TestParamSplitting:
    """Test that MuonAdamW correctly splits parameters."""

    def test_basic_split(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3, weight_decay=0.05)

        muon_groups = [g for g in opt.param_groups if g["_optimizer"] == "muon"]
        adamw_groups = [g for g in opt.param_groups if g["_optimizer"] == "adamw"]

        muon_params = {id(p) for g in muon_groups for p in g["params"]}
        adamw_params = {id(p) for g in adamw_groups for p in g["params"]}

        # linear1.weight and linear2.weight should be Muon (2D, no exclude pattern)
        assert id(toy_model.linear1.weight) in muon_params
        assert id(toy_model.linear2.weight) in muon_params

        # embed.weight should be AdamW (name contains "embed")
        assert id(toy_model.embed.weight) in adamw_params

        # head.weight should be AdamW (name contains "head")
        assert id(toy_model.head.weight) in adamw_params

        # All biases should be AdamW (1D)
        assert id(toy_model.linear1.bias) in adamw_params
        assert id(toy_model.linear2.bias) in adamw_params
        assert id(toy_model.head.bias) in adamw_params

        # Norm params should be AdamW (1D)
        assert id(toy_model.norm.weight) in adamw_params
        assert id(toy_model.norm.bias) in adamw_params

    def test_no_overlap(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        all_params = []
        for g in opt.param_groups:
            all_params.extend(id(p) for p in g["params"])
        assert len(all_params) == len(set(all_params)), "Duplicate parameter across groups"

    def test_all_params_accounted(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        model_param_ids = {id(p) for p in toy_model.parameters() if p.requires_grad}
        assert opt_param_ids == model_param_ids

    def test_custom_weight_decay(self, toy_model_custom_wd):
        opt = MuonAdamW(toy_model_custom_wd.named_parameters(), lr=1e-3, weight_decay=0.05)

        for g in opt.param_groups:
            for p in g["params"]:
                if id(p) == id(toy_model_custom_wd.linear2.weight):
                    assert g["weight_decay"] == 1e-3
                elif id(p) == id(toy_model_custom_wd.linear2.bias):
                    assert g["weight_decay"] == 1e-3
                elif id(p) == id(toy_model_custom_wd.head.weight):
                    assert g["weight_decay"] == 0.0
                elif id(p) == id(toy_model_custom_wd.head.bias):
                    assert g["weight_decay"] == 0.0


# ─── 3. Optimizer Step Tests ────────────────────────────────────────────────────


class TestOptimizerStep:
    """Test that step() updates all parameters."""

    def test_step_updates_params(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-2, weight_decay=0.0)
        params_before = {n: p.clone() for n, p in toy_model.named_parameters()}

        _fake_grads(toy_model)
        opt.step()

        for name, param in toy_model.named_parameters():
            assert not torch.equal(param, params_before[name]), f"Parameter {name} was not updated by optimizer step"

    def test_zero_grad(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        _fake_grads(toy_model)

        opt.zero_grad()
        for p in toy_model.parameters():
            assert p.grad is None or p.grad.abs().sum() == 0

    def test_zero_grad_set_to_none(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        _fake_grads(toy_model)

        opt.zero_grad(set_to_none=True)
        for p in toy_model.parameters():
            assert p.grad is None

    def test_multiple_steps(self, toy_model):
        """Verify optimizer can run multiple consecutive steps without error."""
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        for _ in range(5):
            _fake_grads(toy_model)
            opt.step()
            opt.zero_grad()


# ─── 4. State Dict Tests ────────────────────────────────────────────────────────


class TestStateDict:
    """Test state_dict save/load round-trip."""

    def test_state_dict_structure(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        _fake_grads(toy_model)
        opt.step()

        sd = opt.state_dict()
        assert "muon" in sd
        assert "adamw" in sd
        assert "state" in sd["muon"]
        assert "param_groups" in sd["muon"]
        assert "state" in sd["adamw"]
        assert "param_groups" in sd["adamw"]

    def test_round_trip(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        _fake_grads(toy_model)
        opt.step()

        sd = opt.state_dict()
        opt.load_state_dict(sd)

        sd2 = opt.state_dict()
        # Muon states should match
        assert len(sd["muon"]["state"]) == len(sd2["muon"]["state"])
        assert len(sd["adamw"]["state"]) == len(sd2["adamw"]["state"])

    def test_load_restores_momentum(self, toy_model):
        """Verify that momentum buffers are preserved across save/load."""
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        _fake_grads(toy_model)
        opt.step()

        sd = opt.state_dict()
        muon_state_count = len(sd["muon"]["state"])
        adamw_state_count = len(sd["adamw"]["state"])
        assert muon_state_count > 0
        assert adamw_state_count > 0

        # Create a fresh optimizer and load
        opt2 = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        opt2.load_state_dict(sd)

        sd_after = opt2.state_dict()
        assert len(sd_after["muon"]["state"]) == muon_state_count
        assert len(sd_after["adamw"]["state"]) == adamw_state_count


# ─── 5. Scheduler Compatibility Tests ───────────────────────────────────────────


class TestSchedulerCompatibility:
    """Test that standard LR schedulers work with MuonAdamW."""

    def test_cosine_annealing(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

        initial_lrs = [g["lr"] for g in opt.param_groups]
        scheduler.step()
        new_lrs = [g["lr"] for g in opt.param_groups]

        # All groups should have the same LR (all started at 1e-2)
        assert len({round(lr, 12) for lr in new_lrs}) == 1
        # LR should have changed
        assert new_lrs[0] != initial_lrs[0]

    def test_lr_stays_synced_across_groups(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=5e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        for _ in range(25):
            _fake_grads(toy_model)
            opt.step()
            scheduler.step()

        lrs = [g["lr"] for g in opt.param_groups]
        assert len({round(lr, 12) for lr in lrs}) == 1

    def test_linear_warmup(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-8, end_factor=1.0, total_iters=100)

        scheduler.step()
        lrs = {round(g["lr"], 15) for g in opt.param_groups}
        assert len(lrs) == 1, f"LR mismatch after warmup step: {lrs}"

    def test_param_groups_have_lr_key(self, toy_model):
        opt = MuonAdamW(toy_model.named_parameters(), lr=1e-3)
        for g in opt.param_groups:
            assert "lr" in g, f"param_group missing 'lr' key: {g.keys()}"


# ─── 6. Edge Cases ──────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and unusual model configurations."""

    def test_all_adamw_model(self):
        """Model with no 2D hidden weights (e.g., all 1D params)."""
        model = nn.Sequential(nn.LayerNorm(32), nn.LayerNorm(32))
        opt = MuonAdamW(model.named_parameters(), lr=1e-3)

        muon_count = sum(len(g["params"]) for g in opt.param_groups if g["_optimizer"] == "muon")
        assert muon_count == 0

        _fake_grads(model)
        opt.step()  # should not crash

    def test_excluded_embedding_goes_to_adamw(self):
        """Embedding marked _exclude_from_muon should go to AdamW."""
        model = nn.ModuleDict({"embed": nn.Embedding(100, 32)})
        model.embed.weight._exclude_from_muon = True
        opt = MuonAdamW(model.named_parameters(), lr=1e-3)

        adamw_params = {id(p) for g in opt.param_groups if g["_optimizer"] == "adamw" for p in g["params"]}
        assert id(model.embed.weight) in adamw_params

    def test_unmarked_2d_goes_to_muon(self):
        """2D param without _exclude_from_muon goes to Muon regardless of name."""
        model = nn.ModuleDict({"embed": nn.Embedding(100, 32)})
        opt = MuonAdamW(model.named_parameters(), lr=1e-3)

        muon_params = {id(p) for g in opt.param_groups if g["_optimizer"] == "muon" for p in g["params"]}
        assert id(model.embed.weight) in muon_params

    def test_single_linear(self):
        """Single linear layer: weight -> Muon, bias -> AdamW."""
        model = nn.Linear(32, 64)
        opt = MuonAdamW(model.named_parameters(), lr=1e-3)

        muon_params = {id(p) for g in opt.param_groups if g["_optimizer"] == "muon" for p in g["params"]}
        adamw_params = {id(p) for g in opt.param_groups if g["_optimizer"] == "adamw" for p in g["params"]}

        assert id(model.weight) in muon_params
        assert id(model.bias) in adamw_params

    def test_frozen_params_excluded(self):
        """Frozen parameters should not appear in any group."""
        model = ToyModel()
        model.embed.weight.requires_grad = False

        opt = MuonAdamW(model.named_parameters(), lr=1e-3)
        all_opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
        assert id(model.embed.weight) not in all_opt_params


# ─── 7. Integration with construct_optimizer ─────────────────────────────────────


class TestConstructOptimizer:
    """Test integration with the Lightning wrapper's construct_optimizer."""

    def test_construct_optimizer_muon(self, toy_model):
        from omegaconf import OmegaConf

        from experiments.lightning_wrappers.base_lightning_wrapper import construct_optimizer

        cfg = OmegaConf.create(
            {
                "__target__": "experiments.optimizers.muon_adamw.MuonAdamW",
                "params": None,
                "lr": 4e-3,
                "weight_decay": 0.05,
            }
        )

        optimizer = construct_optimizer(toy_model, cfg)
        assert isinstance(optimizer, MuonAdamW)

        # Verify it works
        _fake_grads(toy_model)
        optimizer.step()
        optimizer.zero_grad()


# ─── 8. Module-level _exclude_from_muon Tests ──────────────────────────────────


class TestModuleExclusions:
    """Verify that modules correctly mark parameters with _exclude_from_muon."""

    def test_siren_positional_embedding_marks_linear(self):
        from nvsubquadratic.modules.kernels_nd import SIRENPositionalEmbeddingND

        pos_emb = SIRENPositionalEmbeddingND(data_dim=2, embedding_dim=32, omega_0=10.0, L_cache=14, use_bias=True)
        assert getattr(pos_emb.linear.weight, "_exclude_from_muon", False) is True
        assert not getattr(pos_emb.linear.bias, "_exclude_from_muon", False)

    def test_random_fourier_positional_embedding_marks_linear(self):
        from nvsubquadratic.modules.kernels_nd import RandomFourierPositionalEmbeddingND

        pos_emb = RandomFourierPositionalEmbeddingND(
            data_dim=2, embedding_dim=32, omega_0=10.0, L_cache=14, use_bias=True
        )
        assert getattr(pos_emb.linear.weight, "_exclude_from_muon", False) is True

    def test_film_generator_no_exclusions(self):
        """All FiLM generator layers are Muon-eligible (2D weights, no exclusion)."""
        from nvsubquadratic.modules.film import KernelFiLMGenerator

        gen = KernelFiLMGenerator(cond_dim=384, kernel_hidden_dim=32, num_film_layers=3, film_hidden_dim=64)
        assert not getattr(gen.mlp[0].weight, "_exclude_from_muon", False)
        assert not getattr(gen.mlp[-1].weight, "_exclude_from_muon", False)

    def test_siren_kernel_hidden_linears_not_excluded(self):
        """SIREN hidden layers are genuine hidden weights and should be Muon-eligible."""
        from nvsubquadratic.modules.kernels_nd import SIRENKernelND

        kernel = SIRENKernelND(
            out_dim=384,
            data_dim=2,
            mlp_hidden_dim=32,
            num_layers=3,
            embedding_dim=32,
            omega_0=10.0,
            L_cache=14,
            use_bias=True,
        )
        for linear in kernel.hidden_linears:
            assert not getattr(linear.weight, "_exclude_from_muon", False)
        assert not getattr(kernel.out_linear.weight, "_exclude_from_muon", False)

    def test_siren_kernel_pos_embed_excluded(self):
        from nvsubquadratic.modules.kernels_nd import SIRENKernelND

        kernel = SIRENKernelND(
            out_dim=384,
            data_dim=2,
            mlp_hidden_dim=32,
            num_layers=3,
            embedding_dim=32,
            omega_0=10.0,
            L_cache=14,
            use_bias=True,
        )
        assert getattr(kernel.positional_embedding.linear.weight, "_exclude_from_muon", False) is True
