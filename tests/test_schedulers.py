import pytest
import torch

from nvsubq_paper.modules.schedulers import WSDScheduler


class TestWSDScheduler:
    @pytest.fixture
    def optimizer(self):
        # Create a simple optimizer
        model = torch.nn.Linear(10, 1)
        return torch.optim.SGD(model.parameters(), lr=1.0)

    def test_wsd_phases(self, optimizer):
        """Test Warmup-Stable-Decay phases with strict checks from verify_wsd.py."""
        total_iterations = 100
        warmup_iterations = 10
        decay_iterations_percentage = 0.1  # 10 steps

        scheduler = WSDScheduler(
            optimizer,
            total_iterations=total_iterations,
            warmup_iterations=warmup_iterations,
            decay_iterations_percentage=decay_iterations_percentage,
            min_lr_ratio=0.0,
        )

        # Step 0 (After 1 update)
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        # Based on verification, last_epoch became 1, so LR became 0.1
        # If last_epoch starts at -1, first step makes it 0 -> LR=0.0
        # If verify_wsd showed 0.1 at first step, implies last_epoch=1.
        # We will check approximate values to be safe or check the specific logic we validated.

        # If we trust the verification script output:
        # Step 0 (after update) -> 0.1
        assert lr == pytest.approx(0.1, abs=1e-6)

        # Step 5 (After 5 more updates -> index 6)
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        # Expected: 0.6
        assert lr == pytest.approx(0.6, abs=1e-6)

        # Step 10 (Stable Start)
        # We were at index 6. Need to reach index 11 (Step 10 check in script was 5 more steps?)
        # verify_wsd.py:
        #   Step 5 check done.
        #   Step 10 loop: range(5). Total steps = 1+5+5 = 11.
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(1.0, abs=1e-6)

        # Step 89 (Stable End)
        # verify_wsd.py: range(79). Total steps = 11 + 79 = 90.
        # last_epoch should be 90. LR should be 1.0.
        for _ in range(79):
            optimizer.step()
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(1.0, abs=1e-6)

        # Step 90 (Decay Start)
        optimizer.step()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        # verify_wsd.py showed 0.9.
        assert lr == pytest.approx(0.9, abs=1e-6)

        # Step 95 (Mid Decay)
        # verify_wsd.py: range(5). Total 91 + 5 = 96.
        # last_epoch = 96.
        # LR should be 0.4.
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(0.4, abs=1e-6)

        # Step 100 (End)
        # verify_wsd.py: range(5). Total 96 + 5 = 101.
        # last_epoch = 101.
        # LR should be 0.0.
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(0.0, abs=1e-6)

    def test_no_warmup(self, optimizer):
        scheduler = WSDScheduler(optimizer, total_iterations=10, warmup_iterations=0, decay_iterations_percentage=0.2)
        # Decay starts at 10 - 2 = 8.

        # Step 0: Stable
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

        for _ in range(7):
            optimizer.step()
            scheduler.step()
        # Step 8 (total): Stable
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)

        optimizer.step()
        scheduler.step()
        # Step 9: Decay step 1.
        # last_epoch=9. decay_step=1. progress=0.5. ratio=1-(0.99*0.5)=0.505
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.505)

        optimizer.step()
        scheduler.step()
        # End of training
