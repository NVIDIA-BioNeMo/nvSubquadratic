# TODO: Add license header here

"""Tests for WellDataModule staging path resolution (Fix 3: DDP safety).

Verifies that ``_resolve_staged_path`` correctly updates ``well_base_path``
when a staging sentinel exists (simulating what every DDP rank sees in
``setup()``), and leaves it unchanged otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.datamodules.pde.well import WellDataModule


@pytest.fixture
def make_datamodule(tmp_path: Path):
    """Factory that creates a WellDataModule pointing at tmp directories."""

    def _make(*, create_sentinel: bool = False):
        network_dir = tmp_path / "network_storage"
        local_dir = tmp_path / "nvme"
        dataset_name = "gray_scott_reaction_diffusion"

        (network_dir / dataset_name).mkdir(parents=True)

        if create_sentinel:
            staged = local_dir / dataset_name
            staged.mkdir(parents=True)
            (staged / ".staging_complete").write_text("ok\n")

        dm = WellDataModule(
            well_base_path=str(network_dir),
            well_dataset_name=dataset_name,
            local_staging_dir=str(local_dir),
            batch_size=1,
            num_workers=0,
        )
        return dm, str(network_dir), str(local_dir)

    return _make


class TestResolveStagedPath:
    """_resolve_staged_path ensures all DDP ranks see the staged directory."""

    def test_resolves_when_sentinel_exists(self, make_datamodule) -> None:
        """When staging is complete, well_base_path should point to local dir."""
        dm, network_dir, local_dir = make_datamodule(create_sentinel=True)
        assert dm.well_base_path == network_dir

        dm._resolve_staged_path()

        assert dm.well_base_path == local_dir

    def test_no_change_without_sentinel(self, make_datamodule) -> None:
        """Without sentinel, well_base_path stays at the original network dir."""
        dm, network_dir, _local_dir = make_datamodule(create_sentinel=False)
        assert dm.well_base_path == network_dir

        dm._resolve_staged_path()

        assert dm.well_base_path == network_dir

    def test_no_change_without_staging_dir(self, tmp_path: Path) -> None:
        """When local_staging_dir is None, _resolve_staged_path is a no-op."""
        dataset_name = "gray_scott_reaction_diffusion"
        base = tmp_path / "data"
        (base / dataset_name).mkdir(parents=True)

        dm = WellDataModule(
            well_base_path=str(base),
            well_dataset_name=dataset_name,
            local_staging_dir=None,
            batch_size=1,
            num_workers=0,
        )
        original = dm.well_base_path
        dm._resolve_staged_path()
        assert dm.well_base_path == original

    def test_idempotent(self, make_datamodule) -> None:
        """Calling _resolve_staged_path multiple times is safe."""
        dm, _network_dir, local_dir = make_datamodule(create_sentinel=True)
        dm._resolve_staged_path()
        dm._resolve_staged_path()
        assert dm.well_base_path == local_dir
