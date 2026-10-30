"""Shared, lightweight campaign inputs for configuration and CLI tests."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def config_data(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "system.pdb").write_text("END\n", encoding="utf-8")
    (tmp_path / "def2-tzvpd.bas").write_text("test basis fixture\n", encoding="utf-8")
    return {
        "system": {
            "pdb": "system.pdb",
            "forcefields": ["amber14-all.xml", "amber14/tip3p.xml"],
            "qm_atoms": [0, 1, 2],
            "qm_charge": 0,
            "qm_multiplicity": 1,
        },
        "orca": {"basis_file": "def2-tzvpd.bas"},
        "conditions": [
            {"name": "water-300K", "temperature_kelvin": 300, "steps": 100, "seed": 42}
        ],
        "output_dir": "runs",
    }


@pytest.fixture
def write_config(tmp_path: Path) -> Callable[[dict[str, Any]], Path]:
    def write(data: dict[str, Any]) -> Path:
        path = tmp_path / "campaign.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    return write


@pytest.fixture
def config_path(
    config_data: dict[str, Any], write_config: Callable[[dict[str, Any]], Path]
) -> Path:
    return write_config(config_data)
