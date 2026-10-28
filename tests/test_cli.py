"""Check reviewable dry runs, execution dispatch, and useful CLI failures."""

import json
from pathlib import Path
from typing import Any

import pytest

from mace_trainer import __main__ as cli
from mace_trainer import __version__
from mace_trainer.config import Campaign


@pytest.fixture
def cli_config(tmp_path: Path) -> Path:
    (tmp_path / "system.pdb").write_text("END\n", encoding="utf-8")
    (tmp_path / "def2-tzvpd.bas").write_text("basis fixture\n", encoding="utf-8")
    config: dict[str, Any] = {
        "system": {
            "pdb": "system.pdb",
            "forcefields": ["amber14-all.xml"],
            "qm_atoms": [0],
            "qm_charge": 0,
            "qm_multiplicity": 1,
        },
        "orca": {"basis_file": "def2-tzvpd.bas"},
        "conditions": [
            {"name": "test", "temperature_kelvin": 300, "steps": 1, "seed": 17}
        ],
        "output_dir": "runs",
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_cli_dry_run_produces_plan_without_running_dynamics(
    cli_config: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_run(campaign: Campaign) -> None:
        pytest.fail(f"Dry run must not execute {campaign.output_dir}")

    monkeypatch.setattr(cli, "run_campaign", forbidden_run)
    assert cli.main([str(cli_config), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ensemble"] == "NVT"
    assert plan["embedding"] == "elstat"
    assert plan["profile"] == "omol25-derived-mbis"
    assert plan["reference_orca_version"] == "6.0.0"
    assert plan["campaign"]["conditions"][0]["seed"] == 17
    assert "pending" in plan["training_export"]
    assert not (cli_config.parent / "runs").exists()


def test_cli_dispatches_validated_campaign(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[Campaign] = []
    monkeypatch.setattr(cli, "run_campaign", captured.append)
    assert cli.main([str(cli_config)]) == 0
    assert len(captured) == 1
    assert captured[0].system.pdb == cli_config.parent / "system.pdb"


@pytest.mark.parametrize(
    "error",
    [OSError("disk full"), ValueError("invalid atoms"), RuntimeError("ORCA failed")],
)
def test_cli_reports_execution_failures_without_traceback(
    cli_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fail_run(campaign: Campaign) -> None:
        assert campaign.conditions
        raise error

    monkeypatch.setattr(cli, "run_campaign", fail_run)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([str(cli_config)])
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == f"mace-trainer: {error}\n"


def test_cli_rejects_missing_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([str(tmp_path / "absent.json")])
    assert exit_info.value.code == 1
    assert "Cannot read campaign configuration" in capsys.readouterr().err


def test_cli_version_needs_no_configuration(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"{__version__}\n"
