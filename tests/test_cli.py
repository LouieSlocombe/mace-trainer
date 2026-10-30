"""Check reviewable dry runs, execution dispatch, and useful CLI failures."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mace_trainer import __main__ as cli
from mace_trainer import __version__
from mace_trainer.config import Campaign


def test_cli_dry_run_produces_plan_without_running_dynamics(
    config_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_run(campaign: Campaign, *, jobs: int) -> None:
        pytest.fail(f"Dry run must not execute {campaign.output_dir} with {jobs} jobs")

    monkeypatch.setattr(cli, "run_campaign", forbidden_run)
    assert cli.main([str(config_path), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["ensemble"] == "NVT"
    assert plan["parallel_jobs"] == 1
    assert plan["embedding"] == "elstat"
    assert plan["profile"] == "omol25-derived-mbis"
    assert plan["reference_orca_version"] == "6.0.0"
    assert plan["campaign"]["conditions"][0]["seed"] == 42
    assert "pending" in plan["training_export"]
    assert not (config_path.parent / "runs").exists()


def test_cli_dry_run_caps_parallel_jobs_at_condition_count(
    config_data: dict[str, Any],
    write_config: Callable[[dict[str, Any]], Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_data["conditions"].append(
        {
            **config_data["conditions"][0],
            "name": "water-350K",
            "temperature_kelvin": 350,
            "seed": 43,
        }
    )
    config_path = write_config(config_data)

    assert cli.main([str(config_path), "--dry-run", "--jobs", "8"]) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["parallel_jobs"] == 2
    assert not (config_path.parent / "runs").exists()


@pytest.mark.parametrize(("options", "jobs"), [([], 1), (["--jobs", "3"], 3)])
def test_cli_dispatches_validated_campaign(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
    jobs: int,
) -> None:
    captured: list[tuple[Campaign, int]] = []

    def capture_run(campaign: Campaign, *, jobs: int) -> None:
        captured.append((campaign, jobs))

    monkeypatch.setattr(cli, "run_campaign", capture_run)
    assert cli.main([str(config_path), *options]) == 0
    assert len(captured) == 1
    campaign, actual_jobs = captured[0]
    assert campaign.system.pdb == config_path.parent / "system.pdb"
    assert actual_jobs == jobs


@pytest.mark.parametrize("jobs", ["0", "-1", "1.5", "many"])
def test_cli_rejects_invalid_job_counts(
    jobs: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["campaign.json", "--jobs", jobs])
    assert exit_info.value.code == 2
    assert "--jobs" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [OSError("disk full"), ValueError("invalid atoms"), RuntimeError("ORCA failed")],
)
def test_cli_reports_execution_failures_without_traceback(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fail_run(campaign: Campaign, *, jobs: int) -> None:
        assert campaign.conditions
        assert jobs == 1
        raise error

    monkeypatch.setattr(cli, "run_campaign", fail_run)
    with pytest.raises(SystemExit) as exit_info:
        cli.main([str(config_path)])
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
