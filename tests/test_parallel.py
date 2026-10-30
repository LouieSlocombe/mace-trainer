"""Check bounded scheduling and isolation of independent campaign conditions."""

import hashlib
import importlib
import json
import os
import threading
from collections.abc import Callable
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
)
from concurrent.futures import wait as wait_futures
from dataclasses import replace
from multiprocessing.context import BaseContext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mace_trainer import runner
from mace_trainer.config import Campaign, Condition, load_config


@pytest.fixture
def parallel_campaign(config_path: Path) -> Campaign:
    campaign = load_config(config_path)
    return replace(
        campaign,
        conditions=tuple(
            replace(campaign.conditions[0], name=name, seed=index + 1)
            for index, name in enumerate(("cold", "warm", "hot", "hotter"))
        ),
    )


@pytest.fixture
def backend_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    modules = ModuleType("openmmqmmm"), ModuleType("openmm")
    monkeypatch.setattr(runner, "_load_backend", lambda: modules)
    return modules


@pytest.mark.parametrize("jobs", [0, -1, True, 1.5, "2"])
def test_invalid_jobs_leave_no_partial_campaign(
    parallel_campaign: Campaign, jobs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_backend() -> None:
        pytest.fail("Invalid jobs must be rejected before importing the backend")

    monkeypatch.setattr(runner, "_load_backend", forbidden_backend)
    with pytest.raises(ValueError, match="jobs must be a positive integer"):
        runner.run_campaign(parallel_campaign, jobs=jobs)
    assert not parallel_campaign.output_dir.exists()


@pytest.mark.parametrize(
    ("condition_count", "jobs", "expected_workers"),
    [(62, 61, 61), (62, 62, None), (1, 100, 1)],
)
def test_windows_limits_effective_workers_before_importing_the_backend(
    parallel_campaign: Campaign,
    monkeypatch: pytest.MonkeyPatch,
    condition_count: int,
    jobs: int,
    expected_workers: int | None,
) -> None:
    campaign = replace(
        parallel_campaign,
        conditions=tuple(
            replace(parallel_campaign.conditions[0], name=f"run-{index}")
            for index in range(condition_count)
        ),
    )

    def forbidden_backend() -> None:
        pytest.fail("Worker limits must be checked before importing the backend")

    monkeypatch.setattr(runner, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(runner, "_load_backend", forbidden_backend)
    if expected_workers is None:
        with pytest.raises(
            ValueError, match="Windows supports at most 61 parallel jobs"
        ):
            runner.run_campaign(campaign, jobs=jobs)
    else:
        assert (
            runner.campaign_plan(campaign, jobs=jobs)["parallel_jobs"]
            == expected_workers
        )
    assert not campaign.output_dir.exists()


@pytest.mark.usefixtures("backend_modules")
def test_parallel_dispatch_caps_jobs_and_uses_archived_inputs(
    parallel_campaign: Campaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    dispatched: list[tuple[Campaign, int, dict[str, str]]] = []

    def capture(campaign: Campaign, workers: int, hashes: dict[str, str]) -> None:
        dispatched.append((campaign, workers, hashes))

    monkeypatch.setattr(runner, "_run_parallel", capture)
    runner.run_campaign(parallel_campaign, jobs=100)

    execution_campaign, workers, hashes = dispatched[0]
    assert workers == len(parallel_campaign.conditions)
    assert hashes == {}
    inputs = parallel_campaign.output_dir / "inputs"
    assert execution_campaign.system.pdb == inputs / "system.pdb"
    assert execution_campaign.orca.basis_file == inputs / "def2-tzvpd.bas"
    manifest = json.loads(
        (parallel_campaign.output_dir / "campaign.json").read_text(encoding="utf-8")
    )
    assert manifest["parallel_jobs"] == workers


@pytest.mark.parametrize("changed", [False, True])
def test_worker_verifies_forcefields_before_running_the_condition(
    parallel_campaign: Campaign,
    backend_modules: tuple[ModuleType, ModuleType],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    changed: bool,
) -> None:
    forcefield = tmp_path / "custom.xml"
    forcefield.write_text("original", encoding="utf-8")
    hashes = {str(forcefield): hashlib.sha256(forcefield.read_bytes()).hexdigest()}
    executed: list[tuple[Campaign, Condition, ModuleType, ModuleType]] = []

    def capture(
        campaign: Campaign,
        condition: Condition,
        backend: ModuleType,
        openmm: ModuleType,
    ) -> None:
        executed.append((campaign, condition, backend, openmm))

    monkeypatch.setattr(runner, "_run_condition", capture)
    condition = parallel_campaign.conditions[0]
    if changed:
        forcefield.write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match="Custom forcefield changed"):
            runner._run_condition_in_process(  # noqa: SLF001 - worker boundary
                parallel_campaign, condition, hashes
            )
        assert executed == []
    else:
        runner._run_condition_in_process(  # noqa: SLF001 - worker boundary
            parallel_campaign, condition, hashes
        )
        assert executed == [(parallel_campaign, condition, *backend_modules)]


@pytest.mark.parametrize("fail", [False, True])
def test_scheduler_bounds_work_and_stops_dispatching_after_failure(
    parallel_campaign: Campaign,
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail: bool,
) -> None:
    barrier = threading.Barrier(2, timeout=5)
    lock = threading.Lock()
    started: list[str] = []
    finished: list[str] = []
    active = 0
    peak_active = 0
    available_slots = 2

    def pool(*, max_workers: int, mp_context: BaseContext) -> ThreadPoolExecutor:
        assert max_workers == 2
        assert mp_context.get_start_method() == "spawn"
        executor = ThreadPoolExecutor(max_workers=max_workers)
        original_submit = executor.submit

        def submit(callback: Callable[..., None], *args: Any) -> Future[None]:
            nonlocal available_slots
            assert available_slots > 0, "Work was queued before a slot became available"
            available_slots -= 1
            return original_submit(callback, *args)

        monkeypatch.setattr(executor, "submit", submit)
        return executor

    def complete(
        futures: set[Future[None]], *, return_when: str
    ) -> tuple[set[Future[None]], set[Future[None]]]:
        nonlocal available_slots
        assert return_when == FIRST_COMPLETED
        # Complete the initial pair together to make the failure check independent
        # of which thread finishes first. Any failed result must prevent refilling.
        done, pending = wait_futures(
            futures, return_when=ALL_COMPLETED if fail else return_when
        )
        available_slots += len(done)
        return done, pending

    def work(
        _campaign: Campaign, condition: Condition, _hashes: dict[str, str]
    ) -> None:
        nonlocal active, peak_active
        with lock:
            started.append(condition.name)
            active += 1
            peak_active = max(peak_active, active)
        try:
            barrier.wait()
            if fail and condition.name == "cold":
                raise RuntimeError("condition failed")
        finally:
            with lock:
                finished.append(condition.name)
                active -= 1

    monkeypatch.setattr(runner, "ProcessPoolExecutor", pool)
    monkeypatch.setattr(runner, "wait", complete)
    monkeypatch.setattr(runner, "_run_condition_in_process", work)
    if fail:
        with pytest.raises(RuntimeError, match="condition failed"):
            runner._run_parallel(parallel_campaign, 2, {})  # noqa: SLF001
        assert sorted(started) == ["cold", "warm"]
    else:
        runner._run_parallel(parallel_campaign, 2, {})  # noqa: SLF001
        assert sorted(started) == ["cold", "hot", "hotter", "warm"]
    assert sorted(finished) == sorted(started)
    assert active == 0
    assert peak_active == 2


def test_spawned_workers_have_independent_processes_and_working_directories(
    parallel_campaign: Campaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    worker = importlib.import_module("parallel_worker")
    monkeypatch.setattr(runner, "_run_condition_in_process", worker.run)
    campaign = replace(parallel_campaign, conditions=parallel_campaign.conditions[:2])
    campaign.output_dir.mkdir()
    original_directory = Path.cwd()

    runner._run_parallel(campaign, 2, {})  # noqa: SLF001 - actual process boundary

    records = [
        json.loads(
            (campaign.output_dir / condition.name / "worker.json").read_text(
                encoding="utf-8"
            )
        )
        for condition in campaign.conditions
    ]
    assert len({record["pid"] for record in records}) == 2
    assert all(record["pid"] != os.getpid() for record in records)
    assert [record["cwd"] for record in records] == [
        str(campaign.output_dir / condition.name) for condition in campaign.conditions
    ]
    assert Path.cwd() == original_directory
