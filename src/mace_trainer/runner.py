"""Run independent NVT conditions through the public openmmqmmm API.

The backend writes into the process working directory. Parallel conditions use
separate spawned processes so their files and backend state remain isolated.
"""

import hashlib
import importlib
import json
import multiprocessing
import platform
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import chdir
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from pathlib import Path
from types import ModuleType
from typing import Any

from .archive import EvaluationArchive
from .config import Campaign, Condition, campaign_to_dict
from .profile import ORCA_REFERENCE_VERSION, PROFILE_NAME, SOURCE_URL, orca_input

_MAX_WINDOWS_WORKERS = 61  # ProcessPoolExecutor's Windows handle limit.


def campaign_plan(campaign: Campaign, *, jobs: int = 1) -> dict[str, object]:
    """Describe the exact method and conditions without importing the MD backend."""
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    workers = min(jobs, len(campaign.conditions))
    if sys.platform == "win32" and workers > _MAX_WINDOWS_WORKERS:
        raise ValueError(
            f"Windows supports at most {_MAX_WINDOWS_WORKERS} parallel jobs"
        )
    simple, blocks = orca_input(campaign.orca.maxcore_mb, campaign.orca.basis_file)
    return {
        "schema_version": 1,
        "campaign": campaign_to_dict(campaign),
        "profile": PROFILE_NAME,
        "profile_source": SOURCE_URL,
        "reference_orca_version": ORCA_REFERENCE_VERSION,
        "orcasimpleinput": simple,
        "orcablocks": blocks,
        "ensemble": "NVT",
        "embedding": "elstat",
        "integrator": "LangevinMiddleIntegrator",
        "parallel_jobs": workers,
        "training_export": "pending MACE-SCF support for point-charge embedding",
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _load_backend() -> tuple[ModuleType, ModuleType]:
    try:
        return importlib.import_module("openmmqmmm"), importlib.import_module("openmm")
    except ImportError as exc:
        raise RuntimeError(
            "Dynamics require openmmqmmm and OpenMM. Install mace-trainer into "
            "your openmmqmmm environment; see README.md."
        ) from exc


def _software() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("mace-trainer", "openmmqmmm", "openmm", "numpy"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "uninstalled source tree"
    return result


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attach_archive(
    qm: Any, qmmm: Any, archive: EvaluationArchive, folder: Path
) -> None:
    """Record actual backend calls while retaining the theory class identities."""
    qm_run, qmmm_run = qm.run, qmmm.run
    frame: dict[str, Any] = {}

    def record_qmmm(**kwargs: Any) -> Any:
        frame.clear()
        frame.update(
            full_system_coords_angstrom=kwargs.get("current_coords"),
            periodic_box_vectors_angstrom=kwargs.get("periodic_box_vectors"),
        )
        return qmmm_run(**kwargs)

    def record_qm(**kwargs: Any) -> Any:
        context = {
            **frame,
            "linkatom_indices": getattr(qmmm, "linkatom_indices", []),
            "linkatom_host_pairs": [
                list(pair) for pair in sorted(getattr(qmmm, "linkatoms_dict", {}))
            ],
            "linkatom_method": qmmm.linkatom_method,
            "linkatom_simple_distance": qmmm.linkatom_simple_distance,
            "chargeboundary_method": qmmm.chargeboundary_method,
            "dipole_correction": qmmm.dipole_correction,
        }
        return archive.evaluate(qm_run, files_dir=folder, context=context, **kwargs)

    qm.run = record_qm
    qmmm.run = record_qmmm


def _run_condition(
    campaign: Campaign, condition: Condition, backend: ModuleType, openmm: ModuleType
) -> None:
    folder = campaign.output_dir / condition.name
    folder.mkdir()
    manifest: dict[str, object] = {
        "status": "running",
        "condition": condition.name,
        "software": _software(),
    }
    _write_json(folder / "run.json", manifest)
    archive = EvaluationArchive(
        folder / "evaluations", tuple(sorted(campaign.system.qm_atoms))
    )
    try:
        with chdir(folder):
            _drive(campaign, condition, backend, openmm, archive)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        manifest["status"] = "completed"
    finally:
        manifest["evaluations"] = archive.count
        _write_json(folder / "run.json", manifest)


def _drive(
    campaign: Campaign,
    condition: Condition,
    backend: ModuleType,
    openmm: ModuleType,
    archive: EvaluationArchive,
) -> None:
    folder = campaign.output_dir / condition.name
    system = campaign.system
    fragment = backend.Fragment(pdbfile=str(system.pdb))
    if max(system.qm_atoms) >= fragment.numatoms:
        raise ValueError("qm_atoms contains an index outside the PDB atom list")
    simple, blocks = orca_input(campaign.orca.maxcore_mb, campaign.orca.basis_file)
    qm = backend.ORCATheory(
        orcadir=campaign.orca.orcadir,
        orcasimpleinput=simple,
        orcablocks=blocks,
        numcores=campaign.orca.numcores,
        filename="orca",
    )
    mm = backend.OpenMMTheory(
        pdbfile=str(system.pdb),
        xmlfiles=list(system.forcefields),
        periodic=system.periodic,
        platform=system.platform,
        autoconstraints=None,
        rigidwater=False,
        hydrogenmass=None,
    )
    qmmm = backend.QMMMTheory(
        qm_theory=qm,
        mm_theory=mm,
        fragment=fragment,
        qmatoms=list(system.qm_atoms),
        qm_charge=system.qm_charge,
        qm_mult=system.qm_multiplicity,
        embedding="elstat",
        truncated_pc=False,
        update_qm_region_charges=False,
    )
    _attach_archive(qm, qmmm, archive, folder)
    (folder / "mm-system.xml").write_text(openmm.XmlSerializer.serialize(mm.system))

    # create_simulation constructs its integrator immediately before the Context.
    # Seed here: setting the seed after Context creation would be too late.
    create_integrator = mm.create_integrator

    def seeded_integrator() -> None:
        create_integrator()
        mm.integrator.setRandomNumberSeed(condition.seed)

    mm.create_integrator = seeded_integrator
    engine = backend.MolecularDynamicsEngine(
        fragment=fragment,
        theory=qmmm,
        timestep=condition.timestep_fs / 1000,
        temperature=condition.temperature_kelvin,
        integrator="LangevinMiddleIntegrator",
        barostat=None,
        traj_frequency=condition.trajectory_interval,
        restartfile_frequency=condition.trajectory_interval,
        trajfilename="trajectory",
        datafilename="state.csv",
    )

    def initialize_velocities(md: Any) -> None:
        md.simulation.context.setVelocitiesToTemperature(
            condition.temperature_kelvin * openmm.unit.kelvin, condition.seed
        )

    try:
        engine.run(
            simulation_steps=condition.steps,
            pre_dynamics_hook=initialize_velocities,
        )
        engine.finalize_simulation()
    finally:
        engine.close()


def _verify_forcefields(forcefield_hashes: dict[str, str]) -> None:
    for name, expected_hash in forcefield_hashes.items():
        if _hash_file(Path(name)) != expected_hash:
            raise ValueError(f"Custom forcefield changed during the campaign: {name}")


def _run_condition_in_process(
    campaign: Campaign, condition: Condition, forcefield_hashes: dict[str, str]
) -> None:
    backend, openmm = _load_backend()
    _verify_forcefields(forcefield_hashes)
    _run_condition(campaign, condition, backend, openmm)


def _run_parallel(
    campaign: Campaign, workers: int, forcefield_hashes: dict[str, str]
) -> None:
    conditions = iter(campaign.conditions)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        pending = {
            executor.submit(
                _run_condition_in_process, campaign, condition, forcefield_hashes
            )
            for condition in islice(conditions, workers)
        }
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Observe every completed result before starting more conditions. On
            # failure, let in-flight runs finish and close their engines normally.
            for future in completed:
                future.result()
            for condition in islice(conditions, len(completed)):
                pending.add(
                    executor.submit(
                        _run_condition_in_process,
                        campaign,
                        condition,
                        forcefield_hashes,
                    )
                )


def run_campaign(campaign: Campaign, *, jobs: int = 1) -> None:
    """Run conditions in up to jobs processes, refusing to overwrite outputs.

    One job runs in the current process. Parallel jobs each load a fresh backend;
    once a failure is reported, active conditions finish but no more are scheduled.
    """
    plan = campaign_plan(campaign, jobs=jobs)
    backend, openmm = _load_backend()
    campaign.output_dir.mkdir(parents=True, exist_ok=False)
    inputs = campaign.output_dir / "inputs"
    inputs.mkdir()
    shutil.copy2(campaign.system.pdb, inputs / "system.pdb")
    shutil.copy2(campaign.orca.basis_file, inputs / "def2-tzvpd.bas")
    execution_campaign = replace(
        campaign,
        system=replace(campaign.system, pdb=inputs / "system.pdb"),
        orca=replace(campaign.orca, basis_file=inputs / "def2-tzvpd.bas"),
    )
    forcefield_hashes = {
        name: _hash_file(Path(name))
        for name in campaign.system.forcefields
        if Path(name).is_absolute()
    }
    plan["input_sha256"] = {
        "pdb": _hash_file(execution_campaign.system.pdb),
        "basis": _hash_file(execution_campaign.orca.basis_file),
        "custom_forcefields": forcefield_hashes,
    }
    plan["execution_campaign"] = campaign_to_dict(execution_campaign)
    plan["orcablocks"] = campaign_plan(execution_campaign)["orcablocks"]
    workers = min(jobs, len(campaign.conditions))
    _write_json(campaign.output_dir / "campaign.json", plan)
    if workers > 1:
        _run_parallel(execution_campaign, workers, forcefield_hashes)
    else:
        for condition in campaign.conditions:
            _verify_forcefields(forcefield_hashes)
            _run_condition(execution_campaign, condition, backend, openmm)
