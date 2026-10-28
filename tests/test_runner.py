"""Exercise the MD orchestration contract without running an electronic calculation."""

import hashlib
import json
import sys
from dataclasses import replace
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mace_trainer import runner
from mace_trainer.config import Campaign, Condition, OrcaConfig, SystemConfig


@pytest.fixture
def campaign(tmp_path: Path) -> Campaign:
    pdb = tmp_path / "source.pdb"
    pdb.write_text("four-atom PDB fixture\n", encoding="utf-8")
    basis = tmp_path / "def2-tzvpd.bas"
    basis.write_text("external basis fixture\n", encoding="utf-8")
    forcefield = tmp_path / "custom.xml"
    forcefield.write_text("<ForceField/>\n", encoding="utf-8")
    return Campaign(
        system=SystemConfig(
            pdb=pdb,
            forcefields=(str(forcefield), "amber14/tip3p.xml"),
            qm_atoms=(0, 1),
            qm_charge=-1,
            qm_multiplicity=2,
            periodic=True,
        ),
        orca=OrcaConfig(basis_file=basis, numcores=2, maxcore_mb=1024),
        conditions=(
            Condition("cold", 250.0, 3, 11, 0.25, 2),
            Condition("warm", 350.0, 5, 22, 0.5, 4),
        ),
        output_dir=tmp_path / "runs",
    )


class FakeBackend:
    """Small public-API stand-ins that record initialization and force calls."""

    def __init__(self) -> None:
        self.engines: list[Any] = []
        self.orca: list[Any] = []
        self.events: list[tuple[object, ...]] = []
        self.fail_evaluation = False
        self.fail_finalize = False
        self.module = ModuleType("openmmqmmm")
        self.openmm = ModuleType("openmm")
        harness = self

        class Fragment:
            def __init__(self, *, pdbfile: str) -> None:
                self.pdbfile = pdbfile
                self.pdb_content = Path(pdbfile).read_text(encoding="utf-8")
                self.numatoms = 4
                self.coords = [[float(i), 0.0, 0.0] for i in range(4)]

        class ORCATheory:
            def __init__(self, **kwargs: Any) -> None:
                self.options = kwargs
                harness.orca.append(self)

            def run(self, **kwargs: Any) -> tuple[float, list[Any], list[Any]]:
                harness.events.append(("orca", Path.cwd().name))
                for suffix in ("inp", "out", "engrad", "pc", "pcgrad", "property.txt"):
                    Path(f"orca.{suffix}").write_text(
                        f"current {suffix}\n", encoding="utf-8"
                    )
                if harness.fail_evaluation:
                    raise RuntimeError("ORCA failed to converge")
                return (
                    -12.5,
                    [[0.1, 0.2, 0.3] for _ in kwargs["current_coords"]],
                    [[0.4, 0.5, 0.6] for _ in kwargs["current_mm_coords"]],
                )

        class Integrator:
            def setRandomNumberSeed(self, seed: int) -> None:  # noqa: N802 - backend API
                harness.events.append(("integrator_seed", seed))

        class OpenMMTheory:
            def __init__(self, **kwargs: Any) -> None:
                self.options = kwargs
                self.system = "serialized independent MM system"
                self.integrator: Integrator | None = None

            def create_integrator(self) -> None:
                harness.events.append(("integrator_created",))
                self.integrator = Integrator()

        class QMMMTheory:
            def __init__(self, **kwargs: Any) -> None:
                self.options = kwargs
                self.qm = kwargs["qm_theory"]
                self.mm = kwargs["mm_theory"]
                self.linkatom_indices = [2]
                self.linkatoms_dict = {(1, 2): 2}
                self.linkatom_method = "simple"
                self.linkatom_simple_distance = 1.09
                self.chargeboundary_method = "shift"
                self.dipole_correction = True

            def run(self, **kwargs: Any) -> Any:
                full_coords = kwargs["current_coords"]
                return self.qm.run(
                    current_coords=[*full_coords[:2], [1.09, 0.0, 0.0]],
                    current_mm_coords=full_coords[2:],
                    mmcharges=[-0.4, 0.4],
                    elems=["C", "C", "H"],
                    charge=-1,
                    mult=2,
                    grad=True,
                    pc=True,
                )

        class Context:
            def setVelocitiesToTemperature(  # noqa: N802 - backend API
                self, temperature: float, seed: int
            ) -> None:
                harness.events.append(("velocities", temperature, seed))

        class MolecularDynamicsEngine:
            def __init__(self, **kwargs: Any) -> None:
                self.options = kwargs
                self.theory = kwargs["theory"]
                self.fragment = kwargs["fragment"]
                self.closed = False
                self.finalized = False
                self.steps = 0
                harness.engines.append(self)

            def run(self, *, simulation_steps: int, pre_dynamics_hook: Any) -> None:
                self.steps = simulation_steps
                self.theory.mm.create_integrator()
                harness.events.append(("context_created",))
                self.simulation = SimpleNamespace(context=Context())
                pre_dynamics_hook(self)
                self.theory.run(
                    current_coords=self.fragment.coords,
                    periodic_box_vectors=[
                        [10.0, 0.0, 0.0],
                        [0.0, 10.0, 0.0],
                        [0.0, 0.0, 10.0],
                    ],
                )
                self.fragment.coords[0][0] = 99.0

            def finalize_simulation(self) -> None:
                if harness.fail_finalize:
                    raise OSError("restart write failed")
                self.finalized = True

            def close(self) -> None:
                self.closed = True

        self.module.__dict__.update(
            Fragment=Fragment,
            ORCATheory=ORCATheory,
            OpenMMTheory=OpenMMTheory,
            QMMMTheory=QMMMTheory,
            MolecularDynamicsEngine=MolecularDynamicsEngine,
        )
        self.openmm.__dict__.update(
            XmlSerializer=SimpleNamespace(serialize=str),
            unit=SimpleNamespace(kelvin=1.0),
        )


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    backend = FakeBackend()
    monkeypatch.setitem(sys.modules, "openmmqmmm", backend.module)
    monkeypatch.setitem(sys.modules, "openmm", backend.openmm)

    def package_version(package: str) -> str:
        if package == "mace-trainer":
            raise PackageNotFoundError(package)
        return "test-version"

    monkeypatch.setattr(runner, "version", package_version)
    return backend


def test_campaign_seeds_before_context_and_runs_independent_conditions(
    campaign: Campaign, fake_backend: FakeBackend
) -> None:
    original_directory = Path.cwd()
    runner.run_campaign(campaign)

    assert Path.cwd() == original_directory
    assert len(fake_backend.engines) == 2
    first, second = fake_backend.engines
    assert first.theory is not second.theory
    assert first.theory.mm is not second.theory.mm
    assert first.fragment is not second.fragment
    assert fake_backend.events == [
        ("integrator_created",),
        ("integrator_seed", 11),
        ("context_created",),
        ("velocities", 250.0, 11),
        ("orca", "cold"),
        ("integrator_created",),
        ("integrator_seed", 22),
        ("context_created",),
        ("velocities", 350.0, 22),
        ("orca", "warm"),
    ]
    assert [engine.options["timestep"] for engine in fake_backend.engines] == [
        0.00025,
        0.0005,
    ]
    assert [engine.steps for engine in fake_backend.engines] == [3, 5]
    assert all(engine.closed and engine.finalized for engine in fake_backend.engines)
    for engine, condition in zip(
        fake_backend.engines, campaign.conditions, strict=True
    ):
        assert engine.options["barostat"] is None
        assert engine.options["integrator"] == "LangevinMiddleIntegrator"
        assert engine.options["traj_frequency"] == condition.trajectory_interval
        assert engine.options["restartfile_frequency"] == condition.trajectory_interval
        assert engine.theory.options["embedding"] == "elstat"
        assert engine.theory.options["truncated_pc"] is False
        assert engine.theory.options["update_qm_region_charges"] is False
        assert engine.theory.mm.options["periodic"] is True
        assert engine.theory.mm.options["autoconstraints"] is None
        assert engine.theory.mm.options["rigidwater"] is False
        assert engine.theory.mm.options["hydrogenmass"] is None
        folder = campaign.output_dir / condition.name
        manifest = json.loads((folder / "run.json").read_text())
        assert manifest["status"] == "completed"
        assert manifest["evaluations"] == 1
        assert manifest["software"]["mace-trainer"] == "uninstalled source tree"
        assert manifest["software"]["openmmqmmm"] == "test-version"
        assert (
            folder / "mm-system.xml"
        ).read_text() == "serialized independent MM system"


def test_campaign_preserves_inputs_hashes_embedding_and_link_context(
    campaign: Campaign, fake_backend: FakeBackend
) -> None:
    runner.run_campaign(campaign)

    assert fake_backend.orca[0].options["filename"] == "orca"
    assert fake_backend.orca[0].options["numcores"] == 2
    plan = json.loads((campaign.output_dir / "campaign.json").read_text())
    assert (
        plan["input_sha256"]["pdb"]
        == hashlib.sha256(campaign.system.pdb.read_bytes()).hexdigest()
    )
    forcefield = campaign.system.forcefields[0]
    assert plan["input_sha256"]["custom_forcefields"] == {
        forcefield: hashlib.sha256(Path(forcefield).read_bytes()).hexdigest()
    }
    assert (
        campaign.output_dir / "inputs" / "system.pdb"
    ).read_bytes() == campaign.system.pdb.read_bytes()
    assert (
        campaign.output_dir / "inputs" / "def2-tzvpd.bas"
    ).read_bytes() == campaign.orca.basis_file.read_bytes()
    for condition in campaign.conditions:
        record_path = (
            campaign.output_dir
            / condition.name
            / "evaluations/00000001/evaluation.json"
        )
        record = json.loads(record_path.read_text())
        assert record["status"] == "success"
        assert record["qm_row_to_system_atom"] == [0, 1, None]
        assert record["context"]["full_system_coords_angstrom"][0] == [0.0, 0.0, 0.0]
        assert record["context"]["periodic_box_vectors_angstrom"][0] == [10.0, 0.0, 0.0]
        assert record["context"]["linkatom_indices"] == [2]
        assert record["context"]["linkatom_host_pairs"] == [[1, 2]]
        assert record["context"]["linkatom_method"] == "simple"
        assert record["context"]["linkatom_simple_distance"] == 1.09
        assert record["context"]["chargeboundary_method"] == "shift"
        assert record["context"]["dipole_correction"] is True
        assert record["run_kwargs"]["mmcharges"] == [-0.4, 0.4]
        assert record["run_kwargs"]["current_coords"][-1] == [1.09, 0.0, 0.0]


def test_failed_evaluation_records_failure_closes_engine_and_restores_directory(
    campaign: Campaign, fake_backend: FakeBackend
) -> None:
    fake_backend.fail_evaluation = True
    original_directory = Path.cwd()

    with pytest.raises(RuntimeError, match="ORCA failed to converge"):
        runner.run_campaign(campaign)

    assert Path.cwd() == original_directory
    assert len(fake_backend.engines) == 1
    assert fake_backend.engines[0].closed
    assert not fake_backend.engines[0].finalized
    manifest = json.loads((campaign.output_dir / "cold/run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["evaluations"] == 1
    assert "RuntimeError: ORCA failed to converge" in manifest["error"]
    record = json.loads(
        (campaign.output_dir / "cold/evaluations/00000001/evaluation.json").read_text()
    )
    assert record["status"] == "failed"
    assert "orca.out" in record["artifacts"]
    assert not (campaign.output_dir / "warm").exists()


def test_later_conditions_use_snapshots_when_source_pdb_and_basis_are_edited(
    campaign: Campaign,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_pdb = campaign.system.pdb.read_bytes()
    original_basis = campaign.orca.basis_file.read_bytes()
    run_condition = runner._run_condition  # noqa: SLF001 - intercept between real runs

    def edit_sources_after_condition(
        execution_campaign: Campaign,
        condition: Condition,
        backend: ModuleType,
        openmm: ModuleType,
    ) -> None:
        run_condition(execution_campaign, condition, backend, openmm)
        campaign.system.pdb.write_text("edited source PDB\n", encoding="utf-8")
        campaign.orca.basis_file.write_text("edited source basis\n", encoding="utf-8")

    monkeypatch.setattr(runner, "_run_condition", edit_sources_after_condition)
    runner.run_campaign(campaign)

    pdb_snapshot = campaign.output_dir / "inputs/system.pdb"
    basis_snapshot = campaign.output_dir / "inputs/def2-tzvpd.bas"
    assert pdb_snapshot.read_bytes() == original_pdb
    assert basis_snapshot.read_bytes() == original_basis
    for engine in fake_backend.engines:
        assert engine.fragment.pdbfile == str(pdb_snapshot)
        assert engine.fragment.pdb_content.encode() == original_pdb
    for qm in fake_backend.orca:
        assert f'GTOName "{basis_snapshot}"' in qm.options["orcablocks"]
        assert str(campaign.orca.basis_file) not in qm.options["orcablocks"]
    plan = json.loads((campaign.output_dir / "campaign.json").read_text())
    assert plan["campaign"]["system"]["pdb"] == str(campaign.system.pdb)
    assert plan["execution_campaign"]["system"]["pdb"] == str(pdb_snapshot)
    assert plan["execution_campaign"]["orca"]["basis_file"] == str(basis_snapshot)
    assert plan["input_sha256"]["pdb"] == hashlib.sha256(original_pdb).hexdigest()
    assert plan["input_sha256"]["basis"] == hashlib.sha256(original_basis).hexdigest()
    assert f'GTOName "{basis_snapshot}"' in plan["orcablocks"]


def test_changed_custom_forcefield_stops_before_the_next_condition(
    campaign: Campaign,
    fake_backend: FakeBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_condition = runner._run_condition  # noqa: SLF001 - intercept between real runs

    def edit_forcefield_after_condition(
        execution_campaign: Campaign,
        condition: Condition,
        backend: ModuleType,
        openmm: ModuleType,
    ) -> None:
        run_condition(execution_campaign, condition, backend, openmm)
        Path(campaign.system.forcefields[0]).write_text(
            "changed forcefield\n", encoding="utf-8"
        )

    monkeypatch.setattr(runner, "_run_condition", edit_forcefield_after_condition)
    with pytest.raises(
        ValueError, match="Custom forcefield changed during the campaign"
    ):
        runner.run_campaign(campaign)

    assert len(fake_backend.engines) == 1
    manifest = json.loads((campaign.output_dir / "cold/run.json").read_text())
    assert manifest["status"] == "completed"
    assert not (campaign.output_dir / "warm").exists()


def test_finalization_failure_still_closes_engine(
    campaign: Campaign, fake_backend: FakeBackend
) -> None:
    fake_backend.fail_finalize = True
    with pytest.raises(OSError, match="restart write failed"):
        runner.run_campaign(campaign)
    assert fake_backend.engines[0].closed
    manifest = json.loads((campaign.output_dir / "cold/run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["evaluations"] == 1


def test_invalid_qm_index_stops_before_creating_theories(
    campaign: Campaign, fake_backend: FakeBackend
) -> None:
    invalid = replace(campaign, system=replace(campaign.system, qm_atoms=(0, 4)))
    with pytest.raises(ValueError, match="outside the PDB atom list"):
        runner.run_campaign(invalid)
    assert not fake_backend.orca
    assert not fake_backend.engines
    manifest = json.loads((campaign.output_dir / "cold/run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["evaluations"] == 0


def test_existing_campaign_is_not_overwritten(
    campaign: Campaign, fake_backend: FakeBackend
) -> None:
    campaign.output_dir.mkdir()
    sentinel = campaign.output_dir / "keep.txt"
    sentinel.write_text("previous results", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.run_campaign(campaign)
    assert sentinel.read_text() == "previous results"
    assert list(campaign.output_dir.iterdir()) == [sentinel]
    assert not fake_backend.engines


def test_missing_backend_leaves_no_partial_campaign(
    campaign: Campaign, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(name: str) -> ModuleType:
        raise ImportError(f"missing {name}")

    monkeypatch.setattr("importlib.import_module", unavailable)
    with pytest.raises(RuntimeError, match="Dynamics require openmmqmmm and OpenMM"):
        runner.run_campaign(campaign)
    assert not campaign.output_dir.exists()
