"""Verify per-call capture at the ORCA protocol boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mace_trainer.archive import EvaluationArchive

_SUFFIXES = (".inp", ".out", ".engrad", ".pc", ".pcgrad", ".property.txt")


def _write_outputs(directory: Path, text: str = "output") -> None:
    for suffix in _SUFFIXES:
        (directory / f"orca{suffix}").write_text(text, encoding="utf-8")


def _record(root: Path, number: int = 1) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(
        (root / f"{number:08d}" / "evaluation.json").read_text(encoding="utf-8")
    )
    return record


def _inputs() -> dict[str, Any]:
    return {
        "current_coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "current_mm_coords": np.array([[3.0, 1.0, 0.0]]),
        "mm_charges": np.array([-0.4]),
        "qm_elems": ["O", "H"],
        "charge": 0,
        "mult": 1,
        "grad": True,
        "pc": True,
    }


def test_records_exact_backend_inputs_caps_and_pointcharge_gradients(
    tmp_path: Path,
) -> None:
    archive = EvaluationArchive(tmp_path / "evaluations", (4,))
    inputs = _inputs()
    coords = inputs["current_coords"]
    gradients = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    pc_gradients = np.array([[0.1, 0.2, 0.3]])
    result = (np.float64(-75.5), gradients, pc_gradients)
    context = {
        "full_system_coords_angstrom": np.zeros((6, 3)),
        "condition": "warm",
        "linkatom_indices": np.array([1]),
        "linkatom_host_pairs": [(4, 5)],
        "seed": np.int64(12),
        "source": tmp_path / "system.pdb",
    }

    def backend(**kwargs: Any) -> Any:
        assert kwargs.keys() == inputs.keys()
        assert kwargs["current_coords"] is coords
        assert kwargs["mm_charges"] is inputs["mm_charges"]
        _write_outputs(tmp_path)
        coords[0, 0] = 99.0
        return result

    returned = archive.evaluate(backend, files_dir=tmp_path, context=context, **inputs)

    assert returned is result
    assert archive.count == 1
    record = _record(archive.root)
    assert record["status"] == "success"
    assert record["qm_atoms"] == [4]
    assert record["qm_row_to_system_atom"] == [4, None]
    assert record["run_kwargs"]["current_coords"] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert record["run_kwargs"]["mm_charges"] == [-0.4]
    assert record["run_kwargs"]["current_mm_coords"] == [[3.0, 1.0, 0.0]]
    assert record["energy_hartree"] == -75.5
    assert record["gradients_hartree_per_bohr"] == gradients.tolist()
    assert record["pointcharge_gradients_hartree_per_bohr"] == pc_gradients.tolist()
    assert record["context"]["linkatom_host_pairs"] == [[4, 5]]
    assert record["context"]["linkatom_indices"] == [1]
    assert record["context"]["full_system_coords_angstrom"] == np.zeros((6, 3)).tolist()
    assert record["context"]["seed"] == 12
    assert record["context"]["source"] == str(tmp_path / "system.pdb")
    assert "orca.property.txt" in record["artifacts"]


def test_each_overwritten_output_is_preserved_and_restart_file_is_retained(
    tmp_path: Path,
) -> None:
    archive = EvaluationArchive(tmp_path / "evaluations", (0, 1))
    (tmp_path / "orca.gbw").write_text("orbital restart", encoding="utf-8")
    (tmp_path / "orca.densities").write_text("large density", encoding="utf-8")

    def backend(**_kwargs: Any) -> Any:
        assert not (tmp_path / "orca.out").exists()
        assert (tmp_path / "orca.gbw").read_text(encoding="utf-8") == "orbital restart"
        _write_outputs(tmp_path, f"call {archive.count}")
        return -75.0, np.zeros((2, 3)), np.zeros((1, 3))

    for _ in range(2):
        archive.evaluate(backend, files_dir=tmp_path, **_inputs())
    assert (archive.root / "00000001" / "orca.out").read_text(
        encoding="utf-8"
    ) == "call 1"
    assert (archive.root / "00000002" / "orca.out").read_text(
        encoding="utf-8"
    ) == "call 2"
    assert not (archive.root / "00000001" / "orca.gbw").exists()
    assert not (archive.root / "00000001" / "orca.densities").exists()
    resumed = EvaluationArchive(archive.root, (0, 1))
    assert resumed.count == 2
    resumed.evaluate(backend, files_dir=tmp_path, **_inputs())
    assert _record(archive.root, 3)["status"] == "success"


def test_failure_archives_new_inputs_and_does_not_reuse_previous_outputs(
    tmp_path: Path,
) -> None:
    archive = EvaluationArchive(tmp_path / "evaluations", (0, 1))
    _write_outputs(tmp_path, "stale previous evaluation")
    failure = RuntimeError("SCF did not converge")

    def backend(**_kwargs: Any) -> Any:
        assert not (tmp_path / "orca.engrad").exists()
        assert not (tmp_path / "orca.pcgrad").exists()
        (tmp_path / "orca.inp").write_text("new input", encoding="utf-8")
        (tmp_path / "orca.out").write_text("failed SCF", encoding="utf-8")
        raise failure

    with pytest.raises(RuntimeError, match="SCF did not converge") as exception:
        archive.evaluate(backend, files_dir=tmp_path, **_inputs())
    assert exception.value is failure
    record = _record(archive.root)
    assert record["status"] == "failed"
    assert record["error"] == {
        "type": "RuntimeError",
        "message": "SCF did not converge",
    }
    assert record["artifacts"] == ["orca.inp", "orca.out"]
    assert record["run_kwargs"]["charge"] == 0
    assert "energy_hartree" not in record
    assert not (archive.root / "00000001" / "orca.engrad").exists()
    assert (archive.root / "00000001" / "orca.inp").read_text(
        encoding="utf-8"
    ) == "new input"


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ((float("nan"), np.zeros((2, 3)), np.zeros((1, 3))), "finite scalar"),
        ((float("inf"), np.zeros((2, 3)), np.zeros((1, 3))), "finite scalar"),
        (
            (-1.0, np.full((2, 3), float("nan")), np.zeros((1, 3))),
            "QM gradient contains",
        ),
        ((-1.0, np.zeros((1, 3)), np.zeros((1, 3))), "QM gradient shape"),
        (
            (-1.0, np.zeros((2, 3)), np.full((1, 3), float("inf"))),
            "Point-charge gradient contains",
        ),
        ((-1.0, np.zeros((2, 3)), np.zeros((2, 3))), "Point-charge gradient shape"),
        ((-1.0, np.zeros((2, 3))), "must return a 3-tuple"),
    ],
)
def test_rejects_invalid_labels_without_marking_success(
    tmp_path: Path, result: Any, message: str
) -> None:
    archive = EvaluationArchive(tmp_path / "evaluations", (0, 1))

    def backend(**_kwargs: Any) -> Any:
        _write_outputs(tmp_path)
        return result

    with pytest.raises(ValueError, match=message):
        archive.evaluate(backend, files_dir=tmp_path, **_inputs())
    record = _record(archive.root)
    assert record["status"] == "failed"
    assert "energy_hartree" not in record
    assert "gradients_hartree_per_bohr" not in record
    assert "orca.engrad" in record["artifacts"]


@pytest.mark.parametrize("missing", [".inp", ".out", ".engrad", ".pc", ".pcgrad"])
def test_missing_required_artifacts_are_failed_evaluations(
    tmp_path: Path, missing: str
) -> None:
    archive = EvaluationArchive(tmp_path / "evaluations", (0, 1))

    def backend(**_kwargs: Any) -> Any:
        _write_outputs(tmp_path)
        (tmp_path / f"orca{missing}").unlink()
        return -75.0, np.zeros((2, 3)), np.zeros((1, 3))

    with pytest.raises(ValueError, match="missing artifacts"):
        archive.evaluate(backend, files_dir=tmp_path, **_inputs())
    record = _record(archive.root)
    assert record["status"] == "failed"
    assert "energy_hartree" not in record
    assert f"orca{missing}" not in record["artifacts"]


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("pc", [False, True])
def test_calls_without_point_charges_preserve_protocol_and_optional_files(
    tmp_path: Path, *, grad: bool, pc: bool
) -> None:
    archive = EvaluationArchive(tmp_path / "evaluations", (0, 1))
    result: Any = (-75.0, np.zeros((2, 3))) if grad else -75.0
    if pc and grad:
        result = (-75.0, np.zeros((2, 3)), np.zeros((0, 3)))

    def backend(**_kwargs: Any) -> Any:
        for suffix in (".inp", ".out", ".engrad", ".pc", ".pcgrad"):
            (tmp_path / f"orca{suffix}").write_text("output", encoding="utf-8")
        return result

    returned = archive.evaluate(
        backend,
        files_dir=tmp_path,
        current_coords=np.zeros((2, 3)),
        current_mm_coords=np.zeros((0, 3)),
        grad=grad,
        pc=pc,
    )
    assert returned is result
    record = _record(archive.root)
    assert record["status"] == "success"
    assert record["energy_hartree"] == -75.0
    if pc and grad:
        assert record["pointcharge_gradients_hartree_per_bohr"] == []
    else:
        assert "pointcharge_gradients_hartree_per_bohr" not in record
    assert ("gradients_hartree_per_bohr" in record) is grad


@pytest.mark.parametrize(
    ("coords", "message"),
    [
        ([[1.0, 2.0]], "must have shape"),
        ([["nan", "0", "0"], ["0", "0", "0"]], "contains nonfinite values"),
        ([[1.0, 2.0, 3.0]], "fewer atoms than the QM atom mapping"),
    ],
)
def test_rejects_invalid_coordinate_mapping(
    tmp_path: Path, coords: Any, message: str
) -> None:
    archive = EvaluationArchive(tmp_path / "evaluations", (0, 1))

    def backend(**_kwargs: Any) -> Any:
        _write_outputs(tmp_path)
        return -75.0

    with pytest.raises(ValueError, match=message):
        archive.evaluate(backend, files_dir=tmp_path, current_coords=coords)
    assert _record(archive.root)["status"] == "failed"


def test_two_archive_instances_never_overwrite_records(tmp_path: Path) -> None:
    first = EvaluationArchive(tmp_path / "evaluations", (0, 1))
    (first.root / "README").write_text("unrelated file", encoding="utf-8")
    second = EvaluationArchive(first.root, (0, 1))

    def backend(**_kwargs: Any) -> Any:
        _write_outputs(tmp_path)
        return -75.0

    for archive in (first, second):
        archive.evaluate(backend, files_dir=tmp_path, current_coords=np.zeros((2, 3)))
    assert _record(first.root, 1)["status"] == "success"
    assert _record(first.root, 2)["status"] == "success"


def test_failed_cleanup_does_not_call_backend_or_archive_stale_files(
    tmp_path: Path,
) -> None:
    archive = EvaluationArchive(tmp_path / "evaluations", (0, 1))
    (tmp_path / "orca.inp").mkdir()
    (tmp_path / "orca.out").write_text("stale output", encoding="utf-8")

    def backend(**_kwargs: Any) -> Any:
        pytest.fail("Backend must not run if stale outputs cannot be cleared")

    with pytest.raises(IsADirectoryError):
        archive.evaluate(backend, files_dir=tmp_path, **_inputs())
    record = _record(archive.root)
    assert record["status"] == "failed"
    assert record["artifacts"] == []
    assert not (archive.root / "00000001" / "orca.out").exists()


def test_archive_io_failure_does_not_hide_backend_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = EvaluationArchive(tmp_path / "evaluations", (0, 1))
    failure = RuntimeError("SCF failed")

    def backend(**_kwargs: Any) -> Any:
        _write_outputs(tmp_path)
        raise failure

    def fail_copy(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("Archive copy denied")

    monkeypatch.setattr("mace_trainer.archive.shutil.copy2", fail_copy)
    with pytest.raises(RuntimeError, match="SCF failed") as exception:
        archive.evaluate(backend, files_dir=tmp_path, **_inputs())
    assert exception.value is failure
    record = _record(archive.root)
    assert record["status"] == "failed"
    assert record["error"]["message"] == "SCF failed"
    assert record["archive_error"] == "Archive copy denied"
