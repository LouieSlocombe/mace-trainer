"""Archive the actual ORCA evaluations requested by QM/MM dynamics."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

_ARTIFACT_SUFFIXES = (".inp", ".out", ".engrad", ".pc", ".pcgrad", ".property.txt")
_COORDINATE_RANK = 2
_CARTESIAN_COMPONENTS = 3


def _json_value(value: Any) -> Any:
    """Snapshot arrays and containers before the backend can mutate them."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, record: dict[str, Any]) -> None:
    """Replace a JSON record atomically, rejecting nonfinite JSON numbers."""
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _coordinate_shape(value: Any, name: str) -> tuple[int, int]:
    coordinates = np.asarray(value, dtype=float)
    # JSON represents an empty (0, 3) point-charge array as [].
    if coordinates.shape == (0,):
        return 0, _CARTESIAN_COMPONENTS
    if (
        coordinates.ndim != _COORDINATE_RANK
        or coordinates.shape[1] != _CARTESIAN_COMPONENTS
    ):
        raise ValueError(f"{name} must have shape (number of atoms, 3)")
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{name} contains nonfinite values")
    return int(coordinates.shape[0]), 3


def _gradient(value: Any, shape: tuple[int, int], name: str) -> list[list[float]]:
    gradient = np.asarray(value, dtype=float)
    if gradient.shape != shape:
        raise ValueError(
            f"{name} shape {gradient.shape} does not match coordinates {shape}"
        )
    if not np.isfinite(gradient).all():
        raise ValueError(f"{name} contains nonfinite values")
    return [[float(component) for component in row] for row in gradient]


def _result_metadata(result: Any, inputs: Mapping[str, Any]) -> dict[str, Any]:
    grad = bool(inputs.get("grad", False))
    pc = bool(inputs.get("pc", False))
    if grad:
        expected = 3 if pc else 2
        if not isinstance(result, tuple) or len(result) != expected:
            raise ValueError(
                f"ORCA gradient calculation must return a {expected}-tuple"
            )
        energy = np.asarray(result[0], dtype=float)
    else:
        energy = np.asarray(result, dtype=float)
    if energy.shape != () or not np.isfinite(energy).all():
        raise ValueError("ORCA energy must be a finite scalar")
    labels: dict[str, Any] = {"energy_hartree": float(energy)}
    if grad:
        shape = _coordinate_shape(inputs.get("current_coords"), "current_coords")
        labels["gradients_hartree_per_bohr"] = _gradient(
            result[1], shape, "QM gradient"
        )
        if pc:
            pc_shape = _coordinate_shape(
                inputs.get("current_mm_coords"), "current_mm_coords"
            )
            labels["pointcharge_gradients_hartree_per_bohr"] = _gradient(
                result[2], pc_shape, "Point-charge gradient"
            )
    return labels


class EvaluationArchive:
    """Persist each backend call, its inputs, gradients, and ORCA text files.

    The backend must use ``filename='orca'`` and run in ``files_dir``. Coordinate
    inputs are in angstrom; returned gradients are in hartree/bohr, not forces.
    Wavefunctions and densities are not archived. The existing ``orca.gbw`` is
    preserved in the working directory so ORCA can reuse its orbital guess.
    The archive is intended for sequential calls within one condition.
    """

    def __init__(self, root: Path, qm_atoms: tuple[int, ...]) -> None:
        """Use root for numbered records and zero-based system QM atom indices."""
        self.root = root
        self.qm_atoms = qm_atoms
        self.root.mkdir(parents=True, exist_ok=True)
        self._count = max(
            (int(path.name) for path in self.root.iterdir() if path.name.isdecimal()),
            default=0,
        )

    @property
    def count(self) -> int:
        """Return the highest allocated evaluation number, including failures."""
        return self._count

    def _allocate(self) -> Path:
        while True:
            self._count += 1
            destination = self.root / f"{self._count:08d}"
            try:
                destination.mkdir()
            except FileExistsError:
                continue
            return destination

    @staticmethod
    def _copy_artifacts(files_dir: Path, destination: Path) -> list[str]:
        copied = []
        for suffix in _ARTIFACT_SUFFIXES:
            source = files_dir / f"orca{suffix}"
            if source.is_file():
                shutil.copy2(source, destination / source.name)
                copied.append(source.name)
        return copied

    @staticmethod
    def _require_artifacts(artifacts: list[str], inputs: Mapping[str, Any]) -> None:
        required = {"orca.inp", "orca.out"}
        if inputs.get("grad", False):
            required.add("orca.engrad")
        if inputs.get("pc", False):
            required.add("orca.pc")
            if inputs.get("grad", False):
                required.add("orca.pcgrad")
        missing = required.difference(artifacts)
        if missing:
            raise ValueError(
                f"ORCA evaluation is missing artifacts: {', '.join(sorted(missing))}"
            )

    def evaluate(
        self,
        run: Callable[..., Any],
        *,
        files_dir: Path,
        context: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Record an ORCA call, return its original result, and rethrow failures.

        ``context`` stores the caller's embedding and MD metadata and is never
        forwarded to ORCA. ``run_kwargs`` snapshots exactly the supplied backend
        arguments, including any link atoms and shifted MM point charges.
        """
        destination = self._allocate()
        record: dict[str, Any] = {
            "schema_version": 1,
            "evaluation": self.count,
            "status": "running",
            "qm_atoms": list(self.qm_atoms),
            "coordinate_unit": "angstrom",
            "run_kwargs": _json_value(kwargs),
            "context": _json_value(context or {}),
            "artifacts": [],
        }
        record_path = destination / "evaluation.json"
        _write_json(record_path, record)
        prepared = False
        try:
            for suffix in _ARTIFACT_SUFFIXES:
                (files_dir / f"orca{suffix}").unlink(missing_ok=True)
            prepared = True
            result = run(**kwargs)
            inputs = record["run_kwargs"]
            labels = _result_metadata(result, inputs)
            qm_count, _ = _coordinate_shape(
                inputs.get("current_coords"), "current_coords"
            )
            if qm_count < len(self.qm_atoms):
                raise ValueError(
                    "ORCA coordinates contain fewer atoms than the QM atom mapping"
                )
            record["qm_row_to_system_atom"] = list(self.qm_atoms) + [None] * (
                qm_count - len(self.qm_atoms)
            )
            record["artifacts"] = self._copy_artifacts(files_dir, destination)
            self._require_artifacts(record["artifacts"], inputs)
            record.update(labels)
            record["status"] = "success"
            _write_json(record_path, record)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            if prepared:
                try:
                    record["artifacts"] = self._copy_artifacts(files_dir, destination)
                except OSError as archive_error:
                    record["archive_error"] = str(archive_error)
            _write_json(record_path, record)
            raise
        return result
