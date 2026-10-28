"""Strict, dependency-free configuration for ORCA-driven QM/MM campaigns."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

_MAX_SEED = 2_147_483_647
_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_FORCEFIELD_PACKAGE_PATTERN = re.compile(
    r"(?:amber\d+|charmm\d+(?:_\d+)?|implicit)/[^/]+\.xml"
)
_RESERVED_CONDITION_NAMES = {"inputs", "con", "prn", "aux", "nul"} | {
    f"{prefix}{index}" for prefix in ("com", "lpt") for index in range(1, 10)
}


class ConfigError(ValueError):
    """Raised when a campaign configuration cannot be safely interpreted."""


@dataclass(frozen=True, slots=True)
class SystemConfig:
    """Molecular input and zero-based QM atom indices in PDB atom order."""

    pdb: Path
    forcefields: tuple[str, ...]
    qm_atoms: tuple[int, ...]
    qm_charge: int
    qm_multiplicity: int
    periodic: bool = False
    platform: str = "CPU"


@dataclass(frozen=True, slots=True)
class OrcaConfig:
    """ORCA resources and the required OMol distributed basis file."""

    basis_file: Path
    orcadir: Path | None = None
    numcores: int = 1
    maxcore_mb: int = 2000


@dataclass(frozen=True, slots=True)
class Condition:
    """An independently seeded fixed-volume MD run with explicit units."""

    name: str
    temperature_kelvin: float
    steps: int
    seed: int
    timestep_fs: float = 0.5
    trajectory_interval: int = 10


@dataclass(frozen=True, slots=True)
class Campaign:
    """A shared system and ORCA settings evaluated at multiple conditions."""

    system: SystemConfig
    orca: OrcaConfig
    conditions: tuple[Condition, ...]
    output_dir: Path


def _object(
    value: object, label: str, required: set[str], optional: set[str]
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a JSON object")
    result = cast("dict[str, object]", value)
    unknown = result.keys() - required - optional
    if unknown:
        raise ConfigError(f"{label} has unknown keys: {', '.join(sorted(unknown))}")
    missing = required - result.keys()
    if missing:
        raise ConfigError(
            f"{label} is missing required keys: {', '.join(sorted(missing))}"
        )
    return result


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigError(
            f"{label} must be a nonempty string without surrounding spaces"
        )
    if "\x00" in value:
        raise ConfigError(f"{label} must not contain null characters")
    return value


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{label} must be at least {minimum}")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{label} must be a finite positive number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ConfigError(f"{label} must be a finite positive number") from error
    if not math.isfinite(result) or result <= 0:
        raise ConfigError(f"{label} must be a finite positive number")
    return result


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{label} must be a nonempty JSON array")
    return cast("list[object]", value)


def _path(value: object, label: str, base: Path) -> Path:
    path = Path(_string(value, label)).expanduser()
    return (base / path).resolve()


def _file(value: object, label: str, base: Path) -> Path:
    path = _path(value, label, base)
    if not path.is_file():
        raise ConfigError(f"{label} must refer to an existing file: {path}")
    return path


def _forcefield(value: object, base: Path) -> str:
    label = "system.forcefields entry"
    name = _string(value, label)
    path = Path(name).expanduser()
    if path.suffix != ".xml":
        raise ConfigError(f"{label} must name an XML file: {name}")
    resolved = (base / path).resolve()
    if resolved.is_file():
        return str(resolved)
    # Bare filenames and OpenMM package identifiers are checked by OpenMM at run
    # time. Explicit filesystem references must already exist during validation.
    package_identifier = _FORCEFIELD_PACKAGE_PATTERN.fullmatch(name) is not None
    if not name.startswith((".", "~", "/")) and (
        len(path.parts) == 1 or package_identifier
    ):
        return name
    raise ConfigError(f"{label} must refer to an existing file: {resolved}")


def _system(value: object, base: Path) -> SystemConfig:
    data = _object(
        value,
        "system",
        {"pdb", "forcefields", "qm_atoms", "qm_charge", "qm_multiplicity"},
        {"periodic", "platform"},
    )
    atom_indices = tuple(
        _integer(atom, "system.qm_atoms entry", minimum=0)
        for atom in _array(data["qm_atoms"], "system.qm_atoms")
    )
    if len(set(atom_indices)) != len(atom_indices):
        raise ConfigError("system.qm_atoms must not contain duplicate indices")
    forcefields = tuple(
        _forcefield(entry, base)
        for entry in _array(data["forcefields"], "system.forcefields")
    )
    if len(set(forcefields)) != len(forcefields):
        raise ConfigError("system.forcefields must not contain duplicate entries")
    periodic = data.get("periodic", False)
    if not isinstance(periodic, bool):
        raise ConfigError("system.periodic must be a boolean")
    return SystemConfig(
        pdb=_file(data["pdb"], "system.pdb", base),
        forcefields=forcefields,
        qm_atoms=atom_indices,
        qm_charge=_integer(data["qm_charge"], "system.qm_charge"),
        qm_multiplicity=_integer(
            data["qm_multiplicity"], "system.qm_multiplicity", minimum=1
        ),
        periodic=periodic,
        platform=_string(data.get("platform", "CPU"), "system.platform"),
    )


def _orca(value: object, base: Path) -> OrcaConfig:
    data = _object(value, "orca", {"basis_file"}, {"orcadir", "numcores", "maxcore_mb"})
    basis_file = _file(data["basis_file"], "orca.basis_file", base)
    if basis_file.name != "def2-tzvpd.bas":
        raise ConfigError("orca.basis_file must be the OMol file named def2-tzvpd.bas")
    orcadir = None
    if data.get("orcadir") is not None:
        orcadir = _path(data["orcadir"], "orca.orcadir", base)
        if not orcadir.is_dir():
            raise ConfigError(f"orca.orcadir must be an existing directory: {orcadir}")
    return OrcaConfig(
        basis_file=basis_file,
        orcadir=orcadir,
        numcores=_integer(data.get("numcores", 1), "orca.numcores", minimum=1),
        maxcore_mb=_integer(data.get("maxcore_mb", 2000), "orca.maxcore_mb", minimum=1),
    )


def _condition(value: object) -> Condition:
    data = _object(
        value,
        "condition",
        {"name", "temperature_kelvin", "steps", "seed"},
        {"timestep_fs", "trajectory_interval"},
    )
    name = _string(data["name"], "condition.name")
    if _NAME_PATTERN.fullmatch(name) is None:
        raise ConfigError(
            "condition.name must begin with a letter or digit and contain only "
            "letters, digits, hyphens, and underscores"
        )
    if name.casefold() in _RESERVED_CONDITION_NAMES:
        raise ConfigError(f"condition.name is reserved: {name}")
    seed = _integer(data["seed"], "condition.seed", minimum=1)
    if seed > _MAX_SEED:
        raise ConfigError(f"condition.seed must be at most {_MAX_SEED}")
    return Condition(
        name=name,
        temperature_kelvin=_positive_number(
            data["temperature_kelvin"], "condition.temperature_kelvin"
        ),
        steps=_integer(data["steps"], "condition.steps", minimum=1),
        seed=seed,
        timestep_fs=_positive_number(
            data.get("timestep_fs", 0.5), "condition.timestep_fs"
        ),
        trajectory_interval=_integer(
            data.get("trajectory_interval", 10),
            "condition.trajectory_interval",
            minimum=1,
        ),
    )


def _reject_constant(value: str) -> object:
    raise ConfigError(f"JSON contains a nonfinite number: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"JSON contains a duplicate key: {key}")
        result[key] = value
    return result


def load_config(path: Path) -> Campaign:
    """Load strict JSON, resolving filesystem paths relative to its location.

    OpenMM built-in forcefield identifiers are preserved for backend validation.
    Use an explicit ``./`` prefix to require a local forcefield file. The basis
    filename is checked here; provenance and basis contents are backend concerns.

    Raises:
        ConfigError: If the JSON, fields, numerical values, or input paths are invalid.
    """
    source = path.expanduser().resolve()
    try:
        value: object = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError(
            f"Cannot read campaign configuration {source}: {error}"
        ) from error
    data = _object(
        value, "campaign", {"system", "orca", "conditions", "output_dir"}, set()
    )
    conditions = tuple(
        _condition(entry) for entry in _array(data["conditions"], "conditions")
    )
    if len({condition.name.casefold() for condition in conditions}) != len(conditions):
        raise ConfigError("conditions must not contain duplicate names ignoring case")
    output_dir = _path(data["output_dir"], "output_dir", source.parent)
    if output_dir.exists() and not output_dir.is_dir():
        raise ConfigError(f"output_dir is not a directory: {output_dir}")
    return Campaign(
        system=_system(data["system"], source.parent),
        orca=_orca(data["orca"], source.parent),
        conditions=conditions,
        output_dir=output_dir,
    )


def campaign_to_dict(campaign: Campaign) -> dict[str, object]:
    """Return a JSON-compatible configuration for manifests and execution plans."""
    return {
        "system": {
            **asdict(campaign.system),
            "pdb": str(campaign.system.pdb),
            "forcefields": list(campaign.system.forcefields),
            "qm_atoms": list(campaign.system.qm_atoms),
        },
        "orca": {
            **asdict(campaign.orca),
            "basis_file": str(campaign.orca.basis_file),
            "orcadir": None
            if campaign.orca.orcadir is None
            else str(campaign.orca.orcadir),
        },
        "conditions": [asdict(condition) for condition in campaign.conditions],
        "output_dir": str(campaign.output_dir),
    }
