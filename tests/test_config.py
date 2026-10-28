"""Configuration validation and reproducible path-resolution tests."""

import json
from pathlib import Path
from typing import Any, cast

import pytest

from mace_trainer.config import ConfigError, campaign_to_dict, load_config


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    (tmp_path / "system.pdb").write_text("END\n", encoding="utf-8")
    (tmp_path / "def2-tzvpd.bas").write_text("test basis fixture\n", encoding="utf-8")
    config = {
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
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _modify(path: Path, section: str, key: str, value: object) -> None:
    config = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    if section == "campaign":
        target = config
    elif section == "condition":
        target = config["conditions"][0]
    else:
        target = config[section]
    target[key] = value
    path.write_text(json.dumps(config), encoding="utf-8")


def test_paths_resolve_from_config_and_defaults_are_explicit(
    config_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    campaign = load_config(config_path)

    assert campaign.system.pdb == tmp_path / "system.pdb"
    assert campaign.system.forcefields == ("amber14-all.xml", "amber14/tip3p.xml")
    assert campaign.system.qm_atoms == (0, 1, 2)
    assert campaign.system.platform == "CPU"
    assert campaign.system.periodic is False
    assert campaign.orca.basis_file == tmp_path / "def2-tzvpd.bas"
    assert campaign.orca.orcadir is None
    assert campaign.orca.numcores == 1
    assert campaign.orca.maxcore_mb == 2000
    assert campaign.conditions[0].timestep_fs == 0.5
    assert campaign.conditions[0].trajectory_interval == 10
    assert campaign.output_dir == tmp_path / "runs"
    assert not campaign.output_dir.exists()


def test_manifest_round_trip_preserves_complete_configuration(
    config_path: Path, tmp_path: Path
) -> None:
    orcadir = tmp_path / "orca-install"
    orcadir.mkdir()
    forcefield = tmp_path / "custom.xml"
    forcefield.write_text("<ForceField/>\n", encoding="utf-8")
    _modify(config_path, "system", "forcefields", ["./custom.xml"])
    _modify(config_path, "system", "periodic", True)
    _modify(config_path, "system", "platform", "Reference")
    _modify(config_path, "system", "qm_charge", -1)
    _modify(config_path, "orca", "orcadir", "orca-install")
    _modify(config_path, "orca", "numcores", 8)
    _modify(config_path, "orca", "maxcore_mb", 4000)
    _modify(config_path, "condition", "timestep_fs", 0.25)
    _modify(config_path, "condition", "trajectory_interval", 5)
    campaign = load_config(config_path)
    destination = tmp_path / "different" / "manifest.json"
    destination.parent.mkdir()
    destination.write_text(json.dumps(campaign_to_dict(campaign)), encoding="utf-8")

    assert load_config(destination) == campaign
    assert campaign.system.forcefields == (str(forcefield),)
    assert campaign.orca.orcadir == orcadir


@pytest.mark.parametrize("section", ["campaign", "system", "orca", "condition"])
def test_unknown_fields_are_rejected(config_path: Path, section: str) -> None:
    _modify(config_path, section, "pressure_bar", 1)
    with pytest.raises(ConfigError, match="unknown keys: pressure_bar"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("orca", "numcores"),
        ("orca", "maxcore_mb"),
        ("system", "qm_multiplicity"),
        ("condition", "steps"),
        ("condition", "seed"),
        ("condition", "trajectory_interval"),
    ],
)
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1", None])
def test_positive_integer_fields_reject_invalid_values(
    config_path: Path, section: str, key: str, value: object
) -> None:
    _modify(config_path, section, key, value)
    with pytest.raises(ConfigError, match=key):
        load_config(config_path)


@pytest.mark.parametrize("key", ["temperature_kelvin", "timestep_fs"])
@pytest.mark.parametrize("value", [True, 0, -1, "1", None, 10**400])
def test_positive_real_fields_reject_invalid_values(
    config_path: Path, key: str, value: object
) -> None:
    _modify(config_path, "condition", key, value)
    with pytest.raises(ConfigError, match=key):
        load_config(config_path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonstandard_nonfinite_json_is_rejected(
    config_path: Path, value: float
) -> None:
    _modify(config_path, "condition", "temperature_kelvin", value)
    with pytest.raises(ConfigError, match="nonfinite number"):
        load_config(config_path)


def test_overflowing_json_number_is_rejected(config_path: Path) -> None:
    content = config_path.read_text(encoding="utf-8").replace("300", "1e999")
    config_path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match="finite positive number"):
        load_config(config_path)


@pytest.mark.parametrize("value", [[0, 0], [-1], [True], [0.0], [], "0,1", None])
def test_invalid_qm_indices_are_rejected(config_path: Path, value: object) -> None:
    _modify(config_path, "system", "qm_atoms", value)
    with pytest.raises(ConfigError, match="qm_atoms"):
        load_config(config_path)


@pytest.mark.parametrize("value", [True, 0.5, "0"])
def test_charge_must_be_an_integer(config_path: Path, value: object) -> None:
    _modify(config_path, "system", "qm_charge", value)
    with pytest.raises(ConfigError, match="qm_charge must be an integer"):
        load_config(config_path)


def test_periodic_requires_a_boolean(config_path: Path) -> None:
    _modify(config_path, "system", "periodic", 1)
    with pytest.raises(ConfigError, match="periodic must be a boolean"):
        load_config(config_path)


@pytest.mark.parametrize(
    "value",
    [
        "../run",
        "nested/run",
        ".",
        "with space",
        "",
        " x",
        "inputs",
        "INPUTS",
        "CON",
        "lpt9",
    ],
)
def test_condition_names_cannot_escape_output_directory(
    config_path: Path, value: str
) -> None:
    _modify(config_path, "condition", "name", value)
    with pytest.raises(ConfigError, match=r"condition\.name"):
        load_config(config_path)


def test_seed_is_limited_to_portable_signed_integer_range(config_path: Path) -> None:
    _modify(config_path, "condition", "seed", 2_147_483_648)
    with pytest.raises(ConfigError, match=r"condition\.seed must be at most"):
        load_config(config_path)


@pytest.mark.parametrize("names", [("repeat", "repeat"), ("repeat", "REPEAT")])
def test_duplicate_condition_names_are_rejected(
    config_path: Path, names: tuple[str, str]
) -> None:
    conditions = [
        {"name": name, "temperature_kelvin": 300, "steps": 10, "seed": seed}
        for name, seed in zip(names, [1, 2], strict=True)
    ]
    _modify(config_path, "campaign", "conditions", conditions)
    with pytest.raises(ConfigError, match="duplicate names"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("system", "pdb", "missing.pdb", "system.pdb must refer to an existing file"),
        ("system", "pdb", ".", "system.pdb must refer to an existing file"),
        ("orca", "basis_file", "missing.bas", "orca.basis_file must refer"),
        ("orca", "basis_file", "system.pdb", "must be the OMol file"),
        ("orca", "orcadir", "missing", "orca.orcadir must be an existing directory"),
        ("orca", "orcadir", "system.pdb", "orca.orcadir must be an existing directory"),
        ("system", "forcefields", ["./missing.xml"], "must refer to an existing file"),
        (
            "system",
            "forcefields",
            ["custom/missing.xml"],
            "must refer to an existing file",
        ),
        ("system", "forcefields", ["system.pdb"], "must name an XML file"),
        ("system", "forcefields", ["a.xml", "a.xml"], "duplicate entries"),
        ("system", "forcefields", [], "nonempty JSON array"),
        ("campaign", "output_dir", "system.pdb", "output_dir is not a directory"),
        ("campaign", "output_dir", "\x00", "must not contain null characters"),
        ("campaign", "conditions", [], "nonempty JSON array"),
        ("campaign", "conditions", [1], "condition must be a JSON object"),
        ("campaign", "system", {}, "system is missing required keys"),
        ("campaign", "orca", {}, "orca is missing required keys"),
    ],
)
def test_invalid_input_paths_and_structures_are_rejected(
    config_path: Path, section: str, key: str, value: object, message: str
) -> None:
    _modify(config_path, section, key, value)
    with pytest.raises(ConfigError, match=message):
        load_config(config_path)


@pytest.mark.parametrize("value", ["{", "[]", "{}", '{"system": {}, "system": {}}'])
def test_invalid_json_documents_are_rejected(config_path: Path, value: str) -> None:
    config_path.write_text(value, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_missing_config_is_reported_as_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Cannot read campaign configuration"):
        load_config(tmp_path / "missing.json")


def test_invalid_utf8_config_is_reported_as_config_error(config_path: Path) -> None:
    config_path.write_bytes(b"\xff")
    with pytest.raises(ConfigError, match="Cannot read campaign configuration"):
        load_config(config_path)
