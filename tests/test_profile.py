"""Protect the scientific profile and the boundary with ORCA input syntax."""

from pathlib import Path

import pytest

from mace_trainer.profile import (
    BASIS_SOURCE_URL,
    ORCA_REFERENCE_VERSION,
    PROFILE_NAME,
    SOURCE_URL,
    orca_input,
)


def test_profile_preserves_reference_settings_and_requests_mbis(tmp_path: Path) -> None:
    simple, blocks = orca_input(2048, tmp_path / "def2-tzvpd.bas")

    assert simple.split() == [
        "!",
        "wB97M-V",
        "def2-TZVPD",
        "RIJCOSX",
        "def2/J",
        "NoUseSym",
        "DIIS",
        "NOSOSCF",
        "NormalConv",
        "DEFGRID3",
        "ALLPOP",
        "NONBO",
        "NONPA",
        "MBIS",
    ]
    assert "%scf Convergence Tight maxiter 300 end" in blocks
    assert "%scf THRESH 1e-12 TCUT 1e-13 end" in blocks
    assert "%elprop Dipole true Quadrupole true end" in blocks
    assert "%method MBIS_LargePrint true end" in blocks
    assert "Print[P_Fockian] 1" in blocks
    assert "Print[P_OrbEn] 2" in blocks
    assert "%maxcore 2048" in blocks
    # The driver adds the derivative task and the system-specific settings.
    assert "engrad" not in simple.lower()
    assert "%pal" not in blocks
    assert "%nbo" not in blocks


def test_profile_resolves_and_quotes_basis_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    _, blocks = orca_input(1024, Path("basis files/def2-tzvpd.bas"))

    expected = tmp_path / "basis files" / "def2-tzvpd.bas"
    assert f'%basis GTOName "{expected}" end' in blocks


@pytest.mark.parametrize("unsafe", ["'", '"', "\n", "\r", "\0"])
def test_profile_rejects_basis_input_injection(tmp_path: Path, unsafe: str) -> None:
    with pytest.raises(ValueError, match="basis_file"):
        orca_input(1024, tmp_path / f"basis{unsafe}file.bas")


def test_profile_rejects_unsafe_resolved_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    basis = tmp_path / "basis.bas"
    monkeypatch.setattr(Path, "resolve", lambda _self: tmp_path / 'unsafe"basis.bas')

    with pytest.raises(ValueError, match="basis_file"):
        orca_input(1024, basis)


@pytest.mark.parametrize("memory", [0, -1, True])
def test_profile_rejects_invalid_memory(memory: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        orca_input(memory, tmp_path / "basis.bas")


def test_profile_identifies_reference_and_intentional_variant() -> None:
    assert PROFILE_NAME == "omol25-derived-mbis"
    assert ORCA_REFERENCE_VERSION == "6.0.0"
    assert SOURCE_URL.endswith("/src/fairchem/data/omol/orca/calc.py")
    assert BASIS_SOURCE_URL.endswith(
        "/src/fairchem/data/omol/orca/basis/def2-tzvpd.bas"
    )
