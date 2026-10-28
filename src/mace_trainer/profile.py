"""Build an OMol25-derived ORCA profile for embedded reference calculations.

The electronic settings follow FAIR-Chem's published input generator:
https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/data/omol/orca/calc.py
Its external basis is distributed at:
https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/data/omol/orca/basis/def2-tzvpd.bas

OMol25 used ORCA 6.0.0; this profile does not assert identical results across
versions or equivalence between embedded calculations and the OMol25 dataset.
The intentional population-analysis changes are NONBO/NONPA (no external NBO
installation) and MBIS with atomic dipoles and quadrupoles. The supplied basis
file must be the intended OMol25 basis; its identity is the caller's responsibility.
Openmmqmmm supplies the EnGrad directive, charge, multiplicity, parallelism,
coordinates and embedding. Singlet symmetry breaking is system dependent and
is not selected by this shared profile.
"""

from pathlib import Path

PROFILE_NAME = "omol25-derived-mbis"
SOURCE_URL = (
    "https://github.com/facebookresearch/fairchem/blob/main/"
    "src/fairchem/data/omol/orca/calc.py"
)
BASIS_SOURCE_URL = (
    "https://github.com/facebookresearch/fairchem/blob/main/"
    "src/fairchem/data/omol/orca/basis/def2-tzvpd.bas"
)
ORCA_REFERENCE_VERSION = "6.0.0"


def orca_input(maxcore_mb: int, basis_file: Path) -> tuple[str, str]:
    """Return method keywords and blocks for an openmmqmmm ORCATheory.

    Args:
        maxcore_mb: Positive integer memory limit in MB per ORCA process.
        basis_file: External basis file, resolved before an MD directory change.
            File existence and contents are checked by the caller.

    Raises:
        ValueError: If memory is invalid or the path cannot be safely quoted.
    """
    if type(maxcore_mb) is not int or maxcore_mb <= 0:
        raise ValueError("maxcore_mb must be a positive integer")
    unsafe_characters = ("'", '"', "\n", "\r", "\0")
    if any(character in str(basis_file) for character in unsafe_characters):
        raise ValueError(
            "basis_file must not contain quotes, newlines or NUL characters"
        )
    basis_path = str(basis_file.expanduser().resolve())
    if any(character in basis_path for character in unsafe_characters):
        raise ValueError(
            "basis_file must not contain quotes, newlines or NUL characters"
        )

    simple_input = (
        "! wB97M-V def2-TZVPD RIJCOSX def2/J NoUseSym DIIS NOSOSCF "
        "NormalConv DEFGRID3 ALLPOP NONBO NONPA MBIS"
    )
    blocks = "\n".join(
        (
            "%scf Convergence Tight maxiter 300 end",
            "%elprop Dipole true Quadrupole true end",
            (
                "%output Print[P_ReducedOrbPopMO_L] 1 Print[P_ReducedOrbPopMO_M] 1 "
                "Print[P_BondOrder_L] 1 Print[P_BondOrder_M] 1 "
                "Print[P_Fockian] 1 Print[P_OrbEn] 2 end"
            ),
            f'%basis GTOName "{basis_path}" end',
            "%scf THRESH 1e-12 TCUT 1e-13 end",
            "%method MBIS_LargePrint true end",
            f"%maxcore {maxcore_mb}",
        )
    )
    return simple_input, blocks
