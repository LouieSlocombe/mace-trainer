# mace-trainer

Generate reference data for **MACE-SCF as an electrostatically embedded QM
driver in QM/MM simulations**. This repository runs ORCA-driven QM/MM molecular
dynamics through [openmmqmmm](https://github.com/LouieSlocombe/openmmqmmm) and
archives the actual QM evaluations under different MD conditions.

The first workflow supports independent NVT runs with configurable temperature,
random seed, timestep and duration. Each condition starts from the supplied PDB
with a fresh MM system, ORCA theory and Langevin integrator. The example is a
small water dimer: one QM water and one MM water. It is a smoke test, not a
production sampling protocol.

## Scientific scope

The `omol25-derived-mbis` profile uses **wB97M-V/def2-TZVPD**, RIJCOSX, def2/J,
DEFGRID3 and the SCF thresholds from the
[official OMol25 input generator](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/data/omol/orca/calc.py).
Supply its [distributed basis file](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/data/omol/orca/basis/def2-tzvpd.bas),
named `def2-tzvpd.bas`. The input is printed by `--dry-run`; the supplied file is
hashed and copied into the archive. Its contents are not checked against a
canonical checksum, so obtaining the correct basis is part of setup.

This is an OMol-derived electronic-structure profile, not a reproduction of the
OMol dataset. Electrostatic embedding changes the Hamiltonian. We add MBIS
atomic multipole output and disable NBO/NPA to avoid requiring an external NBO
installation. OMol25 used ORCA **6.0.0**; the installed executable's actual
version remains in each archived ORCA log. Results from a different ORCA version
must be identified as such. Openmmqmmm supplies the EnGrad directive, charge,
multiplicity, coordinates, point charges and parallel settings.

**Current integration limit:** the MACE-SCF ORCA converter currently rejects
point-charge and QM/MM calculations. These archives preserve the embedded
reference problem; they are not yet MACE-SCF training files. A subsequent adapter
must represent the inhomogeneous MM electrostatic field and define consistent
energy/force targets. Stripping the point charges or treating total QM/MM energy
as isolated QM energy would produce incorrect labels.

## Install

Python 3.12 or newer is supported. For lightweight development and dry runs:

```bash
uv sync
uv run mace-trainer --help
```

For dynamics, install this package into an environment containing a current
`openmmqmmm` and OpenMM with `PythonForce` support (the inspected backend uses
OpenMM 8.6). Follow the
[openmmqmmm installation instructions](https://github.com/LouieSlocombe/openmmqmmm#installation)
first, then, in that activated environment:

```bash
python -m pip install -e /path/to/mace-trainer
```

ORCA is installed separately. Set `OPENMMQMMM_ORCADIR` to its installation
directory, put ORCA on `PATH`, or provide `orca.orcadir` in the configuration.
For multiple ORCA processes, the backend also requires compatible OpenMPI.
`maxcore_mb` is memory **per ORCA process**.

## Run a campaign

Save the official basis file as `examples/def2-tzvpd.bas`. Inspect the configuration
and generated method settings before starting:

```bash
python -m mace_trainer examples/water-dimer.json --dry-run
python -m mace_trainer examples/water-dimer.json
```

The dry run validates configuration and input paths without importing OpenMM or
launching ORCA. OpenMM resolves built-in forcefield XML identifiers when running;
a dry run does not validate forcefield compatibility, QM boundaries or electronic
convergence. Custom XML files can be specified with explicit paths.

All paths are relative to the configuration file. QM indices are zero based in
PDB atom order; charge and multiplicity refer to the QM region. Temperature is
kelvin, timestep is femtoseconds, and duration is a positive step count. Unknown
configuration keys and invalid numbers are rejected. Use a prepared,
parameterized PDB; this workflow does not solvate, minimize or equilibrate it.

Runs use unconstrained bonds, physical hydrogen masses, the full MM point-charge
field and fixed MM charges. ORCA evaluates the forces during integration. Each
condition has a seeded Langevin thermostat and seeded thermal initial velocities;
this does not promise identical floating-point trajectories across platforms.
Periodic systems use openmmqmmm's finite, imaged embedding field, not a periodic
QM Ewald Hamiltonian. Cell-size convergence remains a sampling design question.

Conditions execute sequentially in separate directories. An existing output
directory is rejected, including one from a failed campaign. Choose a new output
directory to rerun; automatic continuation and resume are not implemented.
Dynamics use the archived PDB and basis copies, so later edits to the source
files cannot change subsequent conditions. Custom forcefield XML files and
their includes remain external and must stay unchanged; top-level custom XML
hashes are checked before each condition. The serialized MM system records the
parameters actually constructed for each run.

## Archive contents

```text
runs/water-dimer/
  campaign.json                 # resolved configuration, profile, input hashes
  inputs/                       # PDB and supplied basis snapshot
  water-300K/
    run.json                    # condition status, software versions, call count
    mm-system.xml               # partitioned MM system before adding QM callback
    trajectory.dcd             # full-system coordinates at reporting interval
    state.csv                   # MD state reports
    evaluations/00000001/
      evaluation.json           # actual inputs, embedding, raw labels and status
      orca.inp
      orca.out
      orca.engrad
      orca.pc
      orca.pcgrad
      orca.property.txt         # when produced by ORCA
```

Every **actual ORCA call** receives an evaluation ID; this is not an MD step
number. OpenMM can request extra evaluations for initialization and reporting,
and reuse cached results for repeated geometries. Each record contains the
QM coordinates including caps, effective MM point-charge coordinates and charges,
QM atom mapping, link-atom host pairs, full system coordinates and supplied box
vectors. Raw QM energies are in hartree and QM/point-charge **gradients** in
hartree/bohr. Forces require a minus sign and explicit unit conversion. The QM
energy includes its interaction with the supplied external charges; it is not the
total QM/MM energy reported by the MD engine. Link gradients are stored before
projection onto the physical atoms. Point-charge rows can include redistributed
boundary sites and are not a direct list of physical MM-atom force labels.

Failed calculations retain their current artifacts and failure metadata, then
stop the campaign. Stale text outputs are removed before each call, and finite
labels, gradient shapes and required files are checked before marking success.
The current wavefunction is kept in the working directory for SCF reuse, but
per-evaluation GBW/density archives are not yet supported. This limits future
repartitioning that requires wavefunctions; MBIS output and property files should
be checked against the eventual training target before large campaigns.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run prek run --all-files
uv build
```

Unit tests exercise configuration validation, profile generation, backend
integration and archival failures without requiring ORCA. A local one-step
water-dimer smoke run was also verified with ORCA 6.1.1 and OpenMM 8.6.1: both
QM evaluations completed with energies, QM/point-charge gradients and MBIS
outputs archived. This checks the execution path, not production sampling
quality or MACE-SCF training compatibility. The runtime MD dependencies are
kept separate from the lightweight core environment.

The next milestones are embedded MACE-SCF export, additional sampling controls,
restartable campaigns and trajectory-level train/validation/test splits.

## License

This repository is available under the [MIT License](LICENSE).
