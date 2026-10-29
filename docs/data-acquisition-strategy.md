# Data acquisition for an embedded MACE-SCF QM/MM driver

Literature review and implementation recommendations, 28 October 2026.

## Objective and current scope

The objective is to generate ORCA reference data for MACE-SCF to act as an
electrostatically embedded QM driver in QM/MM simulations. The repository
currently runs ORCA-driven QM/MM dynamics through openmmqmmm and archives the
actual QM evaluations using an OMol-derived electronic-structure profile.

The most useful improvement is to sample both **molecular geometry and
electrostatic environment**, then adapt acquisition to what the model has not
learned. More temperatures and longer trajectories help, but do not reliably
cover polarization responses, reaction barriers or unfamiliar solvent
arrangements. The methods below are recommendations, not implemented features
or a single published protocol.

There are two constraints on the strategy:

- Selecting frames after ORCA-driven MD reduces training redundancy, but those
  ORCA calculations have already been paid for. Reducing electronic-structure
  cost requires better trajectory allocation or inexpensive exploration followed
  by selected calculations at the target ORCA level.
- The inspected local MACE-SCF ORCA importer rejects point-charge embedding. Its
  standard data path accepts a uniform external field, whereas these simulations
  produce spatially varying MM potentials. An extxyz converter alone is
  insufficient: the supported data and model interface must represent that
  embedding. Validate a small training and inference round trip before scaling
  production.

## 1. Sample several MM environments for each selected QM geometry

An embedded model must distinguish a change in molecular geometry from a change
in its surroundings. FieldSchNet provides a direct precedent: its solvent
training data included three classical charge environments per vacuum ethanol
geometry. [Gastegger et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8409491/)

For this repository, implement an environment-resampling workflow:

1. Select representative QM geometries from independent trajectories.
2. Freeze the QM coordinates and resample the surrounding solvent using
   inexpensive MD.
3. Select environments with different local fields, hydrogen bonds and contacts.
4. Evaluate each geometry/environment pair with the target ORCA settings.

For covalently connected QM regions, maintain the boundary and cap construction
so that an apparent environment response is not caused by a changed capped
geometry. Physically sampled solvent environments are the default; arbitrary
random charge perturbations need separate physical justification.

Assign the variants a shared `geometry_family_id`. Preserve the exact effective
point-charge positions and charges supplied to ORCA, including boundary charge
redistribution. Keep an entire geometry family in one dataset split.

## 2. Select configurations using joint structural and electrostatic diversity

Farthest-point sampling and CUR selection are established approaches to choosing
representative configurations for interatomic potentials. They can be used
before a preliminary model exists.
[Imbalzano et al., 2018](https://arxiv.org/html/1804.02150v1)

The proposed QM/MM extension is to construct a fingerprint from:

- QM distances, torsions, coordination numbers and relevant hydrogen bonds.
- MM electrostatic potentials and fields evaluated at QM atoms.
- Selected field gradients where nearby charges create strongly nonuniform
  environments.

Feature scales, electrostatic conventions and rotation handling must be
consistent. A molecule and its rotated copy should not appear different merely
because Cartesian field components were compared directly. Conversely, two
identical QM geometries in different electrostatic environments must remain
distinguishable.

Select within temperature, chemical-state and reaction-coordinate groups, then
apply diversity selection. Retain a representative sample of ordinary
configurations as well as unusual ones; a global farthest-point selector can
overemphasize extreme outliers. Limit selections from adjacent trajectory
segments.

Use selected configurations as restart points for future trajectories. This
directs additional ORCA expenditure toward gaps in coverage, whereas filtering
an already completed trajectory only reduces the downstream dataset size.

## 3. Target reactions and rare configurations explicitly

Equilibrium MD can produce many basin configurations while missing barrier
regions relevant to proton transfer, bond formation or conformational changes.
A directly relevant QM/MM study combines external electrostatic-potential
descriptors, continual learning and replica-exchange umbrella sampling for
chemical and enzymatic reactions.
[Lei, Yagi and Sugita, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11912204/)

Add restrained or umbrella-sampled trajectories along physically meaningful
collective variables. For proton transfer, a starting coordinate is

$$
\xi = r_{\mathrm{donor-H}} - r_{\mathrm{acceptor-H}}.
$$

Run independent replicas and select diverse configurations within each window.
The coordinate and window range must follow the intended chemistry; temperature
alone is not a reliable substitute for reaction-coordinate coverage.

Store the collective variables and sampling bias separately. Training targets
must contain the underlying ORCA energy and forces without artificial sampling
forces. Coverage sufficient for training and convergence of a free-energy
calculation are separate acceptance criteria.

When adding new conditions, retain representative older data and evaluate both
old and new regimes. Relevant charge, protonation and spin states also require
explicit identities and a compatible model; recording spin as metadata does not
by itself make a model spin-conditioned.

## 4. Collect paired references that expose electrostatic response

For selected geometries, calculate embedded and vacuum references using
identical electronic-structure settings, QM coordinates, caps, charge and
multiplicity:

$$
\Delta E_{\mathrm{emb}}(R)
= E_{\mathrm{QM}}(R;\mathrm{MM\ charges}) - E_{\mathrm{QM}}(R;0).
$$

This difference includes electrostatic interaction and polarization; it is not
purely a polarization energy. Separating vacuum contributions from
environment-induced energy and force changes has been demonstrated in
electrostatic-embedding ML models.
[Mazzeo et al., 2024](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00295d)

Add a vacuum calculation for a subset of geometry families. One vacuum reference
can serve several environments. Preserve corresponding changes in QM forces
and multipoles. These paired data support response diagnostics and possible
delta-learning; improved learning is a hypothesis to test, not a guaranteed
benefit of the additional calculation.

Retain molecular dipoles, consistent MBIS multipoles and embedding-site
gradients alongside energies and QM gradients. Raw ORCA gradients require a
minus sign and unit conversion to become forces. Effective point-charge sites
may differ from physical MM atoms, so their mapping must remain explicit.

Sparse positive/negative uniform-field probes can additionally test response
around representative geometries. The MACE-SCF work uses field-dependent data
and atomic multipole supervision, although its reference calculations and
partitioning differ from this ORCA workflow.
[Baldwin et al., 2026 preprint](https://arxiv.org/abs/2603.14700)

Choose probe strengths using the relevant environment-field distribution.
Uniform probes complement spatially varying MM environments. Define units,
signs, origins and multipole conventions explicitly; MBIS charges alone are not
automatically equivalent to a molecular dipole or another density partition.

## 5. Introduce calibrated active learning

Once embedding-aware MACE-SCF training and inference work, train several
independent models, explore using the inexpensive model and request ORCA labels
where predictions disagree. Committee-based selection has demonstrated data
savings for molecular potentials; the amount saved depends on the application.
[Smith et al., 2018](https://arxiv.org/abs/1801.09319)

A proposed acquisition signal is the maximum force disagreement over QM atoms:

$$
u_F(x) = \max_{i\in\mathrm{QM}}
\sqrt{\frac{1}{K}\sum_{k=1}^{K}
\left\|\mathbf F_i^{(k)}(x)-\overline{\mathbf F}_i(x)\right\|^2}.
$$

Here, $K$ is the number of models and $\overline{\mathbf F}_i$ is their
mean force prediction for atom $i$. Taking the maximum avoids averaging
localized disagreement away across the full system. Supplement it with
disagreement in polarization observables and embedding-site forces where
supported. Form an uncertain candidate pool, then
select a diverse batch instead of labelling every adjacent uncertain frame.

Calibrate thresholds against actual ORCA errors on independent data. Use
separate thresholds for requesting labels and stopping unreliable exploration.
Keep a random audit stream, including configurations where the committee agrees:
agreement can still be confidently wrong. Work on uncertainty-biased MD shows
why calibrated uncertainty matters for useful exploration; calibration under
one distribution does not establish reliability in a new environment.
[Zaverkin et al., 2024](https://arxiv.org/abs/2312.01416)

Use exactly the target electronic-structure profile for final reference labels.
Lower-level calculations may support exploration, but their outputs must not be
silently mixed with target-level labels. Preserve SCF failures and acquisition
provenance for diagnosis; failed calculations are not valid training targets.

## Implementation order

| Stage | Proposed addition | Purpose or prerequisite |
| --- | --- | --- |
| 1 | Catalogue archived evaluations; assign trajectory and geometry-family identities; compute joint descriptors and grouped splits | Establish coverage and prevent correlated train/test leakage |
| 2 | Stratified diversity selection and restart recommendations | Improve allocation of subsequent trajectories without requiring a trained model |
| 3 | Fixed-geometry environment resampling and paired vacuum labels | Isolate and measure electrostatic response |
| 4 | Reaction-coordinate sampling with separate bias metadata | Cover chemically important regions missed by equilibrium MD |
| 5 | Committee acquisition, calibrated stopping and random audits | Reduce expensive labels once embedding-aware training and inference are validated |

Develop the nonuniform embedding data/model interface alongside the early
stages. Demonstrate that changing the MM environment at fixed QM geometry
changes model predictions through the intended input path before a large
production campaign. Richer labels cannot compensate for an interface that
cannot distinguish the environments.

## Validation and success criteria

Compare the proposed acquisition strategy with the existing temperature/seed
sampling baseline at **equal ORCA compute cost**, recording core-hours, failed
calculations and successful evaluations. Measure downstream training cost
separately. Use the same model configuration when comparing acquisition methods.

Keep complete trajectories and geometry families together when splitting data.
Do not randomly distribute neighboring frames or paired vacuum/embedded
variants across training and test sets. Reserve unseen environments as explicit
transfer tests and report their errors separately.

Evaluate:

- QM force errors, including large per-atom errors and reaction-region errors.
- Embedding-induced energy differences, multipoles and embedding-site forces.
- Coverage of geometry, local fields and reaction-coordinate windows.
- SCF convergence and the rate of unusable proposed configurations.
- Stability and relevant observables in short QM/MM validation trajectories.
- Actual ORCA errors among nominally low-uncertainty audit configurations.

The initial milestone should be the catalogue, joint diversity selector and
grouped evaluation splits, followed by a small environment-family pilot with
paired vacuum labels. Scale acquisition only after this pilot demonstrates
useful response coverage and a working embedded MACE-SCF training path.
