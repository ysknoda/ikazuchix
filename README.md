# ikazuchix

> **Alpha release.** This project is under active development.
> Interfaces, behavior, and output formats may change without notice
> between releases, and it has not yet been broadly tested. Do not
> rely on it for production use.

> **Correctness notice (0.1.0a2).** Releases up to and including
> **0.1.0a1 computed the Kumagai-Oba lattice term incorrectly**: the
> Ewald energy depended on the splitting parameter `eta`, and was
> typically ~10x too large (reciprocal-space term missing a factor 1/2
> and using `G.eps^-1.G` instead of `G.eps.G`, background term with a
> spurious `1/eps`). Formation energies, transition levels and
> equilibrium concentrations from those versions should be recomputed.
> See [CHANGELOG.md](CHANGELOG.md) and
> [docs/adr/0006](docs/adr/0006-ko-ewald-lattice-energy-fix.md).
>
> **0.1.0a3:** a raw far-field `delta_v` also double-counts the defect's own
> point-charge potential. List the sites it was averaged over in
> `alignment_sampling_sites` so ikazuchix subtracts that potential
> ([docs/adr/0009](docs/adr/0009-ko-alignment-point-charge-subtraction.md)).

Defect formation energy calculator for semiconductors, from VASP-style DFT total energies, with the Kumagai-Oba (KO) finite-size charge correction:

> Y. Kumagai and F. Oba, *Phys. Rev. B* **89**, 195205 (2014).

Given a YAML description of a bulk supercell, one or more defect supercells (multiple defect types and charge states in a single run), the dielectric tensor, band structure, and chemical potentials, it computes:

- the Kumagai-Oba Ewald-sum finite-size correction for every charge state,
- the formation energy `E_f(q, E_F)` as a function of Fermi level,
- charge-state transition levels (CTLs), classified as thermodynamically stable (on the envelope) or meta-stable,
- plain-text reports/tables and (optionally) publication-style plots.

## Features

- Multiple defect types and charge states in one run, with a shared Fermi-level sweep.
- Anisotropic dielectric screening: full tensor in both Ewald sums (dielectric metric `R.eps^-1.R` in real space, `G.eps.G` in reciprocal space), validated against eta-invariance, the simple-cubic Madelung constant and an independent reference (`tests/test_ewald_physics.py`).
- Automatic composition completion from `removed_atoms`/`added_atoms`.
- Thorough input validation: malformed or inconsistent YAML (bad charge values, missing nested keys, non-positive-definite dielectric tensors, degenerate lattices, ...) fails fast with a clear, contextualized error instead of a cryptic crash deep inside the calculation.
- Chemical-potential consistency checks (host stability, thermodynamic bounds, missing species), optionally extended with explicit **competing secondary phases** (see [Chemical potential stability region](#chemical-potential-stability-region)).
- Optional, opt-in **numba** JIT acceleration of the Ewald summation (bit-for-bit identical results, just faster) and an **adaptive high-precision mode** that grows the shell cutoff until converged.
- Optional **MPI** (mpi4py) distribution of the work across ranks under `mpirun`, with automatic fallback to a single process when unavailable; a failure on any single rank (bad input, a compute-time error on one work item) is caught and reported clearly on every rank instead of leaving the others hanging forever.
- Shared Ewald shell-sum caching across a defect's charge states, and incremental (not full-recompute) shell growth in adaptive mode — both purely internal speed-ups with zero effect on results (see [Performance & parallelization](#performance--parallelization)).
- Optional self-consistent **Fermi level solving** from charge neutrality, with equilibrium defect and carrier concentrations (see [Equilibrium Fermi level & concentrations](#equilibrium-fermi-level--concentrations)).
- Optional matplotlib plots (per-defect + multi-defect comparison + chemical-potential stability region).

Every accelerator/analysis feature above beyond the core calculation is **fully optional**: the tool runs correctly, and its default output is unaffected, with only `numpy` + `PyYAML` installed (see [Installation](#installation)).

## Installation

```bash
git clone https://github.com/ysknoda/ikazuchix.git
cd ikazuchix
pip install -e .
```

That's the complete, minimal install — both `ikazuchix ...` (console script) and `python -m ikazuchix ...` (no install needed at all, if `numpy`/`PyYAML` are already available) work from here.

Optional extras (any subset; each is independently detected and used automatically when present):

```bash
pip install -e ".[plot]"        # matplotlib, for -p/--plot
pip install -e ".[fast]"        # numba, JIT-accelerated Ewald summation
pip install -e ".[mpi]"         # mpi4py, for mpirun -n N distribution
pip install -e ".[plot,fast,mpi]"   # any combination
pip install -e ".[all]"         # all three at once
```

See [pyproject.toml](pyproject.toml)'s `[project.optional-dependencies]` for what each extra enables. Notably, **mpi4py alone is not enough** for real multi-process execution — see [Performance & parallelization](#performance--parallelization) below.

For running the test suite:

```bash
pip install -e ".[dev]"
pytest
```

## Quick start

```bash
# Minimal run (single-point report at E_F = VBM):
python -m ikazuchix -i input.yaml

# Set reference Fermi level and generate plots:
python -m ikazuchix -i input.yaml -f 1.5 -p

# Extend the sweep below VBM / above CBM, write everything under results/:
python -m ikazuchix -i input.yaml -f -0.3 \
    --fermi-min -0.5 --fermi-max 4.0 -p --out-dir results/

# Dry-run: validate the input file only, no calculation:
python -m ikazuchix -i input.yaml --dry-run

# Verbose Kumagai-Oba correction breakdown:
python -m ikazuchix -i input.yaml -v

# Solve the self-consistent Fermi level and report equilibrium
# defect/carrier concentrations:
python -m ikazuchix -i input.yaml --solve-fermi --temperature 300

# Check chemical potentials against explicit competing phases and
# (ternary hosts) plot the 2D stability region:
python -m ikazuchix -i input.yaml --stability-diagram -p
```

(If installed via `pip install -e .`, the shorter `ikazuchix` console command works identically in place of `python -m ikazuchix` in every example above.)

A ready-to-run example (synthetic, illustrative wurtzite-GaN-like data — **not** a validated DFT dataset) is provided at [examples/example_input.yaml](examples/example_input.yaml):

```bash
python -m ikazuchix -i examples/example_input.yaml -p
```

Run `python -m ikazuchix --help` for the full flag reference (input format table, output file list, and more examples are included in the `--help` epilog as well as below).

## Input file format

All input is a single YAML file. Two layouts are accepted:

- **Multi-defect** (recommended): top-level key `defects`, a list of defect blocks.
- **Legacy single-defect**: top-level `defect` + `charge_states`, automatically wrapped into the multi-defect form.

| Section / Key | Description |
|---|---|
| `bulk.formula` | Chemical formula of the host material |
| `bulk.energy` [eV] | DFT total energy of the perfect supercell |
| `bulk.num_atoms` | Number of atoms in the bulk supercell. Optional; if given, it must equal `sum(composition)` or loading fails with `ValueError` (it is never used in a calculation, only as a check on `composition`; see `docs/adr/0010`) |
| `bulk.composition` | Atom counts `{species: N, ...}` |
| `chemical_potentials.<species>` [eV] | `Δμ_i` = deviation from the elemental reference (e.g. `Ga: 0.000` for Ga-rich) |
| `elemental_references.<species>` [eV/atom] | DFT reference energy of the elemental solid/molecule |
| `band_structure.vbm` / `.cbm` / `.bandgap` / `.fermi_level` [eV] | Absolute VBM/CBM, band gap, default reference Fermi level. Every calculation uses `cbm − vbm`; `bandgap` is only echoed in the `results.txt` header and must equal `cbm − vbm` within 1e-3 eV, otherwise loading fails with `ValueError`. After a scissor correction (shifting `cbm`), update `bandgap` too. |
| `dielectric.tensor` | Full 3x3 static dielectric tensor (diagonal for cubic/hexagonal systems) |
| `supercell.lattice_vectors` [Å] | `a`, `b`, `c` lattice vectors (can be overridden per defect) |
| `supercell.defect_position` | Fractional coordinates of the defect site |
| `defects[].name` / `.defect_type` | Label (e.g. `"V_Ga"`) and `vacancy` / `interstitial` / `substitution` |
| `defects[].removed_atoms` / `.added_atoms` | `{species: count}` — drives automatic composition completion |
| `defects[].charge_states[].charge` | Integer charge `q` (negative = acceptor-like) |
| `defects[].charge_states[].energy` [eV] | DFT total energy of the defect supercell |
| `defects[].charge_states[].num_atoms` | Atom count of the defect supercell. Optional; if given, it must equal `sum(composition)` (explicit or auto-completed) or loading fails with `ValueError` |
| `defects[].charge_states[].composition` | Atom counts (optional — auto-computed from bulk + removed/added atoms if omitted) |
| `defects[].charge_states[].potential_alignment.delta_v` [eV] | `V_defect − V_bulk` averaged over the sampling region, **in VASP's own sign convention** (OUTCAR site potentials / LOCPOT are electron potential energies, `−eφ`). Without `alignment_sampling_sites` it must already have the point-charge potential subtracted; with them, give the raw difference. Irrelevant for `q = 0`. |
| `defects[].alignment_sampling_sites` | Optional list of fractional `[x, y, z]` atomic sites over which the raw `delta_v` was averaged. When given, ikazuchix subtracts the Kumagai–Oba point-charge potential averaged over the same sites: `E_align = q·(delta_v + ⟨φ_PC⟩)`. See `docs/adr/0009`. |

**Getting `delta_v` from VASP (Kumagai–Oba):** take the per-atom site potentials that OUTCAR prints by default ("average (electrostatic) potential at core") for the defect and bulk supercells. Match the atoms, and average `V_defect − V_bulk` over the atoms outside the largest sphere inscribed in the supercell's Wigner–Seitz cell (minimum-image distance from the defect > min |R|/2). Put that raw average in `delta_v` and those atoms' fractional coordinates in `alignment_sampling_sites`. `ikazuchix.ewald.point_charge_potential()` exposes the model potential for other workflows.

All fields above are required. The following are **optional** — every input file that omits them keeps working exactly as before; each is read only by the specific opt-in flag noted:

| Section / Key | Description | Read by |
|---|---|---|
| `band_structure.effective_mass_e` / `.effective_mass_h` | Effective mass, as a multiple of the free-electron mass. Default: `1.0` (free-electron) for either if omitted. | `--solve-fermi` |
| `defects[].charge_states[].degeneracy` | Charge-state degeneracy `g_q` (must be `> 0`). Default: `1`. | `--solve-fermi` |
| `defects[].site_density` [cm⁻³] | Host-site density for this defect. Auto-derived from `removed_atoms` + `bulk.composition` + supercell volume when the defect removes exactly one species (a simple vacancy or single-species substitution); **required** explicitly otherwise (interstitials, multi-species removal). | `--solve-fermi` |
| `competing_phases` (top-level list) | Explicit secondary phases to check chemical potentials against: `[{formula, energy, composition}, ...]`, where `energy` [eV] and `composition` (`{species: count}`) describe the phase the same way `bulk` describes the host. Default: `[]` (elemental-reference-only checks, the pre-existing behaviour). | `check_chemical_potentials` (always-on, silent unless a phase is violated) and `--stability-diagram` |
| `bulk.formation_reference` | `{energy, composition}` of a **k-point-converged** host calculation in any cell with the host's stoichiometry (e.g. the primitive cell). When given, it replaces `bulk.energy` for the host formation enthalpy: the host-stability checks and `--stability-diagram`. `bulk.energy` (which must match the defect supercells' settings) stays the reference for `E_def - E_bulk`. The composition must have the same species and ratios as `bulk.composition`, otherwise loading fails. Default: absent (`bulk.energy` is used for both, the pre-0.1.0a2 behaviour). See `docs/adr/0007`. | `check_chemical_potentials` and `--stability-diagram` |

## Output files

All written under `--out-dir` (default: current directory):

| File | Contents |
|---|---|
| `results.txt` | Full text report (all defects & charges) |
| `ef_vs_ef/<name>_ef_vs_ef.txt` | `E_f(q, E_F)` table + stable envelope |
| `summary/<name>_summary.txt` | KO correction breakdown + `E_f` at the reference Fermi level |
| `ctls_summary.txt` | Unified CTL table (stable + meta-stable, all defects) |
| `plots/<name>_ef.png` (with `-p`) | Per-defect two-panel plot |
| `plots/comparison_plot.png` (with `-p`) | All defect envelopes overlaid |
| `equilibrium_summary.txt` (with `--solve-fermi`) | Solved self-consistent Fermi level, carrier concentrations, per-(defect, charge) equilibrium concentrations |
| `stability_summary.txt` (with `--stability-diagram`) | Chemical-potential feasibility check + (ternary hosts) stability-region polygon vertices |
| `plots/stability_region.png` (with `--stability-diagram -p`, ternary hosts only) | 2D chemical-potential stability region plot |

## Performance & parallelization

Both accelerators below are optional and change **only speed**, never results — verified by the golden-output regression tests in `tests/test_golden_regression.py`, which compare against a reference captured from a plain `numpy`+`PyYAML`-only run.

### numba (JIT)

```bash
pip install numba
python -m ikazuchix -i input.yaml            # numba used automatically if installed
python -m ikazuchix -i input.yaml --no-numba # force the pure-Python path
```

See `benchmarks/bench_numba.py` for a speed comparison on this machine (real observed speedup: two orders of magnitude on the Ewald shell sums at larger cutoffs).

The default fixed shell cutoff (`--real-cutoff 3 --recip-cutoff 3`) together with the default splitting parameter `eta = sqrt(pi * eps_avg) / V^(1/3)` converges both Ewald sums to well below `1e-6` eV for typical 30-100 atom supercells. For unusual cells, or an explicit `--ewald-eta`, opt into adaptive convergence:

```bash
python -m ikazuchix -i input.yaml --adaptive-ewald
```

This grows the cutoff (from `--real-cutoff`/`--recip-cutoff`) until the outermost-shell contribution drops below `--ewald-tol` (default: `1e-6` eV) or `--max-ewald-shells` (default: 15) is reached. It is **opt-in**: default output is unaffected unless this flag is given.

### MPI

`mpirun -n N` distributes the (defect, charge_state) work list round-robin across N ranks; only rank 0 writes any output, so results are identical to a single-process run.

```bash
mpirun -n 4 python -m ikazuchix -i input.yaml
```

mpi4py alone is **not sufficient** for multi-rank execution — it also needs a system MPI runtime (OpenMPI or MPICH) providing `libmpi.dylib`/`.so`, which is not a pip-installable Python package. The simplest way to get both:

```bash
conda install -c conda-forge openmpi mpi4py
```

If `mpi4py` is installed but no MPI runtime is available, it raises `RuntimeError` (not `ImportError`) at `from mpi4py import MPI` — this is detected automatically and the tool falls back to a single process, exactly as if `mpi4py` were absent. `--no-mpi` forces this fallback even when real MPI is available (combining `--no-mpi` with `mpirun -n N>1` makes every rank redundantly compute and write the full output — a warning is printed in that case). If `mpirun` launches the process but `mpi4py` itself is unusable in that Python environment, a similar warning is printed (detected via the `OMPI_COMM_WORLD_SIZE`/`PMI_SIZE` environment variables OpenMPI/MPICH set on every launched process). See `benchmarks/bench_mpi.py` for a scaling comparison; note that for small problems (few defects/charge states), fixed per-process Python/numpy startup overhead can dominate wall-clock time more than the actual distributed computation does — MPI pays off most for larger multi-defect screening studies, especially across multiple cluster nodes.

**Known limitation**: under `-v`/`--verbose` with `mpirun -n N > 1`, the detailed per-term Kumagai-Oba breakdown (printed directly inside the correction calculator) is not captured/reordered across ranks like the rest of the console output is, so it may interleave non-deterministically. All file outputs are unaffected.

A rank-0-only failure (a bad input file, or a compute-time error isolated to one work item) is caught, reported with full context (which defect/charge state, which rank, the original error), and re-raised **consistently on every rank** — rather than leaving the other ranks blocked forever waiting at the next collective call. See `ikazuchix/mpi_utils.py`'s `raise_consistently()`.

### Internal caching (always on, zero effect on results)

Two purely-internal speed-ups, both verified bit-for-bit (fixed cutoff) or to relative `1e-9` (`--adaptive-ewald`, where floating-point summation order isn't preserved — see below) against the uncached path in `tests/test_ewald.py`:

- **Shared shell-sum cache**: the raw (unscaled) Ewald shell sums depend only on a defect's lattice/dielectric tensor/`eta`, never on the charge `q` — so `runner.py` now computes them once per defect (`EwaldShellCache`) instead of once per charge state. The *decision* of whether a given cutoff has converged (and whether to warn) still depends on `q`, and is still made independently for every charge state.
- **Incremental adaptive growth**: `--adaptive-ewald` used to re-walk the entire cutoff cube from scratch every time it grew the shell count by one. `EwaldShellCache` now grows the cached sum incrementally, via new O(n²) "outer shell only" kernels (`ewald_kernels.py`), instead of re-visiting already-converged inner shells every step.

See `benchmarks/bench_perf1_shell_cache.py` and `benchmarks/bench_perf2_adaptive_incremental.py` for scaling comparisons on this machine.

## Equilibrium Fermi level & concentrations

```bash
python -m ikazuchix -i input.yaml --solve-fermi --temperature 300
```

Opt-in (`--solve-fermi`): solves the self-consistent Fermi level from the charge-neutrality condition

```
sum_defects sum_q  q * C_{D,q}(E_F)  +  p(E_F) - n(E_F)  =  0
```

via plain bisection over `[--fermi-min, --fermi-max]` (default: the band gap) — the residual is provably strictly monotonic in `E_F`, so bisection alone is guaranteed to find the unique root; no `scipy`/Brent's-method dependency is needed. Defect concentrations use the dilute-limit mass-action law `C_{D,q} = N_sites · g_q · exp(-E_f(q,E_F)/kT)`; carrier concentrations use the standard parabolic-band effective density of states, derived from CODATA/SI physical constants rather than a hardcoded reference value. Writes `equilibrium_summary.txt`.

Flags: `--temperature K` (default `300.0`), `--no-carriers` (solve for defect-defect charge compensation only, excluding the electron/hole terms), `--fermi-solve-tol eV` (bisection tolerance, default `1e-8`).

**Known limitations** (dilute-limit, non-degenerate approximations — standard in the defect-formation-energy literature, but not appropriate for heavily-doped/degenerate systems): if `effective_mass_e`/`effective_mass_h` are not given, the free-electron default (`1.0 m₀`) is used and a console warning is printed — real-material carrier concentrations can be off by an order of magnitude with the default. A `site_density` is required (explicit or auto-derived — see the input-format table above) for every defect with at least one non-zero charge state.

## Chemical potential stability region

```bash
python -m ikazuchix -i input.yaml --stability-diagram -p
```

Opt-in (`--stability-diagram`): checks the input file's `chemical_potentials` against every linear thermodynamic constraint — the host stability equation, each species' elemental-reference bound (`Δμᵢ ≤ 0`), and (if given) every explicit `competing_phases` entry's bound `Σᵢ nᵢ·Δμᵢ ≤ ΔH_f(phase)` — and, for a **ternary** host (exactly 3 species), computes the full 2D stability-region polygon via Sutherland-Hodgman half-plane clipping (numpy only, no extra dependency). The two plot axes are the two lightest species by atomic number (e.g. `O` and `Mg` for MgAl₂O₄); the heaviest is eliminated through the host equation. Writes `stability_summary.txt`, and (with `-p`, ternary hosts only) `plots/stability_region.png` via `--stability-plot`.

For a binary (2-species) host, the existing chemical-potential-bound check (part of every run, not just `--stability-diagram`) already incorporates any `competing_phases` you provide. For 4+ species, only a direct feasibility check is performed (a specific `Δμ` point in or out) — the full region lives in too many dimensions to usefully enumerate as a polygon.

## Project structure

```
pyproject.toml                # project metadata, dependencies, `ikazuchix` console script
ikazuchix/                     # implementation package; run as `python -m ikazuchix ...`
    __main__.py                 # CLI entry point (enables `python -m ikazuchix`)
    constants.py                 # physical constants
    io_utils.py                   # YAML loading, text report writers
    geometry.py                    # lattice / dielectric-tensor helpers
    numba_shim.py                   # optional numba JIT fallback
    ewald_kernels.py                 # JIT-able Ewald shell-sum kernels
    ewald.py                          # Kumagai-Oba correction calculator
    chem_potentials.py                 # chemical-potential checks/corrections
    formation_energy.py                 # E_f formula + CTL analysis
    equilibrium.py                       # self-consistent Fermi level + concentrations (opt-in)
    stability_region.py                   # chemical-potential stability region (opt-in)
    mpi_utils.py                           # optional MPI fallback + warning replay
    runner.py                               # per-defect / all-defects orchestration
    report.py                                # plain-text report formatting
    plotting.py                               # optional matplotlib plots
    cli.py                                     # argparse definitions + help text
    main.py                                     # top-level orchestration
tests/                        # pytest suite (see below)
benchmarks/                   # standalone timing scripts (not part of pytest)
examples/example_input.yaml   # synthetic example input
examples/AlN_vasp_smoke.yaml  # real small-cell VASP data, binary host (unconverged; smoke test)
examples/MgAl2O4_vasp_smoke.yaml  # real small-cell VASP data, ternary host with stability polygon
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

(`pyproject.toml` sets `testpaths = ["tests"]`, so plain `pytest` from the repo root is enough.) The suite runs fully (with some tests appropriately skipped) using only the core + dev dependencies; also install the `plot`/`fast`/`mpi` extras (or just `.[all,dev]`) for full coverage (numba/matplotlib/MPI-specific tests). `tests/test_mpi_consistency.py` additionally requires a real MPI runtime (`mpirun` on `PATH`, with `mpi4py` usable in the same Python environment) and is skipped otherwise.

## Method / citation

If you use this tool, please cite:

1. Y. Kumagai and F. Oba, *Phys. Rev. B* **89**, 195205 (2014) — the Kumagai-Oba finite-size correction implemented here.
2. C. Freysoldt et al., *Rev. Mod. Phys.* **86**, 253 (2014) — a comprehensive review of charged-defect formation-energy methods.
3. J. L. Lyons and C. G. Van de Walle, *npj Comput. Mater.* **3**, 12 (2017) — an example application to GaN point defects.

## License

[MIT](LICENSE)
