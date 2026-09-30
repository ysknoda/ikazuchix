# Changelog

## 0.1.0a3 — 2026-10-01

### Fixed

- **KO alignment term (docs/adr/0009).** The raw far-field `delta_v` still contains the defect's own point-charge potential, which `E_lat` already accounts for, so it was counted twice. The new optional per-defect `alignment_sampling_sites` list makes ikazuchix subtract the KO point-charge potential averaged over those sites: `E_align = q·(delta_v + <φ_PC>)`. On real data the effect is up to about 2 eV (32-atom AlN V_Al with q = −3). The sign convention is fixed as VASP's own (electron potential energy); it was determined from real data, where the subtracted residual reproduces the neutral state's offset. Inputs without the field behave exactly as before.
- With `--stability-diagram --solve-fermi`, the stability check now runs and writes `stability_summary.txt` (and its plot) **before** the Fermi-level solve. At an infeasible chemical-potential point the solver fails, since formation energies are strongly negative and no root can be bracketed; the summary explaining that failure is now already on disk.

### Added

- `ikazuchix.ewald.point_charge_potential()`: the anisotropic, background-neutralized Ewald potential of a periodic point charge. It is validated against eta invariance, an independent reference, and the exact `r -> 0` relation to `E_PC`.
- Optional `bulk.formation_reference: {energy, composition}` holds a k-converged host energy for the host formation enthalpy: the host-stability checks and `--stability-diagram`. The defect-supercell `bulk.energy` stays the reference for `E_def - E_bulk`. A stoichiometry mismatch with `bulk.composition` is an input error. See `docs/adr/0007`.
- `examples/AlN_vasp_smoke.yaml` and `examples/MgAl2O4_vasp_smoke.yaml` hold numbers extracted from real, small-cell, unconverged VASP 6.6.1 PBE runs, for smoke tests only. `tests/test_vasp_smoke_examples.py` pins the ternary stability polygon against an independent vertex enumeration and checks the KO energy of the simple-cubic spinel cell against the Madelung constant.

### Changed

- The stability-region axes are ordered by atomic number, not alphabetically. For MgAl2O4 the x and y axes are `O` and `Mg`, and `Al` is eliminated. Labels that are not element symbols sort last.
- CI: the `core` job runs on Python 3.10–3.13, and `full`/`mpi` run on 3.13.

## 0.1.0a2 — 2026-09-30

### Fixed (breaking: results change)

- **Kumagai-Oba point-charge lattice energy.** Up to 0.1.0a1, the Ewald sum behind the lattice term had three errors:
  - The reciprocal-space term was missing a factor 1/2.
  - The reciprocal-space term used `G.eps^-1.G` (with a `1/sqrt(det eps)` prefactor) instead of the screened `G.eps.G`.
  - The real-space and background terms used a scalar `eps_avg` convention inconsistent with the reciprocal-space term.

  As a result the lattice term depended on the Ewald splitting parameter `eta` and was typically about 10x too large. For example, on a 32-atom AlN cell it came out as +3.13 eV where the correct value is +0.312 eV. The sum now uses one consistent anisotropic form. It is validated against:
  - eta invariance,
  - the simple-cubic Madelung constant (alpha = 2.837297),
  - an independent brute-force reference with a full, non-diagonal dielectric tensor.

  See `docs/adr/0006`.
- **Default `eta`** is now `sqrt(pi * eps_avg) / V^(1/3)` (was `sqrt(pi) / V^(1/3)`). With it, the default 3-shell cutoff converges without warnings. An explicit `--ewald-eta` is used verbatim.

### Changed

- Output: the KO breakdown now reports the point-charge energy `E_PC` (negative) and the lattice correction `E_lat = -E_PC` (positive), which appear in `results.txt`, `summary/*_summary.txt` and `-v`. `E_KO = E_lat + q*delta_v` is added to the formation energy exactly as before.
- `examples/*.yaml`: the charged-state placeholder energies had been tuned against the old, oversized correction. They were shifted mechanically by `E_KO(old) - E_KO(new)`, so every example's formation energies and transition levels are unchanged.
- `tests/golden/` was regenerated from the corrected code. `tests/test_ewald_physics.py` validates the new reference against physics.

## 0.1.0a1

- First alpha release.
