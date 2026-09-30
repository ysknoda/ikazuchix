# Changelog

## Unreleased

### Added

- Optional `bulk.formation_reference: {energy, composition}` holds a k-converged host energy for the host formation enthalpy: the host-stability checks and `--stability-diagram`. The defect-supercell `bulk.energy` stays the reference for `E_def - E_bulk`. A stoichiometry mismatch with `bulk.composition` is an input error. See `docs/adr/0007`.

### Changed

- The stability-region axes are ordered by atomic number, not alphabetically. For MgAl2O4 the x and y axes are `O` and `Mg`, and `Al` is eliminated. Labels that are not element symbols sort last.

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
