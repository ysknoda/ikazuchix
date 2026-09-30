---
status: accepted
date: 2026-09-30
---

# Optional `bulk.formation_reference` decouples the host's thermodynamic energy from the defect-supercell energy

Until now `bulk.energy` played two roles. It was the reference in E_def − E_bulk, where it must match the defect supercells' settings (same cell, same k-mesh). It was also the host energy in ΔH_f, the host-stability check and the stability region, where it must be k-point converged just like the elemental references and competing phases it is compared against.

With a Γ-only 32-atom AlN supercell, the two roles disagreed by 0.62 eV per formula unit (finding F4 in `vasp_smoke/REPORT.md`). For a host that is only marginally stable against its competing phases, as spinel MgAl2O4 is against MgO + Al2O3, a shift of that size empties the stability region outright.

We add an optional `bulk.formation_reference: {energy, composition}`. It holds a k-converged host calculation in any cell, e.g. the primitive cell. When present, it is used for:

- the host-stability and decomposition-bound checks in `check_chemical_potentials`;
- `stability_region`: the host equality, feasibility of the point, and the 2D polygon.

`bulk.energy` remains the reference for E_def − E_bulk. When `formation_reference` is absent, behaviour is exactly as before, so existing inputs and goldens are unchanged.

`formation_reference.composition` must have the same species and the same stoichiometric ratios as `bulk.composition`, within a 1e-6 relative tolerance; otherwise input loading raises an error. A different stoichiometry would silently produce a wrong ΔH_f.

This is standard practice rather than a physics-model change (ADR-0004): ΔH_f comes from a converged calculation of the host, and defect energies come from differences between supercells computed with identical settings.

## Considered Options

- Compute the bulk and every defect supercell with a converged k-mesh. Rejected: this multiplies the cost 4–8x for the MgAl2O4 smoke test and still ties ΔH_f to supercell settings.
- Accept the inconsistency and report it. Rejected: the ternary validation (ADR-0008) needs a non-empty stability region.
