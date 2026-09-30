---
status: accepted
date: 2026-09-30
---

# Spinel MgAl2O4 is the ternary-host VASP smoke test

The AlN smoke test (ADR-0003) could not exercise the stability-region polygon or its plot, which exist only for three-species hosts (ADR-0005). MgAl2O4 was chosen for four reasons:

- It reuses the AlN run's Al, O2 and Al2O3 calculations, which have identical settings.
- It adds only Mg (hcp) and MgO (rock salt).
- Its antisites (Mg_Al, Al_Mg) exercise substitution defects with both `removed_atoms` and `added_atoms`, including automatic site densities.
- Its small stabilization against MgO + Al2O3 makes the polygon a sharp test of boundary handling.

Rejected alternatives:

- SrTiO3: about 320 electrons in a 40-atom cell.
- ZnSnN2: d electrons and complex competing phases.

As in ADR-0001, agreement with literature values is out of scope.

## Protocol (settings identical to ADR-0003 unless stated)

| Calculation | Cell / k-mesh | Notes |
|---|---|---|
| Host | primitive, 14 atoms, Γ-centred 6×6×6, `ISIF=3` | Start: Fd-3m, a = 8.083 Å, O 32e x = 0.2624 (origin choice 2), Mg 8a, Al 16d. Its energy is `bulk.formation_reference` (ADR-0007). |
| DFPT | relaxed primitive, 4×4×4 | Expect an isotropic tensor (cubic). |
| Bulk supercell | conventional, 56 atoms, Γ only (`vasp_gam`) | `LVHAR`. |
| Defects | same cell, unrelaxed, Γ only, `ISPIN=2` | V_O 0/+1/+2; V_Mg 0/−1/−2; Mg_Al 0/−1; Al_Mg 0/+1. Each site is the one nearest (0.5, 0.5, 0.5). |
| Mg reference | hcp, 2 atoms, Γ-centred 16×16×10, `ISIF=3`, `ISMEAR=1`, `SIGMA=0.2` | New. |
| MgO | rock salt, 2-atom primitive, 8×8×8, `ISIF=3` | New competing phase. |
| Al, O2, Al2O3 | — | Reused from the AlN run. |

Additional settings:

- POTCAR: standard `Mg` (2 valence electrons, ENMAX 200 eV), plus the existing `Al` and `O`.
- Competing phases: MgO and Al2O3 only. Mg–Al intermetallics (Al3Mg2, Mg17Al12) are left out for cost, so the metal-rich corner is not cut by them; this is recorded as a known gap.
- Budget: `-np 4`, at most 2 h of wall time, run resumably.

## Gate

The thermodynamic calculations (host primitive, Mg, MgO) run first. If the stability region computed from them is empty, the defect calculations do not start and the user is consulted.

## Chemical-potential points

Separate YAML inputs are run for each polygon vertex, for the centroid, and for one point outside the polygon; the outside point must be reported as a violation.

## Related package change

`free_species_order()` now orders species by atomic number instead of alphabetically, per the user's convention for species axes. For MgAl2O4 the plot axes are x = O and y = Mg, and Al is eliminated. Only ternary polygon output is affected; the binary goldens are not.

## Artifacts

The extracted numbers-only YAML inputs for AlN and MgAl2O4 go to `examples/`, labelled as unconverged small-cell data. A regression test checks the polygon vertices against independently computed constraint intersections.
