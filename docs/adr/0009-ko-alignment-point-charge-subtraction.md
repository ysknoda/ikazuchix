---
status: accepted
date: 2026-09-30
supersedes: the "report only, do not fix" part of ADR-0002
---

# Subtract the point-charge potential from the alignment term, sampled at atomic sites (Kumagai–Oba)

ikazuchix computed the alignment as `q * delta_v`, where `delta_v` was a raw far-field DFT potential difference. That raw ΔV still contains the tail of the defect's own point-charge potential, which the lattice term E_lat already accounts for, so it was counted twice.

The real AlN and MgAl2O4 smoke data settle both the model and the sign. VASP's site potentials (OUTCAR) and LOCPOT are electron potential energies, −eφ. With that convention, subtracting the KO model potential leaves a residual ΔV_q/b + φ_PC that:
- matches the neutral state's value, e.g. MgAl2O4 V_Mg gives −0.549 / −0.533 / −0.582 eV for q = 0 / −1 / −2;
- is flatter in distance for every charged defect.

The opposite convention drifts in proportion to q. With the electron-PE convention, ikazuchix's existing `+q * delta_v` sign is correct. The size of the missing subtraction is q·⟨φ_PC⟩, e.g. −2.01 eV for AlN V_Al with q = −3 in a 32-atom cell.

## Decisions

- **Package.**
  - A new optional per-defect field `alignment_sampling_sites` lists the fractional coordinates of the atoms over which the raw `delta_v` was averaged.
  - When it is present, E_alignment = q·(delta_v + ⟨φ_PC⟩_sites). φ_PC is the anisotropic Ewald potential of the periodic, background-neutralized point charge, computed from the same formula as E_PC.
  - When it is absent, `delta_v` is taken as already model-subtracted, which is the existing behaviour. Existing inputs and goldens are unchanged.
  - `ikazuchix.ewald.point_charge_potential` is public, so other workflows can use the same implementation.
- **Sign convention (documented).** `delta_v` = V_defect − V_bulk exactly as VASP reports it (OUTCAR site potentials or LOCPOT), i.e. in the electron potential-energy convention.
- **Raw ΔV source.** Atomic-site potentials from OUTCAR, following KO. LOCPOT planar averages are kept only as a cross-check in the smoke report, because subtracting a point model from a planar average would need a planar-averaged model.
- **Sampling region.** Following KO, use atoms outside the largest sphere inscribed in the supercell's Wigner–Seitz cell: minimum-image distance from the defect > min over R ≠ 0 of |R|/2. The glue script applies this rule; the package only receives coordinates.
- **Neutral offset (finding F2).** It is kept. ΔV_q/b is the defect-minus-bulk difference, as in KO, not ΔV_q/0. The residual after subtraction equals the neutral-state value, confirming that it is a genuine short-range reference shift and not an artifact of the model.

## Consequences

- The real-data examples `examples/*_vasp_smoke.yaml` are regenerated with sampling sites, and their transition levels change.
- The synthetic examples and the goldens are unchanged.
- Target release: 0.1.0a3, held until GitHub Actions usage is available again.
