---
status: accepted
date: 2026-09-30
---

# Wurtzite AlN in a 32-atom Γ-only supercell is the smoke-test system

We need a binary host, so that `--stability-diagram` and the host-stability checks are non-trivial. We also want a non-orthogonal lattice with an anisotropic dielectric tensor, and a substitutional impurity that introduces a third species. Wurtzite AlN with V_Al, V_N and O_N covers all of these, matches `examples/AlN_test_input.yaml`, and fits the 1-hour budget in a 32-atom cell. None of the settings below are convergence-tested: they are fit for I/O validation, not for physics.

## Protocol

| Calculation | Cell | Binary / k-points | Key settings |
|---|---|---|---|
| Host lattice | AlN primitive (4 atoms) | `vasp_std`, Γ-centred 8×8×5 | `ISIF=3` PBE relaxation; the relaxed a, c, u are used everywhere below |
| Dielectric tensor | relaxed primitive | `vasp_std`, Γ-centred 8×8×5 | `IBRION=8`, `LEPSILON=.TRUE.`, `EDIFF=1e-8`; YAML gets electronic (local-field) + ionic tensors summed; expect off-diagonals ≈ 0 and xx = yy ≠ zz |
| Bulk supercell | 2×2×2 of primitive, 32 atoms | `vasp_gam`, Γ only | `LVHAR=.TRUE.`; VBM/CBM taken at Γ (AlN gap is direct at Γ) |
| Defects | same 32-atom cell, unrelaxed (`NSW=0`) | `vasp_gam`, Γ only | V_Al q = 0,−1,−2,−3; V_N q = 0,+1,+2,+3; O_N q = 0,+1 (10 runs); `ISPIN=2`, `NUPDOWN` free, default MAGMOM; `NELECT = N0 − q`; `LVHAR=.TRUE.` |
| Al reference | fcc, 1 atom | `vasp_std`, Γ-centred 16×16×16 | `ISIF=3`, `ISMEAR=1`, `SIGMA=0.2`, uses `energy(sigma->0)` |
| N reference | N2 in 12×12.5×13 Å box | `vasp_gam` | `ISPIN=1`, bond relaxed (`IBRION=2`, `ISIF=2`) |
| O reference | O2 in the same box | `vasp_gam` | triplet ³Σg⁻: `ISPIN=2`, `NUPDOWN=2`, `MAGMOM=2*1.0`, bond relaxed; the run must show a 2.00 μB moment |
| Competing phase | corundum Al2O3, 10-atom primitive | `vasp_std`, Γ-centred 6×6×6 | `ISIF=3`; exercises the `competing_phases` code path |

Common settings: PAW-PBE `Al`, `N`, `O` (standard, potpaw_PBE.64). `ENCUT=520` eV (1.3 × the 400 eV ENMAX of N/O, the usual Pulay-stress margin for `ISIF=3`). `PREC=Accurate`, `LASPH=.TRUE.`, `LREAL=.FALSE.`, `EDIFF=1e-6`. `ISMEAR=0`, `SIGMA=0.05` for insulators and molecules. `LWAVE=LCHARG=.FALSE.`

Defect site: for each species, the site nearest to fractional (0.5, 0.5, 0.5) of the supercell. Its fractional coordinates are written to `defect_position` verbatim.

## Derived YAML inputs

- Band gap: the PBE gap of the bulk supercell. All numbers come from VASP, and the PBE underestimate is accepted as out of scope. Effective masses are copied from `examples/AlN_test_input.yaml` as placeholders.
- Chemical potentials (main run): Al-rich limit, Δμ_Al = 0, Δμ_N = ΔH_f(AlN), and Δμ_O = ΔH_f(Al2O3)/3. This point lies exactly on the Al2O3 boundary, so it tests the tolerance handling. An extra `--dry-run --stability-diagram` at Δμ_O = 0 must report an Al2O3 violation.

## Considered Options

- MgO — cheaper, but its isotropic dielectric tensor weakly tests the tensor handling.
- 72-atom 3×3×2 cell, or a 2×2×2 k-mesh — exceeds the time budget.
- Experimental lattice constants — DFPT on a stressed cell risks an unphysical ionic dielectric response.
- Dropping O_N / Al2O3 — would leave the substitution, three-species and competing-phase paths untested.
