---
status: accepted
date: 2026-09-30
---

# Correct the Kumagai–Oba Ewald lattice energy, and re-baseline the goldens on the corrected physics

The real-VASP smoke test (ADR-0001–0004, finding F1 in `vasp_smoke/REPORT.md`) showed that the KO lattice term depended on the Ewald splitting parameter η. On a 32-atom AlN cell it was +3.13 eV, where an independent calculation gives +0.312 eV. Up to 0.1.0a1 the sum had three errors:

- The reciprocal-space term was missing a factor ½.
- The reciprocal-space term used G·ε⁻¹·G with a 1/√det ε prefactor, instead of the screened G·ε·G.
- The real-space and background terms used a scalar-ε̄ convention that was inconsistent with the reciprocal-space term.

The errors dated from the original monolithic script and were locked in by the bit-for-bit golden tests. We replaced the sum with a single consistent anisotropic form (Rurali & Cartoixà 2009; Kumagai & Oba 2014, Eq. 8):

E_PC = (q²K/2)·[Σ_R erfc(η r_R)/(√det ε · r_R) + (4π/V) Σ_G e^(−G·ε·G/4η²)/(G·ε·G) − 2η/(√π √det ε) − π/(Vη²)], with r_R = √(R·ε⁻¹·R).

The correction is reported as E_lat = −E_PC > 0. E_KO = E_lat + q·ΔV is still added to E_f, so no downstream code changed.

## Decisions

- **Goldens are regenerated from the corrected code**, not kept for backward compatibility. There is no `--legacy-ko` path. The new golden is trustworthy because the physics tests validate it, not because it matches an earlier implementation. Those tests are `tests/test_ewald_physics.py`:
  - η invariance to 1e-8 eV;
  - the simple-cubic Madelung constant α = 2.837297 at ε = 1, 5 and 12.3;
  - an independent brute-force reference on a triclinic cell with a non-diagonal ε;
  - joint-rotation invariance, 1/ε scaling and q² scaling.
- **The default η becomes √(π ε̄)/V^(1/3)**, up from √π/V^(1/3). In the dielectric metric, real-space distances shrink by about √ε̄ and reciprocal ones grow by the same factor. Without this rescaling the fixed 3-shell default left an outer-shell error of about 4×10⁻² eV; with it, the error is below 10⁻⁸ eV. An explicit `--ewald-eta` is used verbatim.
- **Example placeholders are shifted, not re-invented.** Each charged state's energy moved by E_KO(old) − E_KO(new), so every example's E_f and CTLs are identical to 0.1.0a1 (within 1e-6 eV, from rounding to six decimals). A hand retune was rejected because the values are placeholders, and any "physical" retune would be arbitrary.
- **Released as 0.1.0a2** with a correctness notice in the README and a CHANGELOG, because 0.1.0a1 is public on GitHub.

## Out of scope (unchanged, see ADR-0002)

- The alignment term q·ΔV, and whether a model potential V_PC should be subtracted from ΔV.
- The LOCPOT sign convention.
