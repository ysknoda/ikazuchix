---
status: accepted
date: 2026-09-30
---

# Smoke-test acceptance criteria and in-flight bug policy

The run passes only if every criterion below holds. None of them compare against literature values (see ADR-0001).

1. Every VASP run reaches electronic SCF convergence and its OUTCAR terminates normally. O2 shows a 2.00 μB moment.
2. For every charge state, OUTCAR `NELECT` equals N0 − q. This catches charge-sign mix-ups.
3. The generated YAML passes `--dry-run`, and the host-stability residual is ≈ 0.
4. A full run (`--solve-fermi --stability-diagram -p -v`) completes, and every PNG is rendered and inspected at pixel level.
5. `mpirun -n 4 ikazuchix` gives output numerically identical to the single-process run.
6. Sanity checks:
   - E_lat scales ≈ q² within each defect.
   - `delta_v` is O(0.1) eV and flips sign with q.
   - The LOCPOT and site-potential `delta_v` are both recorded, together with their difference (ADR-0002).
7. The full existing test suite (285 tests at the start of the run) still passes whenever the package is modified.

**Bug policy:** obvious package defects found during the run are fixed on the spot, each with a regression test and a full-suite rerun. Examples are crashes, parse errors and unit mix-ups. Anything that changes the physics model is reported only, never fixed in this task (e.g. ADR-0002's V_PC).
