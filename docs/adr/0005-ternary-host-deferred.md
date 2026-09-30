---
status: accepted
date: 2026-09-30
---

# Stability-region polygon path is not covered by the AlN smoke test; a ternary host is a follow-up task

`feasible_region_2d` and `plot_stability_region` only run for a host with exactly three species (`ikazuchix/stability_region.py`). For binary AlN, `--stability-diagram` performs only the point-feasibility check. We accept this gap in the AlN smoke test and record it as untested in the report. The polygon path is already covered by unit tests and does not touch VASP I/O.

Once the AlN smoke test is complete, VASP validation of a ternary host will be scoped as a **separate task**. The system, cell size and budget are to be decided then; that task gets its own ADR.
