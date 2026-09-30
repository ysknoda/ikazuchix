---
status: superseded by ADR-0009 (the point-charge subtraction and the sign convention; the neutral offset is kept)
date: 2026-09-30
---

# `delta_v` is fed as the raw far-field potential difference; ikazuchix's alignment model is left unchanged

ikazuchix computes the alignment term as `E_alignment = q * delta_v` (`ikazuchix/ewald.py`). It does **not** subtract the point-charge model potential V_PC, which the original Kumagai–Oba scheme averages as (ΔV − V_PC) over a sampling region. For the smoke test we keep the package as is and feed `delta_v` exactly as the package defines it: `V_defect(far) − V_bulk(far)`, taken as-is from VASP.

- **Primary value (written to YAML):** the difference of the LOCPOT (`LVHAR=.TRUE.`) planar averages along c, at the plane farthest from the defect. This follows the CLI help TIP.
- **Cross-check (reported only):** the difference of OUTCAR atomic-site electrostatic potentials, averaged over atoms beyond a radius R from the defect. Because no ions are relaxed, bulk and defect atoms are matched one-to-one by index.

## Considered Options

- Change ikazuchix to subtract V_PC now — rejected. This is a physics-model change, and it should be decided on the evidence from this run in a separate task.

## Consequences

- The missing V_PC subtraction and the LOCPOT sign convention (electron potential energy vs electrostatic potential) are recorded as **findings** in the report, not fixed.
- The run checks that `delta_v` changes sign between q > 0 and q < 0, and records any large disagreement between the two extraction methods. A large disagreement is itself an I/O finding.
