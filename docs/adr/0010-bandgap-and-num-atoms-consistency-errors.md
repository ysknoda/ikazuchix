---
status: accepted
date: 2026-10-21
---

# Treat an inconsistent `bandgap` or `num_atoms` as an input error

Two input fields are redundant: each restates a quantity the package already derives from other fields.

- `band_structure.bandgap` restates `cbm − vbm`. Every calculation (the Fermi-level range, the plots, `--solve-fermi`) uses `cbm − vbm`, and `bandgap` is only printed in the `Band gap` line of the `results.txt` header. Nothing checked it.
- `num_atoms` (bulk and charge states) restates `sum(composition)`. It is never used in a calculation. A mismatch raised only a `UserWarning` on stderr, and loading continued.

Both issues came up while writing the GaN tutorial:

- **Scissor workflow.** A scissor correction is applied by editing `cbm` in the YAML. If `bandgap` is not edited too, `results.txt` reports the uncorrected gap while all numbers use the corrected one.
- **Lost warning.** A wrong `bulk.num_atoms` was visible only on stderr, which is lost when a user redirects only stdout (`> log`).

## Decisions

- **Bandgap.** `|bandgap − (cbm − vbm)| > 1e-3 eV` raises `ValueError` in `load_input`. We chose to raise rather than warn, because the inconsistency makes the report contradict itself.
- **Tolerance of 1e-3 eV.** Band edges are usually quoted to 1e-4–1e-3 eV, and 1e-3 eV is far below any physically meaningful gap difference. It also absorbs binary floating-point noise: 6.7 − 5.0 = 1.7000000000000002.
- **`num_atoms`.**
  - A declared `num_atoms` that differs from `sum(composition)` raises `ValueError`. So does a non-numeric value, where bool counts as non-numeric (YAML `yes`/`no`).
  - The check applies to the bulk and to every charge state, with an explicit or auto-completed composition.
  - The reason: `composition` feeds the chemical-potential term, so a mismatch means one of the two is wrong, and the package cannot tell which.
  - `num_atoms` stays optional; omitting it skips the check.
- **We did not derive `bandgap` or drop it.** Removing the field, or overwriting it with `cbm − vbm`, would silently change the meaning of existing inputs. Rejecting the inconsistency keeps the YAML self-describing.

## Consequences

- Every valid input (all example, smoke and tutorial inputs satisfy both checks) gives byte-identical output, and `tests/golden` is unchanged.
- Inputs that used to run with a stderr warning, or silently, now fail at load time with a message naming both values and the fix. This includes the GaN tutorial's deliberately broken `err_num_atoms.yaml`: its troubleshooting section must now show an error instead of a warning.
- The decision is covered by tests in `tests/test_io_utils.py`: mismatch raises, the tolerance boundary on both sides, float rounding, non-numeric `num_atoms`, and omitted `num_atoms`.
