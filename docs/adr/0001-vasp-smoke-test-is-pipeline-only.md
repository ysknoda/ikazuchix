---
status: accepted
date: 2026-09-30
---

# VASP smoke test validates the VASP → YAML → ikazuchix pipeline, not in-package readers

ikazuchix reads no VASP files: every physical input (total energies, `delta_v`, VBM/CBM, dielectric tensor) is a number hand-copied into the YAML input. To exercise the package against real DFT data we run small-cell VASP 6.6.1 calculations and extract those numbers with an **external** glue script, rather than adding VASP readers to the package. The goal is an end-to-end check of the package's behaviour on real data (magnitudes, signs, units, KO-correction behaviour); agreement with literature values is explicitly out of scope.

## Considered Options

- **In-package VASP readers** (e.g. `energy: {from: OUTCAR}` or an `ikazuchix-from-vasp` command) — rejected for now; the package keeps its two-dependency core (numpy, PyYAML). May be revisited once the manual transcription pain is measured.
- **External glue script (chosen)** — a single self-contained Python script with argparse subcommands that (1) generates VASP inputs, (2) drives the runs resumably (one completion marker per calculation, `nohup` + log file), and (3) parses OUTCAR / LOCPOT / vasprun.xml with pymatgen and writes the ikazuchix YAML. pymatgen is used by the script only; it is **not** added to ikazuchix's dependencies.

## Consequences

- All VASP work, the glue script, generated YAML and ikazuchix outputs live in a git-ignored `vasp_smoke/` directory. POTCARs cannot be committed (license), and WAVECAR/LOCPOT are large.
- VASP environment: `source ~/vasp/vasp661_env.sh` (puts Homebrew OpenMPI ahead of miniconda's broken `mpirun`); POTCARs from `$VASP_POTCAR_DIR` (`potpaw_PBE.64`).
- Compute budget: `mpirun -np 4` (the four performance cores of the Apple M4; efficiency cores are left out to avoid synchronisation stalls), about 1 hour of wall time in total.
- The results are written up in `vasp_smoke/REPORT.md`.
