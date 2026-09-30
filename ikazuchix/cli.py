"""Command-line argument parsing and help text."""

import argparse
import textwrap

# ===========================================================================
# CLI
# ===========================================================================

_DESC = textwrap.dedent("""\
    ikazuchix -- Defect Formation Energy Calculator with Kumagai-Oba Finite-Size Correction
    ===========================================================================
    Computes the formation energy of point defects in semiconductors:

      E_f(q, E_F) = (E_defect - E_bulk) - sum_i n_i*mu_i + q*(E_VBM + E_F) + E_KO

    where E_KO is the Kumagai-Oba charge correction (PRB 89, 195205, 2014).
    Multiple defect types and charge states are handled in a single run.
""")

_REQUIRED_DATA = textwrap.dedent("""\
  +-------------------------------------------------------------------------+
  |              REQUIRED INPUT DATA  (all supplied via YAML)               |
  +--------------------------+----------------------------------------------+
  | Section / Key            | Description                                  |
  +--------------------------+----------------------------------------------+
  | [bulk]                   |                                              |
  |   formula                | Chemical formula of the host material        |
  |   energy          [eV]   | DFT total energy of the perfect supercell    |
  |   num_atoms              | Number of atoms in the bulk supercell        |
  |   composition            | Atom counts  {species: N, ...}               |
  +--------------------------+----------------------------------------------+
  | [chemical_potentials]    | Deltamu_i = deviation from elemental reference    |
  |   <species>       [eV]   | e.g.  Ga: 0.000  (Ga-rich: Deltamu_Ga = 0)      |
  |                          | Constrained by host stability condition:     |
  |                          |   sum_i x_i*mu_i = E[host] / f.u.             |
  +--------------------------+----------------------------------------------+
  | [elemental_references]   | DFT total energy per atom of elemental solid |
  |   <species>    [eV/atom] | or molecule used as the chemical reference   |
  |                          | e.g.  Ga: -3.031  (bulk alpha-Ga, PBE)         |
  |                          |        N: -8.328  (1/2 E[N_2], PBE)           |
  +--------------------------+----------------------------------------------+
  | [band_structure]         |                                              |
  |   vbm             [eV]   | Valence band maximum (absolute DFT value)    |
  |   cbm             [eV]   | Conduction band minimum                      |
  |   bandgap         [eV]   | Band gap (= cbm - vbm)                       |
  |   fermi_level     [eV]   | Default reference Fermi level (absolute)     |
  |                          | Note: scissors correction can be applied by  |
  |                          |   setting cbm to the corrected value while   |
  |                          |   keeping PBE total energies for E_defect    |
  +--------------------------+----------------------------------------------+
  | [dielectric]             |                                              |
  |   tensor    [3x3, dim'less] Full static dielectric tensor (electronic  |
  |                          | + ionic contributions).  Use the tensor from |
  |                          | a bulk DFPT calculation.  Diagonal for cubic |
  |                          | or hexagonal systems (off-diagonal = 0).     |
  +--------------------------+----------------------------------------------+
  | [supercell]              | Geometry of the defect supercell             |
  |   lattice_vectors  [Ang]   | a, b, c vectors as 3-element lists          |
  |   defect_position        | Fractional coordinates of the defect site    |
  |                          | (Can be overridden per defect entry)         |
  +--------------------------+----------------------------------------------+
  | [defects]  (list)        | One entry per defect type                    |
  |   name                   | Label string, e.g. "V_Ga"                   |
  |   defect_type            | vacancy | interstitial | substitution        |
  |   removed_atoms          | Species removed: {Ga: 1}  (vacancy)         |
  |   added_atoms            | Species added:   {N: 1}   (interstitial)    |
  |   [charge_states] (list) | One entry per charge state q                |
  |     charge               | Integer charge q (negative = acceptor-like)  |
  |     energy        [eV]   | DFT total energy of the defect supercell     |
  |     num_atoms            | Number of atoms in the defect supercell      |
  |     composition          | Atom counts (optional: auto-computed from    |
  |                          | bulk + removed_atoms + added_atoms)          |
  |     potential_alignment  |                                              |
  |       delta_v     [eV]   | V_defect - V_bulk averaged over far atomic   |
  |                          | sites, in VASP's sign (electron potential    |
  |                          | energy). Raw if alignment_sampling_sites is  |
  |                          | given (phi_PC is then subtracted), else      |
  |                          | already point-charge subtracted.             |
  |   alignment_sampling_    | Optional per-defect list of fractional       |
  |     sites                | [x,y,z] sites delta_v was averaged over      |
  +--------------------------+----------------------------------------------+

  TIP -- obtaining delta_v from VASP (Kumagai-Oba, docs/adr/0009):
    Average the OUTCAR per-atom site potentials V_defect - V_bulk over atoms
    outside the Wigner-Seitz inscribed sphere (distance > min|R|/2), and
    list those atoms' fractional coordinates in alignment_sampling_sites.
""")

_OUTPUTS = textwrap.dedent("""\
  OUTPUT FILES  (all written under --out-dir, default: current directory)
  ------------
    results.txt                    Full text report (all defects & charges)
    ef_vs_ef/<name>_ef_vs_ef.txt   E_f(q, E_F) table + stable envelope
    summary/<name>_summary.txt     KO correction breakdown + E_f at E_F_ref
    ctls_summary.txt               Unified CTL table (stable + meta-stable)
    plots/<name>_ef.png (with -p)  Per-defect 2-panel plot
    plots/comparison_plot.png (-p) All defect envelopes overlaid
""")

_EXAMPLES = textwrap.dedent("""\
  EXAMPLES
  --------
    # Minimal run (single-point report at E_F = VBM):
    python -m ikazuchix -i input.yaml

    # Set reference Fermi level and generate plots:
    python -m ikazuchix -i input.yaml -f 1.5 -p

    # Extend sweep below VBM and above CBM, all output to 'results/' dir:
    python -m ikazuchix -i input.yaml -f -0.3 \\
        --fermi-min -0.5 --fermi-max 4.0 -p --out-dir results/

    # Dry-run: validate input only, no calculation:
    python -m ikazuchix -i input.yaml --dry-run

    # Verbose KO details:
    python -m ikazuchix -i input.yaml -v

    # High-precision adaptive Ewald convergence (opt-in; default output
    # is unaffected unless this flag is given):
    python -m ikazuchix -i input.yaml --adaptive-ewald

    # Distribute the (defect, charge_state) work list across 4 MPI
    # ranks (falls back to a single process automatically if mpi4py /
    # an MPI runtime are not installed -- results are identical either
    # way, only wall-clock time differs):
    mpirun -n 4 python -m ikazuchix -i input.yaml

  REFERENCES
  ----------
    [1] Y. Kumagai and F. Oba, Phys. Rev. B 89, 195205 (2014).
        -- Finite-size correction method (KO correction)
    [2] C. Freysoldt et al., Rev. Mod. Phys. 86, 253 (2014).
        -- Comprehensive review of defect calculations
    [3] J. L. Lyons and C. G. Van de Walle, npj Comp. Mat. 3, 12 (2017).
        -- Example application for GaN point defects

  DEPENDENCIES
  ------------
    Required : numpy, pyyaml
    Optional : matplotlib  (needed only for --plot)
               numba       (JIT-accelerates the Ewald summation; falls
                            back to plain Python if absent -- same
                            results, just slower)
               mpi4py      (distributes work across MPI ranks under
                            mpirun; falls back to a single process if
                            absent, or if installed but no MPI runtime
                            such as OpenMPI/MPICH is available -- same
                            results, just slower)
    Install  : pip install numpy pyyaml matplotlib numba mpi4py
               (mpi4py additionally requires a system MPI runtime, e.g.
               `conda install -c conda-forge openmpi mpi4py`)
""")


def parse_args():
    p = argparse.ArgumentParser(
        prog="ikazuchix",
        description=_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_REQUIRED_DATA + _OUTPUTS + _EXAMPLES,
        add_help=True,
    )

    # --- Input / output (B-3: unified --out-dir) ----------------------------
    io_grp = p.add_argument_group("Input / Output")
    io_grp.add_argument(
        "-i", "--input",
        required=True, metavar="FILE",
        help="Path to the YAML input file (required).")
    io_grp.add_argument(
        "--out-dir",
        default=".", dest="out_dir", metavar="DIR",
        help=(
            "Root directory for ALL output files.  Sub-directories "
            "ef_vs_ef/, summary/, and plots/ are created automatically "
            "inside this directory (default: current directory '.')."))
    io_grp.add_argument(
        "-o", "--output",
        default="results.txt", metavar="FILE",
        help=(
            "Filename for the main plain-text report "
            "(written inside --out-dir, default: results.txt)."))

    # --- Fermi level ---------------------------------------------------------
    ef_grp = p.add_argument_group("Fermi-level control")
    ef_grp.add_argument(
        "-f", "--fermi",
        type=float, default=None, metavar="eV",
        help=(
            "Reference Fermi level relative to the VBM [eV].  "
            "Negative values (below VBM) are fully supported.  "
            "Overrides the 'fermi_level' field in the input file."))
    ef_grp.add_argument(
        "--fermi-min",
        type=float, default=None, dest="fermi_min", metavar="eV",
        help=(
            "Lower bound of the Fermi-level sweep [eV, rel. VBM].  "
            "Negative values extend the plot/table below the VBM.  "
            "Default: auto = min(0, E_F_ref) - 10%% of band gap."))
    ef_grp.add_argument(
        "--fermi-max",
        type=float, default=None, dest="fermi_max", metavar="eV",
        help=(
            "Upper bound of the Fermi-level sweep [eV, rel. VBM].  "
            "Default: auto = max(gap, E_F_ref) + 10%% of band gap."))

    # --- Plotting ------------------------------------------------------------
    plt_grp = p.add_argument_group("Plotting  (requires matplotlib)")
    plt_grp.add_argument(
        "-p", "--plot",
        action="store_true",
        help=(
            "Generate plots.  Produces one two-panel PNG per defect "
            "(E_f vs E_F with stable CTL markers + E_KO bar chart) and "
            "one comparison PNG with all defect envelopes overlaid."))
    plt_grp.add_argument(
        "--comparison-plot",
        default="comparison_plot.png", dest="comparison_plot", metavar="FILE",
        help=(
            "Filename for the multi-defect comparison PNG "
            "(written inside --out-dir/plots/, "
            "default: comparison_plot.png)."))

    # --- Ewald summation (precision / performance) --------------------------
    ewald_grp = p.add_argument_group("Ewald summation  (precision / performance)")
    ewald_grp.add_argument(
        "--real-cutoff",
        type=int, default=3, dest="real_cutoff", metavar="N",
        help=(
            "Real-space Ewald shell cutoff (starting cutoff in "
            "--adaptive-ewald mode).  Default: 3 (matches the original "
            "hardcoded value)."))
    ewald_grp.add_argument(
        "--recip-cutoff",
        type=int, default=3, dest="recip_cutoff", metavar="N",
        help=(
            "Reciprocal-space Ewald shell cutoff (starting cutoff in "
            "--adaptive-ewald mode).  Default: 3 (matches the original "
            "hardcoded value)."))
    ewald_grp.add_argument(
        "--ewald-eta",
        type=float, default=None, dest="ewald_eta", metavar="Ang^-1",
        help=(
            "Ewald splitting parameter eta [Ang^-1].  "
            "Default: auto = sqrt(pi * eps_avg) / V^(1/3), eps_avg = tr(eps)/3."))
    ewald_grp.add_argument(
        "--adaptive-ewald",
        action="store_true", dest="adaptive_ewald",
        help=(
            "Opt-in high-precision mode: automatically grow the real- "
            "and reciprocal-space shell cutoffs (starting from "
            "--real-cutoff / --recip-cutoff) until the outermost-shell "
            "contribution drops below --ewald-tol, instead of only "
            "warning at the fixed starting cutoff.  Default output is "
            "unaffected unless this flag is given."))
    ewald_grp.add_argument(
        "--ewald-tol",
        type=float, default=None, dest="ewald_tol", metavar="eV",
        help=(
            "Convergence tolerance used by --adaptive-ewald "
            "[eV].  Default: same as the fixed-cutoff convergence "
            "warning threshold (1e-6 eV)."))
    ewald_grp.add_argument(
        "--max-ewald-shells",
        type=int, default=15, dest="max_ewald_shells", metavar="N",
        help=(
            "Safety cap on the shell count --adaptive-ewald may grow "
            "to.  If convergence is not reached within this many "
            "shells, a warning is emitted and the best available "
            "result is used.  Default: 15."))

    # --- Parallelization (MPI / Numba) ---------------------------------------
    par_grp = p.add_argument_group("Parallelization  (MPI / Numba)")
    par_grp.add_argument(
        "--no-mpi",
        action="store_true", dest="no_mpi",
        help=(
            "Force single-process execution even if mpi4py and a "
            "working MPI runtime are available (and even when actually "
            "launched via mpirun -n N>1, in which case every rank "
            "redundantly computes and writes the full output -- a "
            "warning is printed in that case).  Has no effect on results, "
            "only on how they are computed."))
    par_grp.add_argument(
        "--no-numba",
        action="store_true", dest="no_numba",
        help=(
            "Force the pure-Python Ewald summation kernels even if "
            "numba is installed.  Has no effect on results (numba "
            "acceleration is bit-for-bit identical to the pure-Python "
            "path), only on speed."))

    # --- Equilibrium Fermi level & concentrations (opt-in) -------------------
    eq_grp = p.add_argument_group("Equilibrium Fermi level & concentrations")
    eq_grp.add_argument(
        "--solve-fermi",
        action="store_true", dest="solve_fermi",
        help=(
            "Opt-in: solve the self-consistent Fermi level from charge "
            "neutrality (sum of ionized defect charges + hole/electron "
            "concentrations = 0), and report equilibrium defect and "
            "carrier concentrations.  Bisects over "
            "[--fermi-min, --fermi-max] (default, i.e. when neither is "
            "given: the band gap [0, gap]) -- widen these if the solver "
            "reports no root bracketed.  Requires a 'site_density' for "
            "any defect whose site count cannot be auto-derived (see "
            "--help epilog).  Writes equilibrium_summary.txt.  Default "
            "output is unaffected unless this flag is given."))
    eq_grp.add_argument(
        "--temperature",
        type=float, default=300.0, dest="temperature", metavar="K",
        help="Temperature used for --solve-fermi concentrations [K].  "
             "Default: 300.0.")
    eq_grp.add_argument(
        "--no-carriers",
        action="store_true", dest="no_carriers",
        help=(
            "With --solve-fermi: exclude the electron/hole terms from "
            "the charge-neutrality equation, solving purely for "
            "charge compensation between defects."))
    eq_grp.add_argument(
        "--fermi-solve-tol",
        type=float, default=1e-8, dest="fermi_solve_tol", metavar="eV",
        help="Bisection convergence tolerance for --solve-fermi [eV].  "
             "Default: 1e-8.")

    # --- Chemical potential stability region (opt-in) ------------------------
    stab_grp = p.add_argument_group("Chemical potential stability region")
    stab_grp.add_argument(
        "--stability-diagram",
        action="store_true", dest="stability_diagram",
        help=(
            "Opt-in: check the input file's 'chemical_potentials' "
            "against the host stability condition, elemental bounds, "
            "and any explicit 'competing_phases' entries, and (for a "
            "ternary host) compute the 2D chemical-potential stability "
            "region.  Writes stability_summary.txt.  Default output is "
            "unaffected unless this flag is given."))
    stab_grp.add_argument(
        "--stability-plot",
        default="stability_region.png", dest="stability_plot", metavar="FILE",
        help=(
            "Filename for the stability-region plot (ternary hosts "
            "only; written inside --out-dir/plots/, requires -p/--plot "
            "together with --stability-diagram).  "
            "Default: stability_region.png."))

    # --- Diagnostics / utilities --------------------------------------------
    diag_grp = p.add_argument_group("Diagnostics / Utilities")
    diag_grp.add_argument(
        "-v", "--verbose",
        action="store_true",
        help=(
            "Print intermediate Kumagai-Oba correction values "
            "(E_real, E_recip, E_self, E_neut, E_PC, E_lat, eta, eps_avg, V) "
            "for every charge state."))
    diag_grp.add_argument(
        "--dry-run",
        action="store_true", dest="dry_run",
        help=(
            "Validate the input file and check chemical potential "
            "consistency without performing any DFT energy calculations.  "
            "Useful for catching input errors before a long run."))

    args = p.parse_args()
    if args.real_cutoff < 1:
        p.error(f"--real-cutoff must be >= 1 (got {args.real_cutoff}).")
    if args.recip_cutoff < 1:
        p.error(f"--recip-cutoff must be >= 1 (got {args.recip_cutoff}).")
    if args.max_ewald_shells < 1:
        p.error(f"--max-ewald-shells must be >= 1 (got {args.max_ewald_shells}).")
    if args.temperature <= 0:
        p.error(f"--temperature must be positive (got {args.temperature}).")
    if args.fermi_solve_tol <= 0:
        p.error(f"--fermi-solve-tol must be positive (got {args.fermi_solve_tol}).")
    return args
