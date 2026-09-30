"""CLI entry point: orchestrates load -> compute -> report/plot.

Runs identically under a plain ``python -m ikazuchix ...`` invocation
(single process) and under ``mpirun -n N python -m ikazuchix ...``
(N ranks): all file/plot output and most [INFO] logging happen only on
rank 0 -- see ikazuchix/mpi_utils.py for the MPI/NullComm fallback that
makes rank 0 the *only* rank whenever real MPI is unavailable or
--no-mpi is given.
"""

import numpy as np

import ikazuchix.mpi_utils as mpi_utils
from ikazuchix.chem_potentials import check_chemical_potentials
from ikazuchix.cli import parse_args
from ikazuchix.constants import N_FERMI_POINTS
from ikazuchix.equilibrium import solve_equilibrium
from ikazuchix.io_utils import (
    _out,
    _safe_mkdir,
    load_input,
    write_ctls_summary,
    write_ef_table,
    write_equilibrium_summary,
    write_stability_summary,
    write_summary_txt,
    write_text,
)
from ikazuchix.mpi_utils import get_comm, raise_consistently
from ikazuchix.numba_shim import set_numba_enabled
from ikazuchix.plotting import plot_comparison, plot_single_defect, plot_stability_region
from ikazuchix.report import format_report
from ikazuchix.runner import run_all_defects
from ikazuchix.stability_region import stability_summary


def main():
    args = parse_args()
    set_numba_enabled(not args.no_numba)

    if args.no_mpi and mpi_utils.HAS_MPI:
        real_comm = mpi_utils.MPI.COMM_WORLD
        if real_comm.Get_size() > 1 and real_comm.Get_rank() == 0:
            print(f"[WARNING] --no-mpi was given but this process is "
                  f"running as part of an MPI job with "
                  f"{real_comm.Get_size()} ranks; every rank will "
                  "redundantly compute and write the full output.")

    comm = get_comm(force_disable=args.no_mpi)
    rank = comm.Get_rank()
    ewald_kwargs = dict(
        real_cutoff=args.real_cutoff, recip_cutoff=args.recip_cutoff,
        ewald_param=args.ewald_eta, adaptive=args.adaptive_ewald,
        ewald_tol=args.ewald_tol, max_shells=args.max_ewald_shells,
    )

    # ------------------------------------------------------------------ #
    # 1. Load and validate input  -- rank 0 only, then broadcast so every
    #    rank works from an identical, already-normalised `data` dict
    #    without each one re-printing load_input's [INFO] lines.
    #
    #    A load/validation failure on rank 0 is caught here rather than
    #    left to propagate immediately: under real multi-rank MPI, every
    #    other rank is about to block waiting inside the comm.bcast()
    #    below, and if rank 0 simply died there would be nothing to wake
    #    them -- see mpi_utils.raise_consistently().
    # ------------------------------------------------------------------ #
    load_exc = None
    if rank == 0:
        print(f"[INFO] Loading: {args.input}")
        try:
            data = load_input(args.input)
        except Exception as exc:
            data = None
            load_exc = exc
    else:
        data = None
    raise_consistently(comm, load_exc)

    data = comm.bcast(data, root=0)

    bs  = data["band_structure"]
    vbm = bs["vbm"]
    cbm = bs["cbm"]
    gap = cbm - vbm

    # (A-4) Chemical potential consistency check
    if rank == 0:
        print("[INFO] Checking chemical potential consistency ...")
        check_chemical_potentials(data)

    # ------------------------------------------------------------------ #
    # 2. Fermi level and sweep grid  (C-2: built once, shared).
    #    Deterministic given data + args, so every rank computes the
    #    same values independently (mpirun launches an identical
    #    command line on every rank) -- no broadcast needed.
    # ------------------------------------------------------------------ #
    if args.fermi is not None:
        fermi_ref = float(args.fermi)
    else:
        fermi_ref = bs["fermi_level"] - vbm
    if rank == 0:
        print(f"[INFO] Reference Fermi level (rel. VBM): {fermi_ref:.4f} eV")
        if fermi_ref < 0:
            print(f"[INFO]   -> {abs(fermi_ref):.4f} eV below the VBM.")
        elif fermi_ref > gap:
            print(f"[INFO]   -> {fermi_ref - gap:.4f} eV above the CBM.")

    margin    = 0.10 * gap
    fmin      = float(args.fermi_min) if args.fermi_min is not None \
                else min(0.0, fermi_ref) - margin
    fmax      = float(args.fermi_max) if args.fermi_max is not None \
                else max(gap, fermi_ref) + margin
    if fmin >= fmax:
        # Deterministic given data + args (see the comment above), so
        # this raises identically on every rank -- no broadcast needed
        # to avoid the collective-boundary hang risk noted at step 1.
        raise ValueError(
            f"Fermi sweep range is empty or inverted: "
            f"--fermi-min ({fmin:.4f} eV) >= --fermi-max ({fmax:.4f} eV).")
    fermi_grid = np.linspace(fmin, fmax, N_FERMI_POINTS)
    if rank == 0:
        print(f"[INFO] Fermi sweep: [{fmin:.4f}, {fmax:.4f}] eV "
              f"({N_FERMI_POINTS} points)")

    # ------------------------------------------------------------------ #
    # 3. Dry-run: stop here
    # ------------------------------------------------------------------ #
    if args.dry_run:
        if rank == 0:
            print("[INFO] --dry-run: input validation complete.  "
                  "No calculations performed.")
            print(f"[INFO] Defects to be computed: "
                  f"{[d['name'] for d in data['defects']]}")
            print(f"[INFO] Band gap: {gap:.4f} eV  |  "
                  f"Sweep: [{fmin:.4f}, {fmax:.4f}] eV")
        return

    # ------------------------------------------------------------------ #
    # 4. Run calculations (MPI-distributed across (defect, charge_state)
    #    pairs when a real multi-rank communicator is active; returns
    #    the assembled results on rank 0 and None elsewhere).
    #
    #    run_all_defects() itself only ever raises (DefectComputationError)
    #    on rank 0, after its own internal comm.gather() has already
    #    completed for every rank -- but that raise would otherwise skip
    #    straight past the comm.Barrier() below, hanging every other
    #    rank (which returns None normally and reaches Barrier() just
    #    fine). Caught and re-raised consistently for the same reason as
    #    step 1's load_input() -- see mpi_utils.raise_consistently().
    # ------------------------------------------------------------------ #
    if rank == 0:
        print("[INFO] Computing KO corrections and formation energies ...")
    run_exc = None
    try:
        all_results = run_all_defects(
            data, fermi_ref, fermi_grid, verbose=args.verbose, comm=comm,
            **ewald_kwargs)
    except Exception as exc:
        all_results = None
        run_exc = exc
    raise_consistently(comm, run_exc)

    comm.Barrier()
    if rank != 0:
        return

    # ------------------------------------------------------------------ #
    # 5. Output  (B-3: all under out_dir) -- rank 0 only
    # ------------------------------------------------------------------ #
    out = args.out_dir
    _safe_mkdir(out)

    # Text report
    report_lines = format_report(data, all_results, fermi_ref,
                                  ewald_eta_override=args.ewald_eta)
    print("\n" + "\n".join(report_lines))
    write_text(_out(out, args.output), report_lines)

    # Per-defect tables
    for res in all_results:
        tag = res["name"].replace(" ", "_")
        write_ef_table(
            _out(out, "ef_vs_ef", f"{tag}_ef_vs_ef.txt"),
            res["name"], res["fermi_grid"],
            res["ef_by_charge"], res["envelope"])
        summary_rows = [
            dict(q=q, E_KO=res["ko"][q]["E_KO"],
                 E_real=res["ko"][q]["E_real"],
                 E_recip=res["ko"][q]["E_recip"],
                 E_self_ewald=res["ko"][q]["E_self_ewald"],
                 E_neutralize=res["ko"][q]["E_neutralize"],
                 E_PC=res["ko"][q]["E_PC"],
                 E_lat=res["ko"][q]["E_lat"],
                 E_alignment=res["ko"][q]["E_alignment"],
                 Ef_at_fermi=res["ef_at_fermi"][q])
            for q in res["charges"]
        ]
        write_summary_txt(
            _out(out, "summary", f"{tag}_summary.txt"),
            res["name"], summary_rows, fermi_ref)

    # Unified CTL summary (B-2)
    write_ctls_summary(_out(out, "ctls_summary.txt"), all_results, gap)

    # Plots
    if args.plot:
        for res in all_results:
            tag = res["name"].replace(" ", "_")
            plot_single_defect(
                data, res, fermi_ref,
                _out(out, "plots", f"{tag}_ef.png"))
        plot_comparison(
            data, all_results, fermi_ref,
            _out(out, "plots", args.comparison_plot))

    # ------------------------------------------------------------------ #
    # 6. Equilibrium Fermi level & concentrations (Feat-1, opt-in) --
    #    rank 0 only (already past the `if rank != 0: return` above), and
    #    self-contained (only reads already-gathered `data`/`all_results`,
    #    no further MPI collective calls), so no raise_consistently() is
    #    needed here.
    # ------------------------------------------------------------------ #
    if args.solve_fermi:
        if bs.get("effective_mass_e") is None or bs.get("effective_mass_h") is None:
            print("[WARNING] --solve-fermi: using the default effective "
                  "mass (1.0 m0) for at least one carrier type; equilibrium "
                  "concentrations may be off by an order of magnitude for "
                  "real materials.  Set 'effective_mass_e'/'effective_mass_h' "
                  "under 'band_structure' for accurate results.")
        # As documented (README / --fermi-min / --fermi-max help text):
        # solve_equilibrium bisects over [--fermi-min, --fermi-max] when
        # either is given explicitly, defaulting per-side to the band
        # gap edges (solve_equilibrium's own default) otherwise -- NOT
        # the wider, fermi_ref-dependent auto sweep range computed for
        # the E_f(q, E_F) table/plot above (fmin/fmax), which would
        # silently change the default --solve-fermi bracket too.
        if args.fermi_min is not None or args.fermi_max is not None:
            eq_lo = float(args.fermi_min) if args.fermi_min is not None else 0.0
            eq_hi = float(args.fermi_max) if args.fermi_max is not None else gap
            fermi_bracket = (eq_lo, eq_hi)
        else:
            fermi_bracket = None
        eq_result = solve_equilibrium(
            data, all_results, args.temperature, fermi_bracket=fermi_bracket,
            include_carriers=not args.no_carriers, tol=args.fermi_solve_tol)
        print(f"\n[INFO] Solved Fermi level (rel. VBM): "
              f"{eq_result['fermi_level_rel_vbm']:+.4f} eV")
        print(f"[INFO]   n = {eq_result['electron_conc_cm3']:.4e} cm^-3, "
              f"p = {eq_result['hole_conc_cm3']:.4e} cm^-3")
        write_equilibrium_summary(
            _out(out, "equilibrium_summary.txt"), data, eq_result)

    # ------------------------------------------------------------------ #
    # 7. Chemical potential stability region (Feat-2, opt-in) -- rank 0
    #    only, self-contained (only reads `data`; no all_results or MPI
    #    dependency), so no raise_consistently() is needed here either.
    # ------------------------------------------------------------------ #
    if args.stability_diagram:
        stab_result = stability_summary(data)
        print(f"\n[INFO] Chemical potential stability check: "
              f"{'FEASIBLE' if stab_result['feasible'] else 'INFEASIBLE'}")
        for v in stab_result["violations"]:
            print(f"[WARNING]   - {v}")
        write_stability_summary(
            _out(out, "stability_summary.txt"), data, stab_result)
        if args.plot:
            plot_stability_region(
                data, stab_result, _out(out, "plots", args.stability_plot))
