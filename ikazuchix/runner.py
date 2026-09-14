"""Per-defect and all-defects calculation orchestration (MPI-aware).

Parallelization granularity
----------------------------
Each (defect, charge_state) pair's Kumagai-Oba correction + formation
energy curve is fully independent of every other pair, so this is the
natural unit of work to distribute across MPI ranks. Charge-state
transition level (CTL) / envelope analysis needs *all* charge states of
a given defect together, so it can only run after gathering.

run_all_defects() always goes through the same flatten -> distribute
-> compute -> gather -> assemble pipeline regardless of whether a real
multi-rank MPI.COMM_WORLD is in play; with the single-process NullComm
fallback (ikazuchix/mpi_utils.py) it transparently reduces to today's serial
behaviour (one rank owns 100% of the work list, gather is a no-op).
run_one_defect() is kept as a separate, always-serial convenience
wrapper (used by tests and single-defect callers) built from the same
two primitives, rather than a thin wrapper around run_all_defects().
"""

import numpy as np

from ikazuchix.chem_potentials import chem_pot_correction
from ikazuchix.ewald import KumagaiObaCorrectionCalculator, build_ewald_shell_cache
from ikazuchix.formation_energy import analytic_ctls, classify_ctls, formation_energy, stable_envelope
from ikazuchix.geometry import build_dielectric_tensor, build_lattice_matrix
from ikazuchix.mpi_utils import capture_warnings_as_picklable, get_comm, replay_warning


class DefectComputationError(RuntimeError):
    """One or more (defect, charge_state) work items failed while being
    computed under run_all_defects().  Raised only on rank 0, only after
    every rank has already reached the comm.gather() collective (see
    run_all_defects docstring) -- so raising this can never leave another
    rank blocked waiting on a collective call that will now never come.
    """


def _resolve_defect_geometry(defect_block, data):
    sc_dict = defect_block.get("supercell", data.get("supercell"))
    if sc_dict is None:
        raise KeyError(
            f"Defect '{defect_block['name']}' has no 'supercell' block "
            "and no top-level 'supercell' is defined.")
    lattice = build_lattice_matrix(sc_dict)
    epsilon = build_dielectric_tensor(data["dielectric"])
    def_pos = sc_dict["defect_position"]
    return lattice, epsilon, def_pos


# ===========================================================================
# Single (defect, charge_state) work item  -- safe to run on any MPI rank
# ===========================================================================

def compute_charge_state_result(defect_block, cs, data, fermi_ref, fermi_grid,
                                 lattice, epsilon, def_pos, verbose=False,
                                 ewald_kwargs=None, shell_cache=None):
    """
    Compute the KO correction and formation-energy curve for a single
    (defect, charge_state) pair.

    Pure/stateless given its arguments (no shared mutable state), so it
    is safe to call independently -- and in any order -- on any MPI
    rank. Console output and warnings are *captured* rather than
    printed/raised directly, since a value computed on a non-root rank
    would otherwise never reach the user; the caller (run_one_defect or
    run_all_defects) replays them, in the original per-defect,
    per-charge-state order, only on rank 0.

    `shell_cache` (Perf-1, optional): an EwaldShellCache shared by every
    charge state of this defect, so the (expensive, q-independent) raw
    Ewald shell sums are computed once per defect rather than once per
    charge state. None (default) preserves the original per-call
    behaviour exactly.

    Returns a dict with keys: q, ko, ef_arr, ef_ref, chem_corr, delta_n,
    console_lines, captured_warnings.
    """
    ewald_kwargs = ewald_kwargs or {}
    defect_name = defect_block["name"]
    q       = int(cs["charge"])
    delta_v = float(cs.get("potential_alignment", {}).get("delta_v", 0.0))

    console_lines = []
    if delta_v == 0.0 and q != 0:
        console_lines.append(
            f"[WARNING] {defect_name} q={q:+d}: delta_v=0.0; "
            "KO alignment term is zero.")

    def _build_and_compute():
        ko_calc = KumagaiObaCorrectionCalculator(
            lattice=lattice, epsilon=epsilon, charge=q,
            defect_frac=def_pos, delta_v=delta_v, verbose=verbose,
            shell_cache=shell_cache, **ewald_kwargs)
        return ko_calc.compute()  # cached after first call, on this instance

    ko, captured_warnings = capture_warnings_as_picklable(_build_and_compute)

    chem_corr, delta_n = chem_pot_correction(data, cs)
    E_def  = float(cs["energy"])
    E_bulk = float(data["bulk"]["energy"])
    vbm    = data["band_structure"]["vbm"]

    # D-2: vectorised -- formation_energy accepts ndarray for fermi_rel
    ef_arr = formation_energy(E_def, E_bulk, chem_corr, q, vbm, fermi_grid, ko["E_KO"])
    ef_ref = formation_energy(E_def, E_bulk, chem_corr, q, vbm, fermi_ref, ko["E_KO"])

    console_lines.append(
        f"[INFO]   q={q:+d}: E_KO={ko['E_KO']:+.4f} eV, "
        f"E_f(E_F={fermi_ref:.3f} eV)={ef_ref:+.4f} eV")

    return dict(
        q=q, ko=ko, ef_arr=ef_arr, ef_ref=ef_ref,
        chem_corr=chem_corr, delta_n=delta_n,
        console_lines=console_lines, captured_warnings=captured_warnings)


def _emit_charge_state_result(r):
    """Print/warn a compute_charge_state_result() payload (rank 0 only)."""
    for line in r["console_lines"]:
        print(line)
    for message, category_name in r["captured_warnings"]:
        replay_warning(message, category_name, stacklevel=3)


# ===========================================================================
# Per-defect assembly  -- needs every charge state of the defect together
# ===========================================================================

def assemble_defect_result(defect_block, charge_results, fermi_grid, gap):
    """
    Combine the per-charge-state results of ONE defect (already computed
    -- and, under MPI, already gathered to this rank) into the same
    result shape historically returned by run_one_defect(): CTL /
    envelope analysis needs all charge states of a defect together, so
    it can only happen at this assembly stage.
    """
    charges, ko_results, ef_by_charge = [], {}, {}
    ef_at_fermi, chem_corrs, delta_ns = {}, {}, {}

    for r in charge_results:
        q = r["q"]
        charges.append(q)
        ko_results[q]   = r["ko"]
        ef_by_charge[q] = r["ef_arr"]
        ef_at_fermi[q]  = r["ef_ref"]
        chem_corrs[q]   = r["chem_corr"]
        delta_ns[q]     = r["delta_n"]

    charges.sort()
    # D-3: analytic CTLs from y-intercepts (E_f at E_F = VBM = 0).
    # Since ef_by_charge[q] = a_q + q*fermi_grid exactly (formation_energy
    # is affine in fermi_rel), a_q can be recovered from any grid point;
    # the one nearest E_F = 0 is used for numerical cleanliness.
    idx_zero = int(np.argmin(np.abs(fermi_grid)))
    ef_intercepts = {
        q: ef_by_charge[q][idx_zero] - q * fermi_grid[idx_zero]
        for q in charges
    }
    all_ctls               = analytic_ctls(charges, ef_intercepts)
    stable_ctls, meta_ctls = classify_ctls(
        all_ctls, charges, ef_by_charge, fermi_grid, gap)
    envelope = stable_envelope(charges, ef_by_charge)

    return dict(
        name=defect_block["name"],
        defect_type=defect_block.get("defect_type", ""),
        charges=charges, ko=ko_results,
        ef_by_charge=ef_by_charge, ef_at_fermi=ef_at_fermi,
        fermi_grid=fermi_grid, chem_pot_corr=chem_corrs,
        delta_n=delta_ns,
        ctls=all_ctls, stable_ctls=stable_ctls, meta_ctls=meta_ctls,
        envelope=envelope,
        # (Feat-1) y-intercepts already computed above; exposed so
        # equilibrium.solve_equilibrium() can reconstruct E_f(q, E_F)
        # at any Fermi level without recomputing the KO correction.
        ef_intercepts=ef_intercepts)


# ===========================================================================
# Serial single-defect convenience wrapper (tests, library use)
# ===========================================================================

def run_one_defect(defect_block, data, fermi_ref, fermi_grid, verbose=False,
                    **ewald_kwargs):
    """
    Compute KO corrections and formation energies for a single defect
    type, always serially on the current process (no MPI distribution
    -- see run_all_defects() for the MPI-aware, multi-defect pipeline).

    Returns the same dict shape as assemble_defect_result().
    """
    bs  = data["band_structure"]
    gap = bs["cbm"] - bs["vbm"]
    lattice, epsilon, def_pos = _resolve_defect_geometry(defect_block, data)
    shell_cache = build_ewald_shell_cache(
        lattice, epsilon, ewald_kwargs.get("ewald_param"))

    defect_name = defect_block["name"]
    print(f"\n[INFO] ---- Defect: {defect_name} ----")

    charge_results = []
    for cs in defect_block["charge_states"]:
        r = compute_charge_state_result(
            defect_block, cs, data, fermi_ref, fermi_grid,
            lattice, epsilon, def_pos, verbose=verbose,
            ewald_kwargs=ewald_kwargs, shell_cache=shell_cache)
        _emit_charge_state_result(r)
        charge_results.append(r)

    return assemble_defect_result(defect_block, charge_results, fermi_grid, gap)


# ===========================================================================
# MPI-aware all-defects pipeline
# ===========================================================================

def run_all_defects(data, fermi_ref, fermi_grid, verbose=False, comm=None,
                     **ewald_kwargs):
    """
    Compute every (defect, charge_state) pair across data["defects"],
    distributing the work list across MPI ranks (round-robin) when a
    real multi-rank communicator is supplied or discoverable, gathering
    results to rank 0, and running CTL/envelope assembly + all console
    output only on rank 0.

    Returns the list of per-defect result dicts on rank 0, and None on
    every other rank (only rank 0 should perform any further output).

    Raises
    ------
    DefectComputationError
        On rank 0, if any (defect, charge_state) work item -- on any
        rank -- raised an exception. Each work item's computation is
        individually try/excepted (below) so that every rank still
        reaches the comm.gather() collective regardless of failures
        elsewhere; only after gathering does rank 0 inspect the results
        for errors and raise once, with every failure's details
        aggregated into a single message. This deliberately does *not*
        wrap the comm.gather() call itself -- every rank must always
        reach that same collective call, or ranks that did not fail
        would hang forever waiting for one that did.
    """
    if comm is None:
        comm = get_comm()
    rank = comm.Get_rank()
    size = comm.Get_size()

    defect_geoms = [_resolve_defect_geometry(d, data) for d in data["defects"]]
    # (Perf-1) One EwaldShellCache per defect, shared by every charge
    # state of that defect -- built here (deterministic given `data` +
    # `ewald_kwargs`, identical on every rank) alongside defect_geoms,
    # for the same reason defect_geoms itself lives outside the
    # try/except work-item loop below: a failure here is guaranteed to
    # happen identically on every rank, so it can never leave one rank
    # stuck at comm.gather() waiting for a peer that failed elsewhere.
    shell_caches = [
        build_ewald_shell_cache(lattice, epsilon, ewald_kwargs.get("ewald_param"))
        for lattice, epsilon, _ in defect_geoms
    ]

    work_items = [
        (d_idx, cs_idx)
        for d_idx, d in enumerate(data["defects"])
        for cs_idx in range(len(d["charge_states"]))
    ]
    my_items = work_items[rank::size]

    local_results = {}
    for d_idx, cs_idx in my_items:
        defect_block = data["defects"][d_idx]
        cs = defect_block["charge_states"][cs_idx]
        lattice, epsilon, def_pos = defect_geoms[d_idx]
        try:
            local_results[(d_idx, cs_idx)] = compute_charge_state_result(
                defect_block, cs, data, fermi_ref, fermi_grid,
                lattice, epsilon, def_pos, verbose=verbose,
                ewald_kwargs=ewald_kwargs, shell_cache=shell_caches[d_idx])
        except Exception as exc:
            local_results[(d_idx, cs_idx)] = dict(
                error=True, error_type=type(exc).__name__,
                error_message=str(exc), defect_name=defect_block.get("name"),
                cs_index=cs_idx, rank=rank)

    gathered = comm.gather(local_results, root=0)

    if rank != 0:
        return None

    merged = {}
    for rank_results in gathered:
        merged.update(rank_results)

    errors = [r for r in merged.values() if r.get("error")]
    if errors:
        detail = "\n".join(
            f"  - defect='{e['defect_name']}' charge_state #{e['cs_index']} "
            f"(computed on rank {e['rank']}): {e['error_type']}: "
            f"{e['error_message']}"
            for e in errors)
        raise DefectComputationError(
            f"{len(errors)} of {len(merged)} (defect, charge_state) work "
            f"item(s) failed:\n{detail}")

    bs  = data["band_structure"]
    gap = bs["cbm"] - bs["vbm"]

    all_results = []
    for d_idx, defect_block in enumerate(data["defects"]):
        print(f"\n[INFO] ---- Defect: {defect_block['name']} ----")
        charge_results = []
        for cs_idx in range(len(defect_block["charge_states"])):
            r = merged[(d_idx, cs_idx)]
            _emit_charge_state_result(r)
            charge_results.append(r)
        all_results.append(
            assemble_defect_result(defect_block, charge_results, fermi_grid, gap))

    return all_results
