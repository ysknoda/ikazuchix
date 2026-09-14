"""Chemical potential consistency checks and formation-energy correction."""

import warnings


# ===========================================================================
# Chemical potential helpers  (A-4: consistency check added)
# ===========================================================================

def check_chemical_potentials(data):
    """
    Verify that the supplied chemical potentials satisfy the host
    stability condition and that delta_mu values are within the
    thermodynamically allowed range.

    Checks performed
    ----------------
    1. sum_i x_i * mu_i  ~  E[host] / f.u.
       (host formation enthalpy is reproduced by the given mu values)
    2. deltamu_i <= 0 for all species (upper bound: elemental reference)
    3. deltamu_i >= -|DeltaH_f| / x_i for each species (lower bound: host
       decomposition stability)

    Warnings are printed; no exception is raised so the calculation
    still proceeds.
    """
    refs      = data["elemental_references"]
    deltas    = data["chemical_potentials"]
    bulk      = data["bulk"]
    comp      = bulk["composition"]
    n_atoms   = sum(comp.values())
    total_e_bulk = float(bulk["energy"])

    # Check 0: every species referenced anywhere (bulk + all defect
    # charge-state compositions) must have both an elemental reference
    # and a chemical potential entry; a missing entry otherwise defaults
    # silently to 0.0 in the formulas below and in chem_pot_correction().
    all_species = set(comp)
    for d in data.get("defects", []):
        for cs in d.get("charge_states", []):
            all_species |= set(cs.get("composition", {}) or {})
    for sp in sorted(all_species):
        if sp not in refs:
            warnings.warn(
                f"[Chem.pot.] Species '{sp}' has no entry in "
                "'elemental_references'; it will silently default to "
                "0.0 eV/atom.  Verify your input data.",
                UserWarning, stacklevel=3)
        if sp not in deltas:
            warnings.warn(
                f"[Chem.pot.] Species '{sp}' has no entry in "
                "'chemical_potentials'; it will silently default to "
                "delta_mu = 0.0 eV.  Verify your input data.",
                UserWarning, stacklevel=3)

    # mu_i = E_ref_i + deltamu_i
    mu = {sp: refs.get(sp, 0.0) + deltas.get(sp, 0.0) for sp in comp}

    # Check 1: host stability  sum x_i mu_i == E_bulk / N
    e_per_atom_from_mu  = sum(comp[sp] * mu[sp] for sp in comp) / n_atoms
    e_per_atom_from_dft = total_e_bulk / n_atoms
    diff = abs(e_per_atom_from_mu - e_per_atom_from_dft)
    if diff > 0.05:
        warnings.warn(
            f"[Chem.pot.] Host stability check: "
            f"sum x_i*mu_i/atom = {e_per_atom_from_mu:.4f} eV  vs  "
            f"E_bulk/atom = {e_per_atom_from_dft:.4f} eV  "
            f"(diff = {diff:.4f} eV).  "
            "This may indicate inconsistent chemical potentials or "
            "a mismatch between the bulk supercell energy and the "
            "elemental references.  Verify your input data.",
            UserWarning, stacklevel=3)
    else:
        print(f"[INFO] Chem.pot. host stability check passed "
              f"(diff = {diff:.4f} eV/atom).")

    # Check 2: deltamu_i <= 0  (cannot exceed elemental reference)
    for sp, dmu in deltas.items():
        if dmu > 1e-6:
            warnings.warn(
                f"[Chem.pot.] delta_mu_{sp} = {dmu:+.4f} eV > 0.  "
                "Chemical potential exceeds the elemental reference; "
                "this is unphysical under equilibrium conditions.",
                UserWarning, stacklevel=3)

    # Check 3: formation enthalpy lower bound
    # DeltaH_f (per atom) ~ E_bulk/atom - sum x_i E_ref_i / N
    dH_f_per_atom = (total_e_bulk / n_atoms
                     - sum(comp[sp] * refs.get(sp, 0.0)
                           for sp in comp) / n_atoms)
    for sp in comp:
        x_i   = comp[sp] / n_atoms
        lb    = dH_f_per_atom / x_i if x_i > 0 else -999
        dmu_i = deltas.get(sp, 0.0)
        if dmu_i < lb - 0.01:
            warnings.warn(
                f"[Chem.pot.] delta_mu_{sp} = {dmu_i:+.4f} eV is below "
                f"the estimated lower bound {lb:+.4f} eV "
                "(host decomposition limit).  "
                "Verify chemical potential conditions.",
                UserWarning, stacklevel=3)

    # Check 4 (Feat-2, opt-in): explicit competing secondary phases.
    # Guarded entirely by 'competing_phases' being present and
    # non-empty -- data.get(...) is None/[] and this loop simply does
    # not execute otherwise, leaving Checks 0-3 above (and their exact
    # output) completely unaffected.
    for phase in (data.get("competing_phases") or []):
        phase_comp, dH_f_phase = competing_phase_bound(phase, refs)
        lhs = sum(n * deltas.get(sp, 0.0) for sp, n in phase_comp.items())
        if lhs > dH_f_phase + 1e-6:
            warnings.warn(
                f"[Chem.pot.] Competing phase "
                f"'{phase.get('formula', '?')}' is thermodynamically "
                "favored under the given chemical potentials: "
                f"sum n_i*delta_mu_i = {lhs:+.4f} eV > "
                f"dH_f(phase) = {dH_f_phase:+.4f} eV.  The host may not "
                "be stable against forming this phase; consider "
                "tightening chemical potentials.",
                UserWarning, stacklevel=3)


def competing_phase_bound(phase, elemental_references):
    """For a competing secondary phase {'formula', 'energy',
    'composition'}, compute (composition, dH_f): dH_f is the RAW
    (un-normalized -- over whatever unit cell the phase's own
    'composition'/'energy' describe) formation enthalpy relative to
    elemental_references:

        dH_f = phase['energy'] - sum_i composition[i] * E_ref_i

    The implied linear inequality constraint on chemical-potential
    deviations Δμ (used above, and by ikazuchix.stability_region) is:

        sum_i composition[i] * Δμ_i  <=  dH_f

    (violated => the competing phase is thermodynamically favored over
    the host under those chemical potentials.)
    """
    composition = phase["composition"]
    dH_f = float(phase["energy"]) - sum(
        n * elemental_references.get(sp, 0.0) for sp, n in composition.items())
    return composition, dH_f


def chem_pot_correction(data, cs_entry):
    """Return (correction [eV], delta_n dict) for one charge-state entry."""
    refs      = data["elemental_references"]
    deltas    = data["chemical_potentials"]
    def_comp  = cs_entry["composition"]
    bulk_comp = data["bulk"]["composition"]
    # sorted(), not set(): the loop below accumulates floats, and
    # floating-point addition is not associative, so a set's iteration
    # order -- which for str keys varies between processes with
    # Python's randomized string hashing (PYTHONHASHSEED) -- would make
    # `corr` differ in the last ULP from one run to the next. Verified:
    # a defect with >=2 non-zero delta_n yields two distinct bit
    # patterns across the species orderings. That silently breaks this
    # project's bit-for-bit reproducibility contract, and can be
    # amplified far beyond 1 ULP downstream, since CTL positions are
    # computed as a difference of two nearly-equal E_f intercepts.
    all_sp    = sorted(set(def_comp) | set(bulk_comp))
    delta_n   = {sp: def_comp.get(sp, 0) - bulk_comp.get(sp, 0)
                 for sp in all_sp}
    corr = 0.0
    for sp, dn in delta_n.items():
        mu    = refs.get(sp, 0.0) + deltas.get(sp, 0.0)
        corr -= dn * mu
    return corr, delta_n
