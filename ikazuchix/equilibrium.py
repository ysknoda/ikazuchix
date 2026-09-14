"""Self-consistent Fermi-level solving and equilibrium defect/carrier
concentrations (Feat-1) -- fully opt-in, engaged only via --solve-fermi;
nothing here is imported or executed on the default code path.

Physical model
--------------
Dilute-limit (non-interacting defects), non-degenerate (Boltzmann)
statistics: a mass-action law for defect concentrations, and a
parabolic-band effective density of states for band carriers. These
are the standard first-order approximations used throughout the
defect-formation-energy literature (see e.g. Freysoldt et al., Rev.
Mod. Phys. 86, 253 (2014)); they assume defect and carrier
concentrations stay well below the site/band-edge density of states,
and are not appropriate for heavily-doped or degenerate systems (where
full Fermi-Dirac statistics would be required instead).

Self-consistency equation
--------------------------
    sum_D sum_q  q * C_{D,q}(E_F)  +  p(E_F) - n(E_F)  =  0

solved for E_F via plain bisection (bisect_fermi_level): the residual
is provably strictly monotonic in E_F (see charge_neutrality_residual),
so bisection alone is guaranteed to converge to the unique root -- no
need for Brent's method or an added scipy dependency.
"""

import math

from ikazuchix.constants import (
    BOLTZMANN_EV_PER_K,
    BOLTZMANN_J_PER_K,
    ELECTRON_MASS_KG,
    PLANCK_CONSTANT_J_S,
)
from ikazuchix.geometry import build_lattice_matrix, lattice_volume


def effective_dos(m_eff_ratio, temperature_k):
    """Parabolic-band effective density of states [cm^-3]:

        N = 2 * (2*pi*m_eff*k_B*T / h^2)^(3/2)     [SI, then -> cm^-3]

    m_eff_ratio: effective mass as a multiple of the free-electron mass
    (dimensionless). Standard textbook result (e.g. Sze & Ng, "Physics
    of Semiconductor Devices"); derived here from CODATA/SI constants
    rather than a hardcoded reference density at one specific
    temperature, so it stays correct for any temperature_k.
    """
    if m_eff_ratio <= 0:
        raise ValueError(
            f"Effective mass ratio must be positive (got {m_eff_ratio}).")
    if temperature_k <= 0:
        raise ValueError(f"Temperature must be positive (got {temperature_k}).")
    m_eff_kg = m_eff_ratio * ELECTRON_MASS_KG
    n_per_m3 = 2.0 * (2.0 * math.pi * m_eff_kg * BOLTZMANN_J_PER_K * temperature_k
                       / PLANCK_CONSTANT_J_S ** 2) ** 1.5
    return n_per_m3 * 1e-6  # m^-3 -> cm^-3


def carrier_concentrations(vbm, cbm, fermi_abs, temperature_k,
                            m_eff_e=1.0, m_eff_h=1.0):
    """Non-degenerate (Boltzmann) electron/hole concentrations [cm^-3]
    at absolute Fermi level fermi_abs, given absolute VBM/CBM [eV]:

        n = N_c * exp(-(E_c - E_F)/kT)
        p = N_v * exp(-(E_F - E_v)/kT)

    Returns (n, p).
    """
    kT = BOLTZMANN_EV_PER_K * temperature_k
    n = effective_dos(m_eff_e, temperature_k) * math.exp(-(cbm - fermi_abs) / kT)
    p = effective_dos(m_eff_h, temperature_k) * math.exp(-(fermi_abs - vbm) / kT)
    return n, p


def defect_site_density(defect_block, bulk_composition, supercell_volume_ang3):
    """Auto-derive a defect's host-site density [cm^-3] from the bulk
    composition and supercell volume: valid when 'removed_atoms' names
    exactly one species (a simple vacancy or single-species
    substitution), in which case every atom of that species in the
    bulk supercell is an equivalent candidate site.

    Returns None when auto-derivation is not possible (an interstitial
    with no removed_atoms, or a defect removing more than one species)
    -- callers should fall back to an explicit 'site_density' field in
    that case; see resolve_site_density().
    """
    removed = defect_block.get("removed_atoms") or {}
    species = [sp for sp, count in removed.items() if count]
    if len(species) != 1:
        return None
    n_sites_per_cell = bulk_composition.get(species[0], 0)
    if n_sites_per_cell <= 0:
        return None
    ang3_per_cm3 = 1e24  # 1 cm^3 = 1e24 Ang^3
    return n_sites_per_cell * ang3_per_cm3 / supercell_volume_ang3


def resolve_site_density(defect_block, bulk_composition, supercell_volume_ang3):
    """An explicit 'site_density' YAML field takes precedence; otherwise
    fall back to defect_site_density(). Raises ValueError, naming the
    defect, when neither is available -- callers should only call this
    for a defect that actually has a non-zero charge state (a q=0-only
    defect never appears in the neutrality sum, so never needs one)."""
    explicit = defect_block.get("site_density")
    if explicit is not None:
        return float(explicit)
    derived = defect_site_density(
        defect_block, bulk_composition, supercell_volume_ang3)
    if derived is not None:
        return derived
    raise ValueError(
        f"Cannot determine a site density for defect "
        f"'{defect_block.get('name')}': 'removed_atoms' does not name "
        "exactly one species (interstitial, or multi-species defect), "
        "and no explicit 'site_density' [cm^-3] field is given.  "
        "Required for --solve-fermi.")


def defect_concentration(ef_intercept_ev, q, fermi_abs, vbm, temperature_k,
                          n_sites_cm3, degeneracy=1):
    """Dilute-limit mass-action-law equilibrium concentration [cm^-3]
    of one (defect, charge q) species at absolute Fermi level fermi_abs:

        C_{D,q} = N_sites * g_q * exp(-E_f(q, E_F) / kT)

    ef_intercept_ev is E_f(q, E_F=VBM) -- see
    runner.assemble_defect_result's ef_intercepts. formation_energy()
    is exactly affine in the Fermi level with slope q, so
    E_f(q, fermi_abs) = ef_intercept_ev + q*(fermi_abs - vbm) exactly,
    without needing to recompute the full KO correction here.
    """
    kT = BOLTZMANN_EV_PER_K * temperature_k
    ef = ef_intercept_ev + q * (fermi_abs - vbm)
    return n_sites_cm3 * degeneracy * math.exp(-ef / kT)


def charge_neutrality_residual(fermi_rel, defect_ctx_list, vbm, cbm, temperature_k,
                                include_carriers=True, m_eff_e=1.0, m_eff_h=1.0):
    """Net charge density [cm^-3] at Fermi level fermi_rel (eV, rel. VBM):

        residual(E_F) = sum_D sum_q q*C_{D,q}(E_F)  +  p(E_F) - n(E_F)

    Strictly monotonically decreasing in E_F: each charged-defect term
    has slope d(q*C_q)/dE_F = -q^2/kT * C_q <= 0, and the carrier term
    p - n is strictly decreasing too (p falls, n rises as E_F rises) --
    this is what justifies solving the equation with plain bisection
    (bisect_fermi_level), with no need for Brent's method or scipy.

    defect_ctx_list: list of {'q', 'ef_intercept', 'n_sites_cm3',
    'degeneracy'} dicts, one per (defect, non-zero charge) species
    (q=0 states never contribute and should be excluded by the caller).
    """
    fermi_abs = vbm + fermi_rel
    residual = 0.0
    for ctx in defect_ctx_list:
        c = defect_concentration(
            ctx["ef_intercept"], ctx["q"], fermi_abs, vbm, temperature_k,
            ctx["n_sites_cm3"], ctx.get("degeneracy", 1))
        residual += ctx["q"] * c
    if include_carriers:
        n, p = carrier_concentrations(
            vbm, cbm, fermi_abs, temperature_k, m_eff_e, m_eff_h)
        residual += p - n
    return residual


def bisect_fermi_level(residual_fn, e_lo, e_hi, tol=1e-8, max_iter=200):
    """Pure-bisection root find of residual_fn (expected strictly
    monotonic -- see charge_neutrality_residual) over [e_lo, e_hi].

    Raises ValueError if e_hi <= e_lo (an inverted/empty bracket would
    otherwise satisfy the loop's own (hi - lo) < tol convergence check
    on its very first iteration -- since hi - lo is already negative,
    hence "less than" any positive tol -- and silently return a
    meaningless value instead of failing loudly), or if residual_fn
    does not change sign across the bracket (no root is guaranteed to
    lie inside it).
    """
    if e_hi <= e_lo:
        raise ValueError(
            f"Bisection bracket is empty or inverted: e_lo={e_lo:.4f} eV "
            f">= e_hi={e_hi:.4f} eV.")
    f_lo = residual_fn(e_lo)
    f_hi = residual_fn(e_hi)
    if f_lo == 0.0:
        return e_lo
    if f_hi == 0.0:
        return e_hi
    if (f_lo > 0) == (f_hi > 0):
        raise ValueError(
            "charge_neutrality_residual does not change sign over "
            f"[{e_lo:.4f}, {e_hi:.4f}] eV (residual = {f_lo:.3e} and "
            f"{f_hi:.3e} cm^-3); no root is bracketed.  Widen the "
            "search range (--fermi-min/--fermi-max) or check site "
            "densities and degeneracies.")

    lo, hi = e_lo, e_hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = residual_fn(mid)
        if f_mid == 0.0 or (hi - lo) < tol:
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5 * (lo + hi)


def solve_equilibrium(data, all_results, temperature_k, fermi_bracket=None,
                       include_carriers=True, tol=1e-8):
    """Solve the self-consistent Fermi level from already-computed
    per-defect results, and report equilibrium carrier and defect
    concentrations at that Fermi level.

    Parameters
    ----------
    data        : the loaded+validated input dict (io_utils.load_input
                  output) -- used for band_structure (vbm/cbm/effective
                  masses), bulk composition, and each defect's
                  removed_atoms/site_density/degeneracy/supercell.
    all_results : list of per-defect result dicts (runner.run_all_defects
                  / run_one_defect output), same order as data['defects'];
                  each must include 'ef_intercepts' (added to
                  assemble_defect_result's return value for this feature).
    fermi_bracket : (e_lo, e_hi) in eV relative to VBM to bisect over.
                  Defaults to (0.0, cbm - vbm), i.e. within the gap.

    Returns a dict: fermi_level_rel_vbm, fermi_level_abs,
    electron_conc_cm3, hole_conc_cm3, defect_concentrations (list of
    {name, q, concentration_cm3}), temperature_k, include_carriers.
    """
    bs  = data["band_structure"]
    vbm, cbm = bs["vbm"], bs["cbm"]
    m_eff_e = bs.get("effective_mass_e", 1.0)
    m_eff_h = bs.get("effective_mass_h", 1.0)
    bulk_composition = data["bulk"]["composition"]

    defect_ctx_list = []
    for res, defect_block in zip(all_results, data["defects"]):
        nonzero = {q: ei for q, ei in res["ef_intercepts"].items() if q != 0}
        if not nonzero:
            continue
        sc_dict = defect_block.get("supercell", data.get("supercell"))
        volume = lattice_volume(build_lattice_matrix(sc_dict))
        n_sites = resolve_site_density(defect_block, bulk_composition, volume)
        degeneracy_by_q = {
            int(cs["charge"]): cs.get("degeneracy", 1)
            for cs in defect_block["charge_states"]
        }
        for q, ef_intercept in nonzero.items():
            defect_ctx_list.append(dict(
                name=res["name"], q=q, ef_intercept=ef_intercept,
                n_sites_cm3=n_sites, degeneracy=degeneracy_by_q.get(q, 1)))

    e_lo, e_hi = fermi_bracket if fermi_bracket is not None else (0.0, cbm - vbm)

    def residual_fn(fermi_rel):
        return charge_neutrality_residual(
            fermi_rel, defect_ctx_list, vbm, cbm, temperature_k,
            include_carriers=include_carriers, m_eff_e=m_eff_e, m_eff_h=m_eff_h)

    fermi_rel = bisect_fermi_level(residual_fn, e_lo, e_hi, tol=tol)
    fermi_abs = vbm + fermi_rel
    n, p = carrier_concentrations(vbm, cbm, fermi_abs, temperature_k, m_eff_e, m_eff_h)

    concentrations = [
        dict(name=ctx["name"], q=ctx["q"],
             concentration_cm3=defect_concentration(
                 ctx["ef_intercept"], ctx["q"], fermi_abs, vbm, temperature_k,
                 ctx["n_sites_cm3"], ctx["degeneracy"]))
        for ctx in defect_ctx_list
    ]

    return dict(
        fermi_level_rel_vbm=fermi_rel, fermi_level_abs=fermi_abs,
        electron_conc_cm3=n, hole_conc_cm3=p,
        defect_concentrations=concentrations,
        temperature_k=temperature_k, include_carriers=include_carriers)
