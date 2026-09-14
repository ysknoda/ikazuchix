"""Formation energy formula and charge-state transition level (CTL) analysis."""

import numpy as np


# ===========================================================================
# Formation energy
# ===========================================================================

def formation_energy(E_defect, E_bulk, chem_corr, q, E_VBM, fermi_rel, E_KO):
    """
    E_f(q, E_F) = (E_defect - E_bulk) + chem_corr + q*(E_VBM + E_F) + E_KO

    fermi_rel may be a scalar (float) or a numpy array.
    When an array is supplied the return value is also an array, enabling
    vectorised sweep calculations (D-2).
    """
    return (E_defect - E_bulk) + chem_corr + q * (E_VBM + fermi_rel) + E_KO


# ===========================================================================
# CTL detection and classification  (A-3, D-3)
# ===========================================================================

def analytic_ctls(charges, ef_intercepts):
    """
    Compute charge-state transition levels (CTLs) analytically (D-3).

    Each E_f curve is a straight line:
        E_f(q, E_F) = a_q + q * E_F
    where a_q = (E_defect - E_bulk) + chem_corr + q*VBM + E_KO is the
    y-intercept at E_F = 0 (VBM).

    The crossing of lines q1 and q2:
        a_q1 + q1 * E_F* = a_q2 + q2 * E_F*
        => E_F* = (a_q2 - a_q1) / (q1 - q2)   [exact, no grid dependence]
        => E_f* = a_q1 + q1 * E_F*

    Parameters
    ----------
    charges       : sorted list of charge integers
    ef_intercepts : dict  {q: float}  --  a_q = E_f(q, E_F=0)

    Returns
    -------
    list of dicts with keys q1, q2, E_F_CTL, E_f_CTL
    """
    ctls = []
    for i, q1 in enumerate(charges):
        for q2 in charges[i + 1:]:
            dq = q1 - q2
            if dq == 0:
                continue          # same slope: parallel lines, no crossing
            ef_star = (ef_intercepts[q2] - ef_intercepts[q1]) / dq
            eform   = ef_intercepts[q1] + q1 * ef_star
            ctls.append(dict(q1=q1, q2=q2, E_F_CTL=ef_star, E_f_CTL=eform))
    return ctls


def find_ctls(charges, ef_by_charge, fermi_grid):
    """Find all CTLs by linear interpolation of sign changes.

    This is the brute-force, grid-based CTL finder that predates the
    exact ``analytic_ctls`` solver above. It is retained purely as an
    independent cross-check for ``analytic_ctls`` in the test suite
    (see ``tests/test_formation_energy.py``) and is not called anywhere
    in the production compute path (``ikazuchix/runner.py``).
    """
    ctls = []
    for i, q1 in enumerate(charges):
        for q2 in charges[i+1:]:
            diff  = ef_by_charge[q1] - ef_by_charge[q2]
            signs = np.where(np.diff(np.sign(diff)))[0]
            for idx in signs:
                x0, x1 = fermi_grid[idx], fermi_grid[idx+1]
                y0, y1 = diff[idx], diff[idx+1]
                if abs(y1 - y0) < 1e-12:
                    continue
                ef_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
                t        = (ef_cross - x0) / (x1 - x0)
                eform    = (ef_by_charge[q1][idx]
                            + t * (ef_by_charge[q1][idx+1]
                                   - ef_by_charge[q1][idx]))
                ctls.append(dict(q1=q1, q2=q2,
                                 E_F_CTL=ef_cross, E_f_CTL=eform))
    return ctls


def classify_ctls(ctls, charges, ef_by_charge, fermi_grid, gap):
    """
    Split CTLs into stable (envelope) and meta-stable lists.

    A CTL is classified as 'stable' (on the envelope) when:
      1. It lies within the band gap: 0 <= E_F_CTL <= gap
      2. At E_F_CTL the formation energy of both q1 and q2 equals the
         envelope minimum to within a small tolerance (1 meV).
    All other intersections are meta-stable.

    The envelope value at E_F_CTL is evaluated *exactly*, not by
    interpolating the sampled envelope array. Each E_f curve is exactly
    affine in the Fermi level with known slope q -- formation_energy()
    is E_f(q, E_F) = a_q + q*E_F -- so
        E_f(q, E_F_CTL) = ef_by_charge[q][0] + q*(E_F_CTL - fermi_grid[0])
    holds to machine precision for any E_F_CTL, and the envelope is
    just the minimum of those over q.

    Sampling the envelope on the grid and interpolating (as this did
    previously) is grid-resolution-dependent and wrong whenever a
    second envelope kink falls within the same grid cell as the CTL:
    the interpolation then straddles two different linear branches and
    reads an envelope value that lies on neither. Verified against a
    grid-free ground truth over random 4-charge-state defects: the old
    interpolation misclassified ~1 CTL in 5000, e.g. reporting a CTL
    3.2 meV above the envelope (comfortably meta-stable at the 1 meV
    tolerance) as on-envelope, because the nearest competing kink sat
    1.6 meV away with a 9.4 meV grid spacing.
    """
    stable, meta = [], []

    for ctl in ctls:
        ef = ctl["E_F_CTL"]
        if not (0.0 <= ef <= gap):
            meta.append(ctl)
            continue
        env = min(ef_by_charge[q][0] + q * (ef - fermi_grid[0])
                  for q in charges)
        if abs(ctl["E_f_CTL"] - env) < 1e-3:
            stable.append(ctl)
        else:
            meta.append(ctl)

    return stable, meta


def stable_envelope(charges, ef_by_charge):
    return np.min(np.vstack([ef_by_charge[q] for q in charges]), axis=0)
