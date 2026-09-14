"""Numba-JIT-compiled Ewald real-space and reciprocal-space shell sums.

These are module-level, numeric-only-signature functions -- a hard
requirement of numba.njit, which cannot compile bound instance methods
that close over an arbitrary ``self``. Each kernel returns the RAW
(unscaled) shell sum plus the raw outermost-shell contribution; the
physical prefactor (0.5*q^2*COULOMB_PREFACTOR, etc.) is applied exactly
once by the caller in ikazuchix/ewald.py, in the same order as the original
pure-Python implementation, so results are bit-for-bit identical
whether or not numba is active (verified empirically against the
original np.linalg.norm / G @ eps_inv @ G expressions).

Do not add fastmath=True or parallel=True/prange to these kernels:
either would permit floating-point reassociation and change rounding,
breaking that bit-for-bit guarantee.
"""

import math

from ikazuchix.numba_shim import njit


@njit(cache=True)
def ewald_real_shell_sum(n, L, eta, eps_avg):
    """Real-space Ewald sum (KO 2014, Eq. 13), summed over shells |n_i|<=n.

    Parameters
    ----------
    n       : int            -- shell cutoff (inclusive)
    L       : float64[3,3]   -- lattice matrix, rows = lattice vectors
    eta     : float64        -- Ewald splitting parameter [Ang^-1]
    eps_avg : float64        -- scalar-averaged dielectric constant

    Returns
    -------
    (E_raw, E_outer_raw) : sum_{R!=0} erfc(eta|R|/sqrt(eps_avg))/(eps_avg|R|)
    and the (absolute-value-summed) contribution from the outermost
    shell alone (max(|n1|,|n2|,|n3|) == n), both unscaled by q/COULOMB_PREFACTOR.
    """
    E = 0.0
    E_outer = 0.0
    for n1 in range(-n, n + 1):
        for n2 in range(-n, n + 1):
            for n3 in range(-n, n + 1):
                if n1 == 0 and n2 == 0 and n3 == 0:
                    continue
                Rx = n1 * L[0, 0] + n2 * L[1, 0] + n3 * L[2, 0]
                Ry = n1 * L[0, 1] + n2 * L[1, 1] + n3 * L[2, 1]
                Rz = n1 * L[0, 2] + n2 * L[1, 2] + n3 * L[2, 2]
                r = math.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
                term = math.erfc(eta * r / math.sqrt(eps_avg)) / (eps_avg * r)
                E += term
                if max(abs(n1), abs(n2), abs(n3)) == n:
                    E_outer += abs(term)
    return E, E_outer


@njit(cache=True)
def ewald_recip_shell_sum(n, G_mat, eps_inv, eta):
    """Reciprocal-space Ewald sum (KO 2014, Eq. 14), summed over shells |m_i|<=n.

    Parameters
    ----------
    n       : int            -- shell cutoff (inclusive)
    G_mat   : float64[3,3]   -- reciprocal lattice matrix, rows = G-vectors
    eps_inv : float64[3,3]   -- inverse dielectric tensor
    eta     : float64        -- Ewald splitting parameter [Ang^-1]

    Returns
    -------
    (E_raw, E_outer_raw) : sum_{G!=0} exp(-GEG/(4eta^2))/GEG  and the
    outermost-shell-only contribution, both unscaled by the
    q^2*COULOMB_PREFACTOR*4*pi/(V*sqrt(det eps)) prefactor.
    """
    E = 0.0
    E_outer = 0.0
    for m1 in range(-n, n + 1):
        for m2 in range(-n, n + 1):
            for m3 in range(-n, n + 1):
                if m1 == 0 and m2 == 0 and m3 == 0:
                    continue
                Gx = m1 * G_mat[0, 0] + m2 * G_mat[1, 0] + m3 * G_mat[2, 0]
                Gy = m1 * G_mat[0, 1] + m2 * G_mat[1, 1] + m3 * G_mat[2, 1]
                Gz = m1 * G_mat[0, 2] + m2 * G_mat[1, 2] + m3 * G_mat[2, 2]
                # G @ eps_inv @ G, computed as (G @ eps_inv) @ G to match
                # the original numpy expression's summation order exactly.
                t0 = Gx * eps_inv[0, 0] + Gy * eps_inv[1, 0] + Gz * eps_inv[2, 0]
                t1 = Gx * eps_inv[0, 1] + Gy * eps_inv[1, 1] + Gz * eps_inv[2, 1]
                t2 = Gx * eps_inv[0, 2] + Gy * eps_inv[1, 2] + Gz * eps_inv[2, 2]
                geg = t0 * Gx + t1 * Gy + t2 * Gz
                if geg < 1e-20:
                    continue
                term = math.exp(-geg / (4.0 * eta ** 2)) / geg
                E += term
                if max(abs(m1), abs(m2), abs(m3)) == n:
                    E_outer += abs(term)
    return E, E_outer


# ===========================================================================
# Perf-2: outer-shell-only ("ring") kernels for incremental adaptive growth
#
# ewald_*_shell_sum() above always re-walks the full (2n+1)^3 cube from
# scratch, so growing the cutoff by one shell at a time (--adaptive-ewald)
# repeats all the inner, already-converged points every single step. The
# two kernels below instead enumerate ONLY the outermost shell
# max(|n1|,|n2|,|n3|) == n directly, in O(n^2) rather than O(n^3), as
# three disjoint face groups (n1=+-n with n2/n3 free; n2=+-n with n1
# restricted to [-(n-1),n-1] to skip the edges the first group already
# covered; n3=+-n with both n1 and n2 so restricted) -- together these
# cover exactly that shell's points once each, with no gaps or overlap.
#
# Summing ring_sum(0) + ring_sum(1) + ... + ring_sum(n) therefore visits
# the identical point set as shell_sum(n, ...), just grouped and ordered
# differently -- mathematically the same total, but in general NOT
# bit-identical (floating-point addition is not associative). This is
# fine: only --adaptive-ewald (opt-in, never part of the bit-for-bit
# golden contract -- see ikazuchix/ewald.py's EwaldShellCache) uses these.
# ===========================================================================

@njit(cache=True)
def ewald_real_shell_ring_sum(n, L, eta, eps_avg):
    """Real-space contribution from ONLY shell max(|n1|,|n2|,|n3|) == n.

    Returns (E_ring, E_outer_ring): the signed sum of that shell's terms,
    and the same shell's outermost-contribution indicator (both equal
    here, since every real-space term is non-negative -- kept as two
    return values so callers can treat this exactly like
    ewald_real_shell_sum's (E, E_outer) for one shell at a time).
    Shell 0 is just the origin, which is always excluded: returns (0, 0).
    """
    if n == 0:
        return 0.0, 0.0
    E = 0.0
    E_outer = 0.0
    for n1 in (-n, n):
        for n2 in range(-n, n + 1):
            for n3 in range(-n, n + 1):
                Rx = n1 * L[0, 0] + n2 * L[1, 0] + n3 * L[2, 0]
                Ry = n1 * L[0, 1] + n2 * L[1, 1] + n3 * L[2, 1]
                Rz = n1 * L[0, 2] + n2 * L[1, 2] + n3 * L[2, 2]
                r = math.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
                term = math.erfc(eta * r / math.sqrt(eps_avg)) / (eps_avg * r)
                E += term
                E_outer += abs(term)
    for n2 in (-n, n):
        for n1 in range(-(n - 1), n):
            for n3 in range(-n, n + 1):
                Rx = n1 * L[0, 0] + n2 * L[1, 0] + n3 * L[2, 0]
                Ry = n1 * L[0, 1] + n2 * L[1, 1] + n3 * L[2, 1]
                Rz = n1 * L[0, 2] + n2 * L[1, 2] + n3 * L[2, 2]
                r = math.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
                term = math.erfc(eta * r / math.sqrt(eps_avg)) / (eps_avg * r)
                E += term
                E_outer += abs(term)
    for n3 in (-n, n):
        for n1 in range(-(n - 1), n):
            for n2 in range(-(n - 1), n):
                Rx = n1 * L[0, 0] + n2 * L[1, 0] + n3 * L[2, 0]
                Ry = n1 * L[0, 1] + n2 * L[1, 1] + n3 * L[2, 1]
                Rz = n1 * L[0, 2] + n2 * L[1, 2] + n3 * L[2, 2]
                r = math.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)
                term = math.erfc(eta * r / math.sqrt(eps_avg)) / (eps_avg * r)
                E += term
                E_outer += abs(term)
    return E, E_outer


@njit(cache=True)
def ewald_recip_shell_ring_sum(n, G_mat, eps_inv, eta):
    """Reciprocal-space contribution from ONLY shell max(|m1|,|m2|,|m3|)
    == n. Same face-group enumeration and (E_ring, E_outer_ring)
    semantics as ewald_real_shell_ring_sum() -- see its docstring."""
    if n == 0:
        return 0.0, 0.0
    E = 0.0
    E_outer = 0.0
    for m1 in (-n, n):
        for m2 in range(-n, n + 1):
            for m3 in range(-n, n + 1):
                Gx = m1 * G_mat[0, 0] + m2 * G_mat[1, 0] + m3 * G_mat[2, 0]
                Gy = m1 * G_mat[0, 1] + m2 * G_mat[1, 1] + m3 * G_mat[2, 1]
                Gz = m1 * G_mat[0, 2] + m2 * G_mat[1, 2] + m3 * G_mat[2, 2]
                t0 = Gx * eps_inv[0, 0] + Gy * eps_inv[1, 0] + Gz * eps_inv[2, 0]
                t1 = Gx * eps_inv[0, 1] + Gy * eps_inv[1, 1] + Gz * eps_inv[2, 1]
                t2 = Gx * eps_inv[0, 2] + Gy * eps_inv[1, 2] + Gz * eps_inv[2, 2]
                geg = t0 * Gx + t1 * Gy + t2 * Gz
                if geg < 1e-20:
                    continue
                term = math.exp(-geg / (4.0 * eta ** 2)) / geg
                E += term
                E_outer += abs(term)
    for m2 in (-n, n):
        for m1 in range(-(n - 1), n):
            for m3 in range(-n, n + 1):
                Gx = m1 * G_mat[0, 0] + m2 * G_mat[1, 0] + m3 * G_mat[2, 0]
                Gy = m1 * G_mat[0, 1] + m2 * G_mat[1, 1] + m3 * G_mat[2, 1]
                Gz = m1 * G_mat[0, 2] + m2 * G_mat[1, 2] + m3 * G_mat[2, 2]
                t0 = Gx * eps_inv[0, 0] + Gy * eps_inv[1, 0] + Gz * eps_inv[2, 0]
                t1 = Gx * eps_inv[0, 1] + Gy * eps_inv[1, 1] + Gz * eps_inv[2, 1]
                t2 = Gx * eps_inv[0, 2] + Gy * eps_inv[1, 2] + Gz * eps_inv[2, 2]
                geg = t0 * Gx + t1 * Gy + t2 * Gz
                if geg < 1e-20:
                    continue
                term = math.exp(-geg / (4.0 * eta ** 2)) / geg
                E += term
                E_outer += abs(term)
    for m3 in (-n, n):
        for m1 in range(-(n - 1), n):
            for m2 in range(-(n - 1), n):
                Gx = m1 * G_mat[0, 0] + m2 * G_mat[1, 0] + m3 * G_mat[2, 0]
                Gy = m1 * G_mat[0, 1] + m2 * G_mat[1, 1] + m3 * G_mat[2, 1]
                Gz = m1 * G_mat[0, 2] + m2 * G_mat[1, 2] + m3 * G_mat[2, 2]
                t0 = Gx * eps_inv[0, 0] + Gy * eps_inv[1, 0] + Gz * eps_inv[2, 0]
                t1 = Gx * eps_inv[0, 1] + Gy * eps_inv[1, 1] + Gz * eps_inv[2, 1]
                t2 = Gx * eps_inv[0, 2] + Gy * eps_inv[1, 2] + Gz * eps_inv[2, 2]
                geg = t0 * Gx + t1 * Gy + t2 * Gz
                if geg < 1e-20:
                    continue
                term = math.exp(-geg / (4.0 * eta ** 2)) / geg
                E += term
                E_outer += abs(term)
    return E, E_outer
