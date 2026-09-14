"""Kumagai-Oba finite-size charge correction (Ewald summation)."""

import math
import warnings

import numpy as np

from ikazuchix.constants import COULOMB_PREFACTOR, EWALD_CONV_TOL
from ikazuchix.ewald_kernels import (
    ewald_real_shell_ring_sum,
    ewald_real_shell_sum,
    ewald_recip_shell_ring_sum,
    ewald_recip_shell_sum,
)
from ikazuchix.geometry import average_dielectric, lattice_volume, reciprocal_lattice
from ikazuchix.numba_shim import use_kernel


def _derive_ewald_params(lattice, epsilon, ewald_param=None):
    """Geometry-derived Ewald quantities that depend only on (lattice,
    epsilon, ewald_param) -- i.e. are identical for every charge state
    of the same defect. Extracted from
    KumagaiObaCorrectionCalculator.__init__ (unchanged formulas) so
    EwaldShellCache can derive them once per defect instead of once per
    charge state (Perf-1)."""
    V        = lattice_volume(lattice)
    G_mat    = reciprocal_lattice(lattice)
    eps_avg  = average_dielectric(epsilon)
    eps_inv  = np.linalg.inv(epsilon)
    # eta: heuristic  eta = sqrtpi / V^(1/3)  [Ang^-1]
    eta      = (float(ewald_param) if ewald_param is not None
                else math.sqrt(math.pi) / (V ** (1.0 / 3.0)))
    if eta <= 0:
        raise ValueError(
            f"Ewald splitting parameter eta must be positive "
            f"(got {eta}).  Check --ewald-eta.")
    return V, G_mat, eps_avg, eps_inv, eta


class KumagaiObaCorrectionCalculator:
    """
    Finite-size charge correction: Kumagai & Oba, PRB 89, 195205 (2014).

    E_KO  = E_lat + q * DeltaV
    E_lat = E_real + E_recip + E_self_ewald + E_neutralize

    Anisotropic dielectric screening (A-1)
    ---------------------------------------
    Following KO 2014, anisotropy is handled exclusively in the
    reciprocal-space sum via the quadratic form G * eps^-1 * G (Eq. 14).
    The real-space sum (Eq. 13) uses the scalar average eps_avg = tr(eps)/3,
    which is the formulation adopted in the original paper.  This is the
    physically correct decomposition: the long-range Fourier modes carry
    the full tensor information, while the short-range real-space sum is
    well-represented by the isotropic average.

    Ewald convergence check (A-2)
    ------------------------------
    After each Ewald loop the outermost shell contribution is compared
    against EWALD_CONV_TOL (default 1e-6 eV, or ewald_tol if given).  A
    RuntimeWarning is emitted when the sum has not converged, prompting
    the user to increase real_cutoff / recip_cutoff (fixed-cutoff mode,
    the default -- reproduces the original script's numerics exactly),
    or to enable adaptive mode.

    Adaptive high-precision mode (opt-in)
    --------------------------------------
    When adaptive=True, real_cutoff/recip_cutoff are treated as a
    *starting* shell count that is grown by one shell at a time until
    the outermost-shell contribution drops below ewald_tol or
    max_shells is reached (in which case a RuntimeWarning is emitted
    and the best available result is used). This is opt-in and does
    not affect the default (adaptive=False) numerics.

    Numba acceleration (optional)
    -------------------------------
    The shell-summation loops (ikazuchix/ewald_kernels.py) are JIT-compiled
    with numba when available, and otherwise run as plain Python (via
    the same source, through the numba_shim fallback decorator) -- the
    calculation always succeeds, only its speed differs.

    Result caching (C-3)
    --------------------
    compute() caches its result on the instance.  Repeated calls
    (same calculator object) return immediately without recalculation.

    Shared shell-sum cache (Perf-1)
    --------------------------------
    An optional `shell_cache` (an EwaldShellCache, built once per defect
    via build_ewald_shell_cache()) may be supplied to avoid redundantly
    recomputing the raw (unscaled) Ewald shell sums for every charge
    state of the same defect -- those sums depend only on (lattice,
    epsilon, eta), never on q. Default None preserves the original
    per-instance-only behaviour exactly.
    """

    def __init__(self, lattice, epsilon, charge, defect_frac, delta_v,
                 ewald_param=None, real_cutoff=3, recip_cutoff=3,
                 verbose=False, adaptive=False, ewald_tol=None,
                 max_shells=15, shell_cache=None):
        self.L        = lattice
        self.eps      = epsilon
        self.q        = int(charge)
        self.r_def    = np.array(defect_frac, dtype=float)
        self.delta_v  = float(delta_v)
        self.verbose  = verbose
        self.V, self.G_mat, self.eps_avg, self.eps_inv, self.eta = \
            _derive_ewald_params(lattice, epsilon, ewald_param)
        self.n_real   = int(real_cutoff)
        self.n_recip  = int(recip_cutoff)
        self.adaptive   = bool(adaptive)
        self.ewald_tol  = float(ewald_tol) if ewald_tol is not None else EWALD_CONV_TOL
        self.max_shells = int(max_shells)
        self.n_real_used  = self.n_real   # actual shell count used (grows if adaptive)
        self.n_recip_used = self.n_recip
        self.shell_cache  = shell_cache   # (Perf-1) optional defect-level raw-sum cache
        self._cache   = None   # (C-3) result cache

    # ------------------------------------------------------------------
    # Ewald real-space sum  (Eq. 13, KO 2014)
    # ------------------------------------------------------------------

    def _E_real(self):
        """
        E_real = (q^2/2) * K * sum_{R!=0} erfc(eta|R| / sqrteps_avg) / (eps_avg |R|)

        Per KO 2014 the real-space sum uses the isotropic scalar dielectric
        eps_avg = tr(eps)/3.  The full anisotropic screening is captured in the
        reciprocal-space sum through G*eps^-1*G (Eq. 14).

        Convergence check (A-2): contribution of the outermost shell is
        accumulated and compared against the tolerance (fixed cutoff), or
        the cutoff is grown until it converges (adaptive mode).
        """
        if self.shell_cache is not None:
            # (Perf-2) Adaptive mode grows n across repeated calls, so it
            # uses the incremental ring-based cache entry point; fixed
            # mode calls this exactly once per charge state and must
            # stay on the full-cube kernel for bit-for-bit golden
            # reproducibility (see EwaldShellCache.raw_real_upto).
            raw = (self.shell_cache.raw_real_upto if self.adaptive
                   else self.shell_cache.raw_real_fixed)
        else:
            kernel = use_kernel(ewald_real_shell_sum)
            raw = lambda m: kernel(m, self.L, self.eta, self.eps_avg)

        n = self.n_real
        E, E_outer = raw(n)
        scale = 0.5 * (self.q**2) * COULOMB_PREFACTOR
        val_outer = scale * E_outer

        if self.adaptive:
            while val_outer > self.ewald_tol and n < self.max_shells:
                n += 1
                E, E_outer = raw(n)
                val_outer = scale * E_outer
            if val_outer > self.ewald_tol:
                warnings.warn(
                    f"[KO real-space] Adaptive convergence did not reach "
                    f"tolerance {self.ewald_tol:.0e} eV within "
                    f"max_ewald_shells={self.max_shells} (outermost shell "
                    f"contribution: {val_outer:.2e} eV).  Consider "
                    "increasing --max-ewald-shells or --ewald-tol.",
                    RuntimeWarning, stacklevel=4)
            self.n_real_used = n
        elif val_outer > EWALD_CONV_TOL:
            warnings.warn(
                f"[KO real-space] Outermost shell contribution "
                f"({val_outer:.2e} eV) "
                f"exceeds tolerance {EWALD_CONV_TOL:.0e} eV.  "
                f"Consider increasing real_cutoff (current: {n}).",
                RuntimeWarning, stacklevel=4)

        return scale * E

    # ------------------------------------------------------------------
    # Ewald reciprocal-space sum  (Eq. 14, KO 2014)
    # ------------------------------------------------------------------

    def _E_recip(self):
        """
        E_recip = (q^2*K*4pi) / (V*sqrtdet eps) * sum_{G!=0}
                  exp(-G*eps^-1*G / (4eta^2)) / (G*eps^-1*G)

        Convergence check (A-2): outermost shell contribution monitored
        (fixed cutoff), or the cutoff is grown until converged (adaptive).
        """
        if self.shell_cache is not None:
            raw = (self.shell_cache.raw_recip_upto if self.adaptive
                   else self.shell_cache.raw_recip_fixed)
        else:
            kernel = use_kernel(ewald_recip_shell_sum)
            raw = lambda m: kernel(m, self.G_mat, self.eps_inv, self.eta)

        sqrt_det = math.sqrt(abs(np.linalg.det(self.eps)))
        prefactor = (self.q**2) * COULOMB_PREFACTOR * 4.0 * math.pi / (self.V * sqrt_det)

        n = self.n_recip
        E, E_outer = raw(n)
        val_outer = E_outer * prefactor

        if self.adaptive:
            while val_outer > self.ewald_tol and n < self.max_shells:
                n += 1
                E, E_outer = raw(n)
                val_outer = E_outer * prefactor
            if val_outer > self.ewald_tol:
                warnings.warn(
                    f"[KO recip-space] Adaptive convergence did not reach "
                    f"tolerance {self.ewald_tol:.0e} eV within "
                    f"max_ewald_shells={self.max_shells} (outermost shell "
                    f"contribution: {val_outer:.2e} eV).  Consider "
                    "increasing --max-ewald-shells or --ewald-tol.",
                    RuntimeWarning, stacklevel=4)
            self.n_recip_used = n
        elif val_outer > EWALD_CONV_TOL:
            warnings.warn(
                f"[KO recip-space] Outermost shell contribution "
                f"({val_outer:.2e} eV) exceeds tolerance "
                f"{EWALD_CONV_TOL:.0e} eV.  "
                f"Consider increasing recip_cutoff (current: {n}).",
                RuntimeWarning, stacklevel=4)

        return prefactor * E

    # ------------------------------------------------------------------
    # Ewald self-energy  (Eq. 15, KO 2014)
    # ------------------------------------------------------------------

    def _E_self_ewald(self):
        """
        E_self = -q^2 * K * eta / (sqrtpi * eps_avg^(3/2))
        """
        return -(self.q**2) * COULOMB_PREFACTOR * self.eta / (
            math.sqrt(math.pi) * self.eps_avg**1.5)

    # ------------------------------------------------------------------
    # Background neutralization  (Eq. 16, KO 2014)
    # ------------------------------------------------------------------

    def _E_neutralize(self):
        """
        E_bg = -q^2 * K * pi / (2 * V * eta^2 * eps_avg)
        """
        return -(self.q**2) * COULOMB_PREFACTOR * math.pi / (
            2.0 * self.V * self.eta**2 * self.eps_avg)

    # ------------------------------------------------------------------
    # Public interface  (C-3: cached)
    # ------------------------------------------------------------------

    def compute(self):
        """
        Return a dict with all KO correction components [eV].
        Result is cached: repeated calls return immediately.
        """
        if self._cache is not None:
            return self._cache

        if self.q == 0:
            self._cache = {k: 0.0 for k in (
                "E_real", "E_recip", "E_self_ewald", "E_neutralize",
                "E_lat", "E_alignment", "E_KO")}
            return self._cache

        E_real       = self._E_real()
        E_recip      = self._E_recip()
        E_self_ewald = self._E_self_ewald()
        E_neutralize = self._E_neutralize()
        E_lat        = E_real + E_recip + E_self_ewald + E_neutralize
        E_alignment  = self.q * self.delta_v
        E_KO         = E_lat + E_alignment

        self._cache = dict(
            E_real=E_real, E_recip=E_recip,
            E_self_ewald=E_self_ewald, E_neutralize=E_neutralize,
            E_lat=E_lat, E_alignment=E_alignment, E_KO=E_KO)

        if self.verbose:
            print(f"\n  --- KO Correction (q={self.q:+d}) ---")
            for k, v in self._cache.items():
                print(f"    {k:20s} = {v:+.6f} eV")
            print(f"    {'Ewald eta':20s} = {self.eta:.6f} Ang^-1")
            print(f"    {'eps_avg':20s} = {self.eps_avg:.4f}")
            print(f"    {'Volume':20s} = {self.V:.4f} Ang^3")

        return self._cache


# ===========================================================================
# Perf-1: per-defect cache of raw (unscaled) Ewald shell sums
# ===========================================================================

class EwaldShellCache:
    """Per-defect, charge-state-independent cache of RAW Ewald shell
    sums -- the (E, E_outer) pair a kernel returns *before* any
    q^2-dependent scaling is applied.

    Every charge state of one defect shares an identical lattice,
    epsilon, and eta (--ewald-eta is a single global CLI setting, not
    per-charge-state), so the kernel calls -- by far the most expensive
    part of a KO correction -- are pure functions of (lattice, epsilon,
    eta) alone and can be computed once per defect instead of once per
    charge state. Results are memoized per shell count n.

    What this cache deliberately does NOT decide: whether a given shell
    count has converged, or whether to warn about it. Those decisions
    depend on the *scaled* outermost-shell contribution (scale =
    0.5*q^2*COULOMB_PREFACTOR for the real-space sum, an analogous q^2
    prefactor for reciprocal space), so they differ between charge
    states even when the underlying raw sum is identical -- e.g. a
    q=-2 charge state may need a convergence warning that a q=-1 charge
    state, reusing the exact same cached raw sum, correctly does not.
    Callers (KumagaiObaCorrectionCalculator._E_real/_E_recip) apply
    that scaling and make that decision themselves, every time, from
    the raw values retrieved here.
    """

    def __init__(self, L, eta, eps_avg, G_mat, eps_inv):
        self.L        = L
        self.eta      = eta
        self.eps_avg  = eps_avg
        self.G_mat    = G_mat
        self.eps_inv  = eps_inv
        self._real_fixed  = {}
        self._recip_fixed = {}
        # (Perf-2) incremental adaptive-mode growth: {n: (cumulative_E,
        # shell_n's_own_E_outer)}, plus the highest n already folded
        # into that cumulative sum (-1 = nothing accumulated yet).
        self._real_upto  = {}
        self._recip_upto = {}
        self._real_upto_max  = -1
        self._recip_upto_max = -1

    def raw_real_fixed(self, n):
        if n not in self._real_fixed:
            kernel = use_kernel(ewald_real_shell_sum)
            self._real_fixed[n] = kernel(n, self.L, self.eta, self.eps_avg)
        return self._real_fixed[n]

    def raw_recip_fixed(self, n):
        if n not in self._recip_fixed:
            kernel = use_kernel(ewald_recip_shell_sum)
            self._recip_fixed[n] = kernel(n, self.G_mat, self.eps_inv, self.eta)
        return self._recip_fixed[n]

    def raw_real_upto(self, n):
        """(Perf-2) Cumulative real-space sum through shell n, grown
        incrementally via the O(n^2) ring kernel from whatever has
        already been accumulated -- rather than re-walking the full
        O(n^3) cube from scratch, as repeated fixed-cutoff-style calls
        with a growing n would. Returns (E_upto_n, E_outer_of_shell_n):
        the second element is shell n's own ring contribution alone
        (exactly what the adaptive-growth convergence check needs),
        not a cumulative outer value.
        """
        if n > self._real_upto_max:
            kernel = use_kernel(ewald_real_shell_ring_sum)
            running = (self._real_upto[self._real_upto_max][0]
                       if self._real_upto_max >= 0 else 0.0)
            for m in range(self._real_upto_max + 1, n + 1):
                e_ring, e_ring_outer = kernel(m, self.L, self.eta, self.eps_avg)
                running += e_ring
                self._real_upto[m] = (running, e_ring_outer)
            self._real_upto_max = n
        return self._real_upto[n]

    def raw_recip_upto(self, n):
        """(Perf-2) Reciprocal-space analogue of raw_real_upto(); see
        its docstring."""
        if n > self._recip_upto_max:
            kernel = use_kernel(ewald_recip_shell_ring_sum)
            running = (self._recip_upto[self._recip_upto_max][0]
                       if self._recip_upto_max >= 0 else 0.0)
            for m in range(self._recip_upto_max + 1, n + 1):
                e_ring, e_ring_outer = kernel(m, self.G_mat, self.eps_inv, self.eta)
                running += e_ring
                self._recip_upto[m] = (running, e_ring_outer)
            self._recip_upto_max = n
        return self._recip_upto[n]


def build_ewald_shell_cache(lattice, epsilon, ewald_param=None):
    """Factory: derive (V, G_mat, eps_avg, eps_inv, eta) once via the
    same formulas KumagaiObaCorrectionCalculator.__init__ uses, and
    package the subset EwaldShellCache needs (all but V, which is only
    used for per-charge-state scaling, not the raw shell sums)."""
    _V, G_mat, eps_avg, eps_inv, eta = _derive_ewald_params(
        lattice, epsilon, ewald_param)
    return EwaldShellCache(lattice, eta, eps_avg, G_mat, eps_inv)
