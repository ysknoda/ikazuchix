"""Unit tests for ikazuchix.ewald.KumagaiObaCorrectionCalculator."""

import warnings

import numpy as np
import pytest

import ikazuchix.ewald as ewald_module
from ikazuchix.ewald import KumagaiObaCorrectionCalculator, build_ewald_shell_cache
from ikazuchix.ewald_kernels import ewald_real_shell_sum, ewald_recip_shell_sum

L = np.array([[9.57, 0.0, 0.0],
              [-4.785, 8.2877, 0.0],
              [0.0, 0.0, 10.38]])
EPS = np.array([[9.5, 0.0, 0.0], [0.0, 9.5, 0.0], [0.0, 0.0, 10.4]])
POS = [0.16667, 0.33333, 0.0]


def _calc(q, **kwargs):
    kwargs.setdefault("delta_v", 0.0)
    return KumagaiObaCorrectionCalculator(
        lattice=L, epsilon=EPS, charge=q, defect_frac=POS, **kwargs)


def test_neutral_charge_is_all_zero():
    result = _calc(0).compute()
    assert result == {k: 0.0 for k in
                       ("E_real", "E_recip", "E_self_ewald", "E_neutralize",
                        "E_PC", "E_lat", "phi_PC_sites", "E_alignment", "E_KO")}


def test_charge_sign_symmetry():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pos = KumagaiObaCorrectionCalculator(
            lattice=L, epsilon=EPS, charge=2, defect_frac=POS, delta_v=0.05).compute()
        neg = KumagaiObaCorrectionCalculator(
            lattice=L, epsilon=EPS, charge=-2, defect_frac=POS, delta_v=0.05).compute()

    # q^2-dependent terms are identical for +q and -q ...
    for key in ("E_real", "E_recip", "E_self_ewald", "E_neutralize", "E_PC", "E_lat"):
        assert pos[key] == pytest.approx(neg[key])
    # ... while the linear alignment term (and hence E_KO) flips sign.
    assert pos["E_alignment"] == pytest.approx(-neg["E_alignment"])
    assert pos["E_alignment"] == pytest.approx(2 * 0.05)


def test_result_is_cached():
    calc = _calc(1, delta_v=0.02)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        first = calc.compute()
    second = calc.compute()
    assert first is second


def test_convergence_warning_fires_for_small_cutoff():
    # One shell is far from converged (~5e-2 eV outer-shell contribution)
    # even with the balanced default eta.
    calc = _calc(3, real_cutoff=1, recip_cutoff=1)
    with pytest.warns(RuntimeWarning, match="real-space|recip-space"):
        calc.compute()


def test_convergence_warning_absent_for_large_cutoff():
    calc = _calc(1, real_cutoff=12, recip_cutoff=12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute()
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []


def test_adaptive_mode_grows_shell_count_and_converges():
    calc = _calc(3, real_cutoff=1, recip_cutoff=1, adaptive=True,
                  ewald_tol=1e-6, max_shells=20)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.compute()
    assert calc.n_real_used > 1 or calc.n_recip_used > 1
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings == []  # should have converged within max_shells=20


def test_adaptive_mode_warns_when_max_shells_too_small():
    calc = _calc(3, real_cutoff=1, recip_cutoff=1, adaptive=True,
                  ewald_tol=1e-6, max_shells=1)  # cannot grow at all
    with pytest.warns(RuntimeWarning, match="did not reach"):
        calc.compute()


def test_non_positive_eta_raises():
    with pytest.raises(ValueError, match="eta"):
        _calc(1, ewald_param=0.0)
    with pytest.raises(ValueError, match="eta"):
        _calc(1, ewald_param=-0.5)


# ===========================================================================
# Perf-1: EwaldShellCache (shared raw-shell-sum cache across charge states)
# ===========================================================================

def test_ewald_shell_cache_memoizes_kernel_calls(monkeypatch):
    """A repeated request for the same shell count n must not re-invoke
    the (expensive) underlying kernel; a new n must."""
    call_count = {"real": 0, "recip": 0}
    original_real  = ewald_module.ewald_real_shell_sum
    original_recip = ewald_module.ewald_recip_shell_sum

    def counting_real(*args, **kwargs):
        call_count["real"] += 1
        return original_real(*args, **kwargs)
    counting_real.py_func = counting_real  # keep use_kernel() working with --no-numba

    def counting_recip(*args, **kwargs):
        call_count["recip"] += 1
        return original_recip(*args, **kwargs)
    counting_recip.py_func = counting_recip

    monkeypatch.setattr(ewald_module, "ewald_real_shell_sum", counting_real)
    monkeypatch.setattr(ewald_module, "ewald_recip_shell_sum", counting_recip)

    cache = build_ewald_shell_cache(L, EPS)
    cache.raw_real_fixed(3)
    cache.raw_real_fixed(3)   # same n: cache hit, no new kernel call
    cache.raw_real_fixed(4)   # different n: new kernel call
    cache.raw_recip_fixed(3)
    cache.raw_recip_fixed(3)  # cache hit

    assert call_count["real"] == 2    # n=3, n=4
    assert call_count["recip"] == 1   # n=3 only


def test_shell_cache_matches_uncached_calculator():
    cache = build_ewald_shell_cache(L, EPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        uncached = _calc(2, real_cutoff=5, recip_cutoff=5).compute()
        cached = _calc(2, real_cutoff=5, recip_cutoff=5, shell_cache=cache).compute()
    assert uncached == cached


def test_shell_cache_shared_across_charge_states_preserves_per_charge_warnings():
    """The core Perf-1 risk: caching the raw (q-independent) shell sums
    must not leak into the (q-dependent) convergence-warning decision.
    Query several charge states through one SHARED cache -- as
    runner.py does for one defect's charge_states loop -- and confirm
    each charge state's warning status still matches what an
    independent, uncached calculation of that same charge would give;
    i.e. querying q=1 first (populating the cache) must not affect
    q=3's later, independently-scaled decision."""
    cache = build_ewald_shell_cache(L, EPS)
    real_cutoff, recip_cutoff = 3, 3

    cached_warned = {}
    for q in (1, -1, 2, -2, 3, -3):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _calc(q, real_cutoff=real_cutoff, recip_cutoff=recip_cutoff,
                  shell_cache=cache).compute()
        cached_warned[q] = any(
            issubclass(w.category, RuntimeWarning) for w in caught)

    for q, expect_warned in cached_warned.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _calc(q, real_cutoff=real_cutoff, recip_cutoff=recip_cutoff).compute()
        uncached_warned = any(
            issubclass(w.category, RuntimeWarning) for w in caught)
        assert uncached_warned == expect_warned, f"mismatch at q={q}"


def test_adaptive_mode_default_off_matches_fixed_cutoff_result():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fixed = _calc(2, real_cutoff=3, recip_cutoff=3).compute()
        explicit_non_adaptive = _calc(
            2, real_cutoff=3, recip_cutoff=3, adaptive=False).compute()
    assert fixed == explicit_non_adaptive


# ===========================================================================
# Perf-2: EwaldShellCache incremental (ring-kernel) adaptive-mode growth
# ===========================================================================

@pytest.mark.parametrize("n", [3, 5, 8])
def test_shell_cache_raw_upto_matches_full_cube_within_tolerance(n):
    """The incrementally ring-grown cumulative sum through shell n must
    equal the from-scratch full-cube kernel's sum at cutoff n, up to
    floating-point reassociation (see ewald_kernels.py's ring-kernel
    module docstring) -- not necessarily bit-exact."""
    cache = build_ewald_shell_cache(L, EPS)
    E_upto_real, _ = cache.raw_real_upto(n)
    E_full_real, _ = ewald_real_shell_sum(n, L, cache.eta, cache.eps_inv, cache.sqrt_det)
    assert E_upto_real == pytest.approx(E_full_real, rel=1e-9)

    E_upto_recip, _ = cache.raw_recip_upto(n)
    E_full_recip, _ = ewald_recip_shell_sum(n, cache.G_mat, cache.eps, cache.eta)
    assert E_upto_recip == pytest.approx(E_full_recip, rel=1e-9)


def test_shell_cache_raw_upto_resumes_from_cached_state_out_of_order():
    """Querying a smaller n after a larger one is a plain cache hit;
    querying a larger n afterward resumes growth from the highest
    previously-cached shell instead of restarting from shell 0 -- the
    final result must be independent of the order requests arrived in."""
    cache = build_ewald_shell_cache(L, EPS)
    cache.raw_real_upto(8)
    E5, _ = cache.raw_real_upto(5)  # already cached from the n=8 growth pass
    assert E5 == cache._real_upto[5][0]

    E12, _ = cache.raw_real_upto(12)  # must resume from n=8, not restart
    assert cache._real_upto_max == 12

    fresh = build_ewald_shell_cache(L, EPS)
    E12_fresh, _ = fresh.raw_real_upto(12)
    assert E12 == E12_fresh


def test_adaptive_mode_with_shell_cache_matches_uncached_within_tolerance():
    """The plan's explicit numerical-reproducibility ruling for
    --adaptive-ewald: incremental (cached) vs from-scratch (uncached)
    adaptive growth must agree to a relative tolerance, not bit-exactly."""
    cache = build_ewald_shell_cache(L, EPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        uncached = _calc(3, real_cutoff=3, recip_cutoff=3, adaptive=True,
                          ewald_tol=1e-6, max_shells=20).compute()
        cached = _calc(3, real_cutoff=3, recip_cutoff=3, adaptive=True,
                        ewald_tol=1e-6, max_shells=20, shell_cache=cache).compute()
    for key in uncached:
        assert cached[key] == pytest.approx(uncached[key], rel=1e-9), key


def test_adaptive_mode_with_shell_cache_shared_across_charge_states():
    """Two charge states of the same defect, both adaptive, sharing one
    cache (as runner.py wires them): each must independently converge
    and match its own uncached equivalent, despite reusing the other's
    already-accumulated ring data."""
    cache = build_ewald_shell_cache(L, EPS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cached_q1 = _calc(1, real_cutoff=3, recip_cutoff=3, adaptive=True,
                           ewald_tol=1e-6, max_shells=20, shell_cache=cache).compute()
        cached_q3 = _calc(3, real_cutoff=3, recip_cutoff=3, adaptive=True,
                           ewald_tol=1e-6, max_shells=20, shell_cache=cache).compute()
        uncached_q1 = _calc(1, real_cutoff=3, recip_cutoff=3, adaptive=True,
                             ewald_tol=1e-6, max_shells=20).compute()
        uncached_q3 = _calc(3, real_cutoff=3, recip_cutoff=3, adaptive=True,
                             ewald_tol=1e-6, max_shells=20).compute()
    for key in uncached_q1:
        assert cached_q1[key] == pytest.approx(uncached_q1[key], rel=1e-9), key
    for key in uncached_q3:
        assert cached_q3[key] == pytest.approx(uncached_q3[key], rel=1e-9), key
