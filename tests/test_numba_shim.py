"""Unit tests for ikazuchix.numba_shim (JIT fallback plumbing)."""

import importlib
import sys

import pytest

from conftest import HAS_NUMBA
import ikazuchix.numba_shim as numba_shim


def test_has_numba_matches_environment():
    try:
        import numba  # noqa: F401
        expected = True
    except ImportError:
        expected = False
    assert numba_shim.HAS_NUMBA is expected


def test_fallback_njit_bare_form():
    @numba_shim._fallback_njit
    def f(x):
        return x + 1

    assert f(1) == 2
    assert f.py_func(1) == 2


def test_fallback_njit_parametrized_form():
    @numba_shim._fallback_njit(cache=True)
    def f(x):
        return x * 2

    assert f(3) == 6
    assert f.py_func(3) == 6


def test_numba_enabled_toggle_round_trips():
    original = numba_shim.numba_active()
    try:
        numba_shim.set_numba_enabled(False)
        assert numba_shim.numba_active() is False
        numba_shim.set_numba_enabled(True)
        assert numba_shim.numba_active() is HAS_NUMBA
    finally:
        numba_shim.set_numba_enabled(original)


def test_use_kernel_respects_runtime_toggle():
    @numba_shim.njit(cache=True)
    def kernel(x):
        return x * 3

    original = numba_shim.numba_active()
    try:
        numba_shim.set_numba_enabled(True)
        f_on = numba_shim.use_kernel(kernel)
        numba_shim.set_numba_enabled(False)
        f_off = numba_shim.use_kernel(kernel)
        assert f_on(2) == f_off(2) == 6
        if HAS_NUMBA:
            # When numba is active, use_kernel returns the dispatcher
            # itself; when disabled, it returns the plain function.
            assert f_off is kernel.py_func
    finally:
        numba_shim.set_numba_enabled(original)


def test_fallback_path_when_numba_unimportable(monkeypatch):
    """Force ImportError on 'numba' and re-import the shim fresh, to
    exercise the HAS_NUMBA=False branch even on a numba-installed
    machine (this dev environment has numba, so this is the only way
    to cover that branch)."""
    monkeypatch.setitem(sys.modules, "numba", None)
    fresh = importlib.reload(numba_shim)
    try:
        assert fresh.HAS_NUMBA is False
        assert fresh.njit is fresh._fallback_njit

        @fresh.njit(cache=True)
        def f(x):
            return x - 1

        assert f(5) == 4
        assert f.py_func(5) == 4
    finally:
        importlib.reload(numba_shim)  # restore real numba-backed state


@pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
def test_real_numba_dispatcher_exposes_py_func():
    from numba import njit

    @njit(cache=True)
    def f(x):
        return x + 10

    assert hasattr(f, "py_func")
    assert f.py_func(1) == 11
