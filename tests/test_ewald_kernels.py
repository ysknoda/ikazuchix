"""Numerical-equivalence tests: numba-JIT kernel vs. its pure-Python .py_func.

Only meaningful when numba is actually installed (otherwise the kernel
*is* the plain Python function already, per ikazuchix.numba_shim).
"""

import numpy as np
import pytest
from conftest import requires_numba

from ikazuchix.ewald_kernels import (
    ewald_real_shell_ring_sum,
    ewald_real_shell_sum,
    ewald_recip_shell_ring_sum,
    ewald_recip_shell_sum,
)

L = np.array([[9.57, 0.0, 0.0], [-4.785, 8.2877, 0.0], [0.0, 0.0, 10.38]])
EPS = np.array([[9.5, 0.0, 0.0], [0.0, 9.5, 0.0], [0.0, 0.0, 10.4]])
EPS_INV = np.linalg.inv(EPS)
G_MAT = 2.0 * np.pi * np.linalg.inv(L).T
ETA = 0.189116
EPS_AVG = 9.8


@requires_numba
def test_real_kernel_matches_py_func():
    for n in (2, 3, 5):
        E_jit, Eo_jit = ewald_real_shell_sum(n, L, ETA, EPS_AVG)
        E_py, Eo_py = ewald_real_shell_sum.py_func(n, L, ETA, EPS_AVG)
        assert E_jit == E_py
        assert Eo_jit == Eo_py


@requires_numba
def test_recip_kernel_matches_py_func():
    for n in (2, 3, 5):
        E_jit, Eo_jit = ewald_recip_shell_sum(n, G_MAT, EPS_INV, ETA)
        E_py, Eo_py = ewald_recip_shell_sum.py_func(n, G_MAT, EPS_INV, ETA)
        assert E_jit == E_py
        assert Eo_jit == Eo_py


def test_real_kernel_matches_reference_numpy_implementation():
    """Cross-check against a direct, independent numpy/itertools formula
    (not the production code path) for one shell count.

    Compared with a tolerance, not bit-exactly, precisely *because* the
    reference is independent: it takes the distance from
    ``np.linalg.norm`` while the kernel computes
    ``math.sqrt(Rx*Rx + Ry*Ry + Rz*Rz)`` directly. Those are different
    algorithms -- numpy dispatches to a scaling-robust BLAS routine --
    so whether they agree in the last bit depends on the platform's
    BLAS and libm. They happen to agree on macOS/Accelerate and differ
    by 1 ULP on Linux/OpenBLAS, which is what an earlier strict `==`
    here tripped over in CI (the discrepancy surfaced only in `Eo`: it
    sums the outermost shell alone, ~200x smaller than the full `E`, so
    a last-ULP term error that `E`'s much larger running sum absorbs
    stays visible in `Eo`).

    1e-12 is ~4 orders of magnitude above that ULP noise while still
    far tighter than any real formula defect (a wrong sign, coefficient
    or missing term) could hide under. Bit-exactness *is* asserted
    elsewhere in this file, where it is a genuine project guarantee:
    between the JIT kernel and its own ``.py_func`` twin.
    """
    import itertools
    import math

    n = 4
    E_ref, Eo_ref = 0.0, 0.0
    for n1, n2, n3 in itertools.product(range(-n, n + 1), repeat=3):
        if n1 == 0 and n2 == 0 and n3 == 0:
            continue
        R = n1 * L[0] + n2 * L[1] + n3 * L[2]
        r = np.linalg.norm(R)
        term = math.erfc(ETA * r / math.sqrt(EPS_AVG)) / (EPS_AVG * r)
        E_ref += term
        if max(abs(n1), abs(n2), abs(n3)) == n:
            Eo_ref += abs(term)

    E, Eo = ewald_real_shell_sum(n, L, ETA, EPS_AVG)
    assert E == pytest.approx(E_ref, rel=1e-12)
    assert Eo == pytest.approx(Eo_ref, rel=1e-12)


# ===========================================================================
# Perf-2: outer-shell-only ("ring") kernels
# ===========================================================================


@requires_numba
def test_real_ring_kernel_matches_py_func():
    for n in (0, 1, 2, 3, 5):
        E_jit, Eo_jit = ewald_real_shell_ring_sum(n, L, ETA, EPS_AVG)
        E_py, Eo_py = ewald_real_shell_ring_sum.py_func(n, L, ETA, EPS_AVG)
        assert E_jit == E_py
        assert Eo_jit == Eo_py


@requires_numba
def test_recip_ring_kernel_matches_py_func():
    for n in (0, 1, 2, 3, 5):
        E_jit, Eo_jit = ewald_recip_shell_ring_sum(n, G_MAT, EPS_INV, ETA)
        E_py, Eo_py = ewald_recip_shell_ring_sum.py_func(n, G_MAT, EPS_INV, ETA)
        assert E_jit == E_py
        assert Eo_jit == Eo_py


def test_real_ring_sum_shell_zero_is_origin_only_and_excluded():
    assert ewald_real_shell_ring_sum(0, L, ETA, EPS_AVG) == (0.0, 0.0)


def test_recip_ring_sum_shell_zero_is_origin_only_and_excluded():
    assert ewald_recip_shell_ring_sum(0, G_MAT, EPS_INV, ETA) == (0.0, 0.0)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_real_ring_sum_of_shells_matches_full_cube(n):
    """sum_{s=0}^{n} ring(s) covers the identical point set as
    shell_sum(n, ...) -- the same total up to floating-point
    reassociation (different accumulation order), not necessarily
    bit-exact -- see the ring-kernel module docstring."""
    cumulative = 0.0
    for s in range(n + 1):
        e_ring, _ = ewald_real_shell_ring_sum(s, L, ETA, EPS_AVG)
        cumulative += e_ring
    E_full, _ = ewald_real_shell_sum(n, L, ETA, EPS_AVG)
    assert cumulative == pytest.approx(E_full, rel=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_recip_ring_sum_of_shells_matches_full_cube(n):
    cumulative = 0.0
    for s in range(n + 1):
        e_ring, _ = ewald_recip_shell_ring_sum(s, G_MAT, EPS_INV, ETA)
        cumulative += e_ring
    E_full, _ = ewald_recip_shell_sum(n, G_MAT, EPS_INV, ETA)
    assert cumulative == pytest.approx(E_full, rel=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_real_ring_sum_outer_matches_full_cube_e_outer(n):
    """A single shell's ring-kernel E_outer must equal the full-cube
    kernel's E_outer at cutoff=n -- both sum over the exact same
    shell-n point set."""
    _, ring_e_outer = ewald_real_shell_ring_sum(n, L, ETA, EPS_AVG)
    _, full_e_outer = ewald_real_shell_sum(n, L, ETA, EPS_AVG)
    assert ring_e_outer == pytest.approx(full_e_outer, rel=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_recip_ring_sum_outer_matches_full_cube_e_outer(n):
    _, ring_e_outer = ewald_recip_shell_ring_sum(n, G_MAT, EPS_INV, ETA)
    _, full_e_outer = ewald_recip_shell_sum(n, G_MAT, EPS_INV, ETA)
    assert ring_e_outer == pytest.approx(full_e_outer, rel=1e-9)
