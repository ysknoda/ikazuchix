"""Physics validation of the Kumagai-Oba point-charge lattice energy E_PC.

These tests pin the Ewald energy to properties any correct implementation
must satisfy, independent of the production code's own numerics:

* invariance under the Ewald splitting parameter eta,
* the simple-cubic Madelung constant in the isotropic limit,
* agreement with an independent brute-force reference (triclinic cell,
  full non-diagonal anisotropic dielectric tensor),
* invariance under a joint rigid rotation of lattice and tensor,
* the 1/eps scaling of an isotropic tensor,
* the sign convention E_lat = -E_PC > 0, E_KO = E_lat + q*delta_v.

See docs/adr/0006 for the defect these guard against (an eta-dependent
E_lat, ~10x too large, in ikazuchix <= 0.1.0a1).
"""

import itertools
import math
import warnings

import numpy as np
import pytest

from ikazuchix.constants import COULOMB_PREFACTOR
from ikazuchix.ewald import KumagaiObaCorrectionCalculator

# Simple-cubic Madelung constant for a point charge in a neutralizing
# background, E = -q^2 alpha / (2 eps L): alpha = 2.8373 (Makov & Payne,
# PRB 51, 4014 (1995), Eq. 15 / Table I; the more precise 2.837297 is
# the standard value of the Wigner-lattice constant for sc).
ALPHA_SC = 2.837297

# Relaxed 32-atom AlN supercell and DFPT dielectric tensor from the
# vasp_smoke run (docs/adr/0003); independent reference value from
# vasp_smoke/indep_ewald.py (eta-invariant to 1e-6 eV).
ALN_L = np.array(
    [[6.257123, 0.0, 0.0], [-3.128561, 5.418827, 0.0], [0.0, 0.0, 10.032086]]
)
ALN_EPS = np.diag([8.236233, 8.236233, 9.759817])
ALN_E_PC_REF = -0.312145

# Deliberately awkward case: triclinic cell and a symmetric,
# positive-definite, non-diagonal dielectric tensor.
TRI_L = np.array([[7.1, 0.0, 0.0], [1.3, 6.4, 0.0], [-0.9, 1.1, 8.2]])
TRI_EPS = np.array([[7.0, 0.8, -0.4], [0.8, 9.0, 0.6], [-0.4, 0.6, 11.0]])


def _e_pc(L, eps, q=1, eta=None, tol=1e-12):
    """E_PC from the production calculator, driven to tight convergence."""
    calc = KumagaiObaCorrectionCalculator(
        lattice=np.asarray(L, float),
        epsilon=np.asarray(eps, float),
        charge=q,
        defect_frac=[0.0, 0.0, 0.0],
        delta_v=0.0,
        ewald_param=eta,
        real_cutoff=2,
        recip_cutoff=2,
        adaptive=True,
        ewald_tol=tol,
        max_shells=40,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # must converge
        return calc.compute()


def _brute_force_e_pc(L, eps, q=1, eta=0.5, n=10):
    """Independent vectorized reference (not the production kernels):
    Rurali-Cartoixa / KO 2014 Eq. 8 anisotropic Ewald energy."""
    L = np.asarray(L, float)
    eps = np.asarray(eps, float)
    V = abs(np.linalg.det(L))
    B = 2.0 * np.pi * np.linalg.inv(L).T
    eps_inv = np.linalg.inv(eps)
    sd = math.sqrt(np.linalg.det(eps))
    idx = np.array(
        [m for m in itertools.product(range(-n, n + 1), repeat=3) if m != (0, 0, 0)],
        dtype=float,
    )
    R = idx @ L
    r = np.sqrt(np.einsum("ij,jk,ik->i", R, eps_inv, R))
    real = np.sum([math.erfc(eta * x) for x in r] / (sd * r))
    G = idx @ B
    geg = np.einsum("ij,jk,ik->i", G, eps, G)
    recip = 4.0 * np.pi / V * np.sum(np.exp(-geg / (4.0 * eta**2)) / geg)
    const = -2.0 * eta / (math.sqrt(math.pi) * sd) - math.pi / (V * eta**2)
    return 0.5 * q * q * COULOMB_PREFACTOR * (real + recip + const)


@pytest.mark.parametrize(
    "L,eps",
    [(ALN_L, ALN_EPS), (TRI_L, TRI_EPS)],
    ids=["AlN-hexagonal", "triclinic-full-tensor"],
)
def test_e_pc_is_independent_of_eta(L, eps):
    values = [_e_pc(L, eps, eta=eta)["E_PC"] for eta in (0.3, 0.5, 0.75, 1.1, None)]
    assert max(values) - min(values) < 1e-8, values


@pytest.mark.parametrize("eps_scalar", [1.0, 5.0, 12.3])
@pytest.mark.parametrize("side", [8.0, 12.5])
def test_simple_cubic_madelung_limit(side, eps_scalar):
    L = side * np.eye(3)
    got = _e_pc(L, eps_scalar * np.eye(3), q=1)["E_PC"]
    expected = -COULOMB_PREFACTOR * ALPHA_SC / (2.0 * eps_scalar * side)
    assert got == pytest.approx(expected, rel=2e-6)


@pytest.mark.parametrize(
    "L,eps",
    [(ALN_L, ALN_EPS), (TRI_L, TRI_EPS)],
    ids=["AlN-hexagonal", "triclinic-full-tensor"],
)
@pytest.mark.parametrize("q", [1, -2, 3])
def test_matches_independent_brute_force_reference(L, eps, q):
    ref = _brute_force_e_pc(L, eps, q=q, eta=0.45, n=9)
    got = _e_pc(L, eps, q=q)["E_PC"]
    assert got == pytest.approx(ref, abs=1e-8)


def test_aln_smoke_value_matches_vasp_smoke_reference():
    assert _e_pc(ALN_L, ALN_EPS)["E_PC"] == pytest.approx(ALN_E_PC_REF, abs=1e-6)


def test_joint_rotation_of_lattice_and_tensor_leaves_e_pc_unchanged():
    a, b = 0.7, -1.2
    Rz = np.array(
        [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
    )
    Rx = np.array(
        [[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]]
    )
    Rot = Rx @ Rz
    L_rot = TRI_L @ Rot.T  # rows are lattice vectors
    eps_rot = Rot @ TRI_EPS @ Rot.T
    assert _e_pc(L_rot, eps_rot)["E_PC"] == pytest.approx(
        _e_pc(TRI_L, TRI_EPS)["E_PC"], abs=1e-9
    )


def test_isotropic_tensor_scales_as_one_over_eps():
    base = _e_pc(TRI_L, np.eye(3))["E_PC"]
    assert _e_pc(TRI_L, 7.5 * np.eye(3))["E_PC"] == pytest.approx(base / 7.5, rel=1e-9)


def test_scales_as_q_squared():
    e1 = _e_pc(ALN_L, ALN_EPS, q=1)["E_PC"]
    for q in (-3, -1, 2):
        assert _e_pc(ALN_L, ALN_EPS, q=q)["E_PC"] == pytest.approx(
            q * q * e1, rel=1e-12
        )


def test_sign_convention_lattice_correction_is_minus_e_pc():
    calc = KumagaiObaCorrectionCalculator(
        lattice=ALN_L,
        epsilon=ALN_EPS,
        charge=-2,
        defect_frac=[0.5, 0.5, 0.5],
        delta_v=0.1,
    )
    r = calc.compute()
    assert r["E_PC"] < 0.0
    assert r["E_lat"] == -r["E_PC"]
    assert r["E_KO"] == pytest.approx(r["E_lat"] + (-2) * 0.1, abs=1e-15)


def test_default_eta_converges_in_three_shells_for_example_cells():
    """The default eta = sqrt(pi*eps_avg)/V^(1/3) balances the two sums
    so the fixed default cutoff (3 shells) needs no convergence warning."""
    for L, eps in ((ALN_L, ALN_EPS), (TRI_L, TRI_EPS)):
        calc = KumagaiObaCorrectionCalculator(
            lattice=L, epsilon=eps, charge=3, defect_frac=[0, 0, 0], delta_v=0.0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = calc.compute()["E_PC"]
        assert got == pytest.approx(_e_pc(L, eps, q=3)["E_PC"], abs=1e-7)
