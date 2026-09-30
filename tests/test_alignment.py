"""Point-charge potential phi_PC and the KO alignment term with atomic-site
sampling (docs/adr/0009).

phi_PC is validated like E_PC (tests/test_ewald_physics.py): eta
invariance, an independent reference, periodicity, and the exact
self-energy limit that ties it to E_PC:

    lim_{r->0} [ phi(r) - q K / (sqrt(det eps) |r|_eps) ] = 2 E_PC / q.
"""

import itertools
import math

import numpy as np
import pytest
import yaml
from conftest import EXAMPLE_INPUT

from ikazuchix.constants import COULOMB_PREFACTOR
from ikazuchix.ewald import KumagaiObaCorrectionCalculator, point_charge_potential
from ikazuchix.io_utils import load_input

TRI_L = np.array([[7.1, 0.0, 0.0], [1.3, 6.4, 0.0], [-0.9, 1.1, 8.2]])
TRI_EPS = np.array([[7.0, 0.8, -0.4], [0.8, 9.0, 0.6], [-0.4, 0.6, 11.0]])
DEF = np.array([0.31, 0.47, 0.52])
SITES = np.array(
    [[0.81, 0.97, 0.02], [0.05, 0.10, 0.90], [0.60, 0.20, 0.15], [0.40, 0.55, 0.60]]
)


def _reference_phi(L, eps, q, r_cart, eta=0.45, n=9):
    """Independent vectorized Ewald potential (not the production code)."""
    V = abs(np.linalg.det(L))
    B = 2 * np.pi * np.linalg.inv(L).T
    ei = np.linalg.inv(eps)
    sd = math.sqrt(np.linalg.det(eps))
    idx = np.array(list(itertools.product(range(-n, n + 1), repeat=3)), float)
    d = r_cart + idx @ L
    x = np.sqrt(np.einsum("ij,jk,ik->i", d, ei, d))
    real = sum(math.erfc(eta * v) / (sd * v) for v in x)
    G = (idx @ B)[np.any(idx != 0, axis=1)]
    geg = np.einsum("ij,jk,ik->i", G, eps, G)
    recip = (
        4 * np.pi / V * np.sum(np.exp(-geg / (4 * eta**2)) / geg * np.cos(G @ r_cart))
    )
    return q * COULOMB_PREFACTOR * (real + recip - math.pi / (V * eta**2))


def test_phi_is_independent_of_eta():
    vals = [
        point_charge_potential(TRI_L, TRI_EPS, 2, DEF, SITES, ewald_param=eta)
        for eta in (0.3, 0.6, 1.0, None)
    ]
    for v in vals[1:]:
        np.testing.assert_allclose(v, vals[0], atol=1e-8)


def test_phi_matches_independent_reference():
    got = point_charge_potential(TRI_L, TRI_EPS, -3, DEF, SITES)
    for s, g in zip(SITES, got):
        # minimum-image-independent: the reference sums over all images anyway
        r = (s - DEF) @ TRI_L
        assert g == pytest.approx(_reference_phi(TRI_L, TRI_EPS, -3, r), abs=1e-8)


def test_phi_self_energy_limit_equals_twice_e_pc_over_q():
    q = 2
    e_pc = KumagaiObaCorrectionCalculator(
        TRI_L, TRI_EPS, q, DEF, 0.0, adaptive=True, ewald_tol=1e-12, max_shells=40
    ).compute()["E_PC"]
    dr_cart = np.array([1e-4, -0.7e-4, 0.5e-4])
    site = DEF + dr_cart @ np.linalg.inv(TRI_L)
    phi = point_charge_potential(TRI_L, TRI_EPS, q, DEF, [site])[0]
    r_eps = math.sqrt(dr_cart @ np.linalg.inv(TRI_EPS) @ dr_cart)
    self_term = q * COULOMB_PREFACTOR / (math.sqrt(np.linalg.det(TRI_EPS)) * r_eps)
    assert (phi - self_term) * q / 2 == pytest.approx(e_pc, abs=1e-6)


def test_phi_is_periodic_translation_invariant_and_linear_in_q():
    base = point_charge_potential(TRI_L, TRI_EPS, 1, DEF, SITES)
    shifted_images = point_charge_potential(TRI_L, TRI_EPS, 1, DEF, SITES + [1, -2, 3])
    np.testing.assert_allclose(shifted_images, base, atol=1e-10)
    moved = point_charge_potential(TRI_L, TRI_EPS, 1, DEF + 0.13, SITES + 0.13)
    np.testing.assert_allclose(moved, base, atol=1e-10)
    np.testing.assert_allclose(
        point_charge_potential(TRI_L, TRI_EPS, -2, DEF, SITES), -2 * base, atol=1e-10
    )


def test_positive_charge_gives_negative_far_field_potential():
    # Zero cell average: the far region of a positive point charge sits
    # below the mean, i.e. phi < 0 at the point farthest from it.
    far = DEF + np.array([0.5, 0.5, 0.5])
    assert point_charge_potential(TRI_L, TRI_EPS, 1, DEF, [far])[0] < 0


def test_alignment_with_sampling_sites():
    q, dv = -2, -0.40
    phi = point_charge_potential(TRI_L, TRI_EPS, q, DEF, SITES)
    r = KumagaiObaCorrectionCalculator(
        TRI_L, TRI_EPS, q, DEF, dv, alignment_sites=SITES
    ).compute()
    assert r["phi_PC_sites"] == pytest.approx(phi.mean())
    assert r["E_alignment"] == pytest.approx(q * (dv + phi.mean()))
    assert r["E_KO"] == pytest.approx(r["E_lat"] + r["E_alignment"])


def test_alignment_without_sites_is_unchanged():
    r = KumagaiObaCorrectionCalculator(TRI_L, TRI_EPS, -2, DEF, -0.40).compute()
    assert r["E_alignment"] == pytest.approx(-2 * -0.40)
    assert r["phi_PC_sites"] == 0.0


def test_neutral_state_alignment_is_zero_even_with_sites():
    r = KumagaiObaCorrectionCalculator(
        TRI_L, TRI_EPS, 0, DEF, -0.4, alignment_sites=SITES
    ).compute()
    assert r["E_alignment"] == 0.0 and r["E_KO"] == 0.0


def _write(tmp_path, sites):
    raw = yaml.safe_load(EXAMPLE_INPUT.read_text(encoding="utf-8"))
    raw["defects"][0]["alignment_sampling_sites"] = sites
    p = tmp_path / "in.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_load_input_accepts_sampling_sites(tmp_path):
    load_input(_write(tmp_path, [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]]))


@pytest.mark.parametrize("sites", [[], [[0.1, 0.2]], [[0.1, "x", 0.3]], "abc"])
def test_load_input_rejects_malformed_sampling_sites(tmp_path, sites):
    with pytest.raises(ValueError, match="alignment_sampling_sites"):
        load_input(_write(tmp_path, sites))
