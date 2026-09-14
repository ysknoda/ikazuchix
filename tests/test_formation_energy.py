"""Unit tests for ikazuchix.formation_energy (formula + CTL analysis)."""

import numpy as np
import pytest

from ikazuchix.formation_energy import (
    analytic_ctls,
    classify_ctls,
    find_ctls,
    formation_energy,
    stable_envelope,
)


def test_formation_energy_scalar():
    ef = formation_energy(E_defect=-10.0, E_bulk=-12.0, chem_corr=0.5,
                           q=-1, E_VBM=5.0, fermi_rel=0.8, E_KO=0.2)
    expected = (-10.0 - -12.0) + 0.5 + (-1) * (5.0 + 0.8) + 0.2
    assert ef == pytest.approx(expected)


def test_formation_energy_array_matches_scalar_pointwise():
    grid = np.linspace(-0.5, 2.0, 37)
    arr = formation_energy(-10.0, -12.0, 0.5, -1, 5.0, grid, 0.2)
    for i, ef_rel in enumerate(grid):
        scalar = formation_energy(-10.0, -12.0, 0.5, -1, 5.0, float(ef_rel), 0.2)
        assert arr[i] == pytest.approx(scalar)


def test_analytic_ctls_two_lines_known_crossing():
    # a_0 = 5.0 (slope 0), a_-1 = 6.0 (slope -1) -> cross at E_F where
    # 5.0 = 6.0 - E_F  =>  E_F = 1.0, E_f = 5.0
    ctls = analytic_ctls([-1, 0], {0: 5.0, -1: 6.0})
    assert len(ctls) == 1
    ctl = ctls[0]
    assert ctl["E_F_CTL"] == pytest.approx(1.0)
    assert ctl["E_f_CTL"] == pytest.approx(5.0)


def test_analytic_ctls_parallel_lines_no_crossing():
    # Two charge states can't have the same q in practice, but the
    # dq == 0 guard should still hold for any equal-slope pair.
    ctls = analytic_ctls([0], {0: 5.0})
    assert ctls == []


def test_analytic_vs_find_ctls_cross_check():
    # Intercepts chosen to be non-collinear in q (an affine a_q would make
    # all three pairwise crossings coincide at a single E_F -- see
    # test_classify_ctls_stable_vs_meta for that scenario), and the grid
    # size is chosen so none of the crossings lands exactly on a grid
    # node (an exact zero would make find_ctls' sign-change detector
    # double-count that crossing, which is a known quirk of the legacy
    # brute-force method rather than something under test here).
    charges = [-2, -1, 0]
    intercepts = {0: 5.5, -1: 6.0, -2: 7.2}
    grid = np.linspace(-1.0, 3.0, 4000)
    ef_by_charge = {q: intercepts[q] + q * grid for q in charges}

    analytic = sorted(analytic_ctls(charges, intercepts), key=lambda c: c["E_F_CTL"])
    brute = sorted(find_ctls(charges, ef_by_charge, grid), key=lambda c: c["E_F_CTL"])

    assert len(analytic) == len(brute)
    for a, b in zip(analytic, brute):
        assert a["q1"] == b["q1"] and a["q2"] == b["q2"]
        # find_ctls is grid-limited; analytic_ctls is exact.
        assert a["E_F_CTL"] == pytest.approx(b["E_F_CTL"], abs=1e-3)
        assert a["E_f_CTL"] == pytest.approx(b["E_f_CTL"], abs=1e-3)


def test_stable_envelope_is_pointwise_minimum():
    grid = np.linspace(-1, 1, 5)
    ef_by_charge = {0: np.full_like(grid, 3.0), -1: 2.0 + grid}
    env = stable_envelope([-1, 0], ef_by_charge)
    np.testing.assert_allclose(env, np.minimum(3.0, 2.0 + grid))


def test_classify_ctls_stable_vs_meta():
    # Same intercepts as the V_Ga case in examples/example_input.yaml
    # (verified end-to-end via the CLI): stable envelope is
    # 0 -> -1 -> -2 with CTLs at E_F=0.5 and E_F=1.2; the direct 0/-2
    # crossing at E_F=0.85 is bypassed by -1 and is therefore meta-stable.
    charges = [-2, -1, 0]
    intercepts = {0: 5.5, -1: 6.0, -2: 7.2}
    grid = np.linspace(-1.0, 3.0, 4000)
    gap = 1.7
    ef_by_charge = {q: intercepts[q] + q * grid for q in charges}
    ctls = analytic_ctls(charges, intercepts)

    stable, meta = classify_ctls(ctls, charges, ef_by_charge, grid, gap)

    stable_pairs = {(c["q1"], c["q2"]) for c in stable}
    meta_pairs = {(c["q1"], c["q2"]) for c in meta}
    assert stable_pairs | meta_pairs == {(-1, 0), (-2, -1), (-2, 0)}
    # The direct -2/0 crossing is never on the envelope (it is bypassed
    # by the -1 state in between), so it must be meta-stable.
    assert (-2, 0) in meta_pairs
    assert (-1, 0) in stable_pairs
    assert (-2, -1) in stable_pairs


def _exact_envelope_classification(charges, intercepts, gap, tol=1e-3):
    """Grid-free ground truth for classify_ctls: every E_f curve is
    exactly affine (E_f(q,E_F) = a_q + q*E_F), so the envelope at any
    E_F is just min_q(a_q + q*E_F) -- no sampling or interpolation
    involved."""
    out = {}
    for ctl in analytic_ctls(charges, intercepts):
        ef = ctl["E_F_CTL"]
        if not (0.0 <= ef <= gap):
            out[(ctl["q1"], ctl["q2"])] = "meta"
            continue
        env = min(intercepts[q] + q * ef for q in charges)
        out[(ctl["q1"], ctl["q2"])] = (
            "stable" if abs(ctl["E_f_CTL"] - env) < tol else "meta")
    return out


def _grid_classification(charges, intercepts, gap, grid):
    ef_by_charge = {q: intercepts[q] + q * grid for q in charges}
    stable, meta = classify_ctls(
        analytic_ctls(charges, intercepts), charges, ef_by_charge, grid, gap)
    out = {(c["q1"], c["q2"]): "stable" for c in stable}
    out.update({(c["q1"], c["q2"]): "meta" for c in meta})
    return out


def test_classify_ctls_kink_within_one_grid_cell_is_not_misclassified():
    """Regression: a CTL with a *different* envelope kink inside the
    same grid cell used to be misclassified.

    These intercepts put the (-2/+1) crossing at E_F = 0.403582 eV,
    3.2 meV above the true envelope (so: clearly meta-stable at the
    1 meV tolerance), while the competing (-2/0) envelope kink sits at
    0.405185 eV -- only 1.6 meV away, well inside the 9.4 meV grid
    spacing. Interpolating the sampled envelope across that cell
    straddles two different linear branches and reported this CTL as
    on-envelope; evaluating the envelope exactly does not.
    """
    charges = [-2, -1, 0, 1]
    intercepts = {-2: 1.926262, -1: 2.720831, 0: 1.115891, 1: 0.715515}
    gap = 3.4
    grid = np.linspace(-0.17, 3.57, 400)

    got = _grid_classification(charges, intercepts, gap, grid)
    assert got[(-2, 1)] == "meta"
    assert got == _exact_envelope_classification(charges, intercepts, gap)


@pytest.mark.parametrize("n_grid_points", [50, 400, 4000])
def test_classify_ctls_is_grid_resolution_independent(n_grid_points):
    """classify_ctls must agree with the exact grid-free ground truth
    regardless of how coarsely the Fermi grid happens to be sampled --
    the grid is a plotting/tabulation artifact, not part of the CTL
    definition."""
    charges = [-2, -1, 0, 1]
    gap = 3.4
    grid = np.linspace(-0.17, 3.57, n_grid_points)
    rng = np.random.default_rng(7)
    for _ in range(400):
        intercepts = {q: float(rng.uniform(-2.0, 4.0)) for q in charges}
        assert (_grid_classification(charges, intercepts, gap, grid)
                == _exact_envelope_classification(charges, intercepts, gap))


def test_classify_ctls_outside_gap_is_meta():
    charges = [0, 1]
    intercepts = {0: 5.0, 1: 3.0}
    grid = np.linspace(-1.0, 10.0, 2001)
    gap = 1.7
    ef_by_charge = {q: intercepts[q] + q * grid for q in charges}
    ctls = analytic_ctls(charges, intercepts)  # crosses at E_F = 2.0, outside gap

    stable, meta = classify_ctls(ctls, charges, ef_by_charge, grid, gap)
    assert stable == []
    assert len(meta) == 1
