"""Unit tests for ikazuchix.stability_region (Feat-2: chemical-potential
stability region analysis, opt-in via --stability-diagram)."""

import pytest

from ikazuchix.stability_region import (
    _clip_polygon,
    feasible_region_2d,
    free_species_order,
    is_feasible_point,
    stability_summary,
)

# A self-consistent ternary host A-B-C, all elemental references at 0.0
# for arithmetic simplicity: dH_f(host) = bulk.energy exactly.
_TERNARY_DATA = dict(
    bulk=dict(formula="ABC", energy=-3.0, composition={"A": 1, "B": 1, "C": 1}),
    elemental_references={"A": 0.0, "B": 0.0, "C": 0.0},
    chemical_potentials={"A": -0.5, "B": -0.5, "C": -2.0},
)


def test_free_species_order_is_deterministic_and_sorted():
    assert free_species_order({"C": 1, "A": 1, "B": 1}) == ["A", "B", "C"]


# ===========================================================================
# Sutherland-Hodgman polygon clipping (general-purpose primitive)
# ===========================================================================

def test_clip_polygon_unit_square_by_diagonal_matches_known_pentagon():
    """Hand-computable case: clip the unit square [0,1]x[0,1] by
    x+y<=1.5. The (1,1) corner is cut off by the segment from (1,0.5)
    to (0.5,1), leaving a pentagon."""
    unit_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    clipped = _clip_polygon(unit_square, 1.0, 1.0, 1.5)
    expected = [(0, 0), (1, 0), (1, 0.5), (0.5, 1), (0, 1)]
    assert len(clipped) == len(expected)
    for (x, y), (ex, ey) in zip(clipped, expected):
        assert x == pytest.approx(ex)
        assert y == pytest.approx(ey)


def test_clip_polygon_non_binding_constraint_leaves_polygon_unchanged():
    unit_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    clipped = _clip_polygon(unit_square, 1.0, 1.0, 100.0)  # never binds
    assert clipped == unit_square


def test_clip_polygon_fully_excluded_returns_empty():
    unit_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    clipped = _clip_polygon(unit_square, 1.0, 0.0, -10.0)  # x <= -10, impossible
    assert clipped == []


# ===========================================================================
# is_feasible_point
# ===========================================================================

def test_is_feasible_point_true_for_consistent_point():
    feasible, violations = is_feasible_point(
        _TERNARY_DATA["chemical_potentials"], _TERNARY_DATA["bulk"]["composition"],
        _TERNARY_DATA["elemental_references"], dH_f_host=-3.0,
        competing_phases=[])
    assert feasible is True
    assert violations == []


def test_is_feasible_point_flags_positive_delta_mu():
    # A: -0.5 -> 0.5 violates the elemental bound; dH_f_host is set to
    # match the new sum (0.5 - 0.5 - 2.0 = -2.0) so only the elemental
    # bound is violated, not also the host-equality constraint.
    deltas = dict(_TERNARY_DATA["chemical_potentials"], A=0.5)
    feasible, violations = is_feasible_point(
        deltas, _TERNARY_DATA["bulk"]["composition"],
        _TERNARY_DATA["elemental_references"], dH_f_host=-2.0,
        competing_phases=[])
    assert feasible is False
    assert any("elemental bound" in v for v in violations)


def test_is_feasible_point_flags_competing_phase_violation():
    competing_phases = [dict(formula="AB", energy=-6.0, composition={"A": 1, "B": 1})]
    feasible, violations = is_feasible_point(
        _TERNARY_DATA["chemical_potentials"], _TERNARY_DATA["bulk"]["composition"],
        _TERNARY_DATA["elemental_references"], dH_f_host=-3.0,
        competing_phases=competing_phases)
    assert feasible is False
    assert any("competing phase" in v for v in violations)


def test_is_feasible_point_works_for_four_species_host():
    """N>=4: feasibility-only, no polygon/vertex enumeration attempted."""
    host = {"A": 1, "B": 1, "C": 1, "D": 1}
    refs = {sp: 0.0 for sp in host}
    deltas = {"A": -1.0, "B": -1.0, "C": -1.0, "D": -1.0}
    feasible, violations = is_feasible_point(
        deltas, host, refs, dH_f_host=-4.0, competing_phases=[])
    assert feasible is True
    assert violations == []


# ===========================================================================
# feasible_region_2d (ternary hosts)
# ===========================================================================

def test_feasible_region_2d_matches_hand_derived_quadrilateral():
    """Host A+B+C, dH_f_host=-3.0, one competing phase AB with
    dH_f=-1.0: the feasible (delta_mu_A, delta_mu_B) region is bounded
    by x<=0, y<=0, x+y>=-3 (from delta_mu_C<=0), x+y<=-1 (competing
    phase) -- a trapezoid with hand-computable corners."""
    competing_phases = [dict(formula="AB", energy=-1.0, composition={"A": 1, "B": 1})]
    sx, sy, sz, vertices = feasible_region_2d(
        _TERNARY_DATA["bulk"]["composition"], _TERNARY_DATA["elemental_references"],
        dH_f_host=-3.0, competing_phases=competing_phases)
    assert (sx, sy, sz) == ("A", "B", "C")
    expected_vertices = [(-1.0, 0.0), (-3.0, 0.0), (0.0, -3.0), (0.0, -1.0)]
    assert len(vertices) == 4
    for ex, ey in expected_vertices:
        assert any(x == pytest.approx(ex) and y == pytest.approx(ey)
                    for x, y in vertices), f"({ex}, {ey}) not found in {vertices}"


def test_feasible_region_2d_empty_when_host_unstable_against_competing_phase():
    """A competing phase far more stable than the host (dH_f=-10, while
    the host's own Δμ_C<=0 constraint requires delta_mu_A+delta_mu_B >=
    -3) leaves no room for delta_mu_A+delta_mu_B <= -10: empty region."""
    competing_phases = [dict(formula="AB", energy=-10.0, composition={"A": 1, "B": 1})]
    _, _, _, vertices = feasible_region_2d(
        _TERNARY_DATA["bulk"]["composition"], _TERNARY_DATA["elemental_references"],
        dH_f_host=-3.0, competing_phases=competing_phases)
    assert vertices == []


def test_feasible_region_2d_requires_exactly_three_species():
    with pytest.raises(ValueError, match="exactly 3 species"):
        feasible_region_2d({"A": 1, "B": 1}, {"A": 0.0, "B": 0.0},
                            dH_f_host=-1.0, competing_phases=[])


def test_feasible_region_2d_box_sizing_scales_with_large_dH_f_host():
    """A large-magnitude dH_f_host must not be silently truncated by a
    fixed starting box: the base triangle (no competing phases) has
    corners at (0,0), (0,dH_f_host), (dH_f_host,0) -- the far corner
    must reach the full value, not get clipped by an undersized box."""
    host = _TERNARY_DATA["bulk"]["composition"]
    refs = _TERNARY_DATA["elemental_references"]
    _, _, _, vertices = feasible_region_2d(
        host, refs, dH_f_host=-1000.0, competing_phases=[])
    assert len(vertices) == 3
    assert min(v[0] for v in vertices) == pytest.approx(-1000.0)
    assert min(v[1] for v in vertices) == pytest.approx(-1000.0)


def test_feasible_region_2d_loose_competing_phase_does_not_shrink_region():
    """A competing phase with a large *positive* dH_f (i.e. very
    unfavorable, easily satisfied) must never bind -- the region must
    equal the no-competing-phase case."""
    host = _TERNARY_DATA["bulk"]["composition"]
    refs = _TERNARY_DATA["elemental_references"]
    loose_phase = [dict(formula="AB", energy=1000.0, composition={"A": 1, "B": 1})]
    _, _, _, vertices_default = feasible_region_2d(
        host, refs, dH_f_host=-3.0, competing_phases=[])
    _, _, _, vertices_with_loose_phase = feasible_region_2d(
        host, refs, dH_f_host=-3.0, competing_phases=loose_phase)
    assert len(vertices_default) == len(vertices_with_loose_phase) == 3


# ===========================================================================
# stability_summary (top-level orchestrator)
# ===========================================================================

def test_stability_summary_ternary_includes_region():
    data = dict(_TERNARY_DATA)
    data["competing_phases"] = [
        dict(formula="AB", energy=-1.0, composition={"A": 1, "B": 1})]
    result = stability_summary(data)
    assert result["feasible"] is True
    assert result["species"] == ["A", "B", "C"]
    assert "region_vertices" in result
    assert len(result["region_vertices"]) == 4


def test_stability_summary_binary_has_no_region():
    data = dict(
        bulk=dict(formula="AB", energy=-10.0, composition={"A": 2, "B": 2}),
        elemental_references={"A": -1.0, "B": -3.0},
        chemical_potentials={"A": 0.0, "B": -1.0},
    )
    result = stability_summary(data)
    assert "region_vertices" not in result


def test_stability_summary_competing_phases_default_empty():
    data = dict(_TERNARY_DATA)  # no 'competing_phases' key at all
    result = stability_summary(data)
    assert result["n_competing_phases"] == 0
    assert result["feasible"] is True
