"""Unit tests for ikazuchix.equilibrium (Feat-1: self-consistent Fermi
level + defect/carrier concentrations, opt-in via --solve-fermi)."""

import math
import warnings

import numpy as np
import pytest
import yaml

from ikazuchix.constants import (
    BOLTZMANN_EV_PER_K,
    BOLTZMANN_J_PER_K,
    ELECTRON_MASS_KG,
    PLANCK_CONSTANT_J_S,
)
from ikazuchix.equilibrium import (
    bisect_fermi_level,
    charge_neutrality_residual,
    defect_site_density,
    effective_dos,
    resolve_site_density,
    solve_equilibrium,
)
from ikazuchix.io_utils import load_input
from ikazuchix.runner import run_one_defect

_DATA = dict(
    bulk=dict(formula="GaN", energy=-462.924, num_atoms=72,
              composition={"Ga": 36, "N": 36}),
    elemental_references={"Ga": -3.031, "N": -8.328},
    chemical_potentials={"Ga": 0.0, "N": -1.5},
    band_structure=dict(vbm=5.0, cbm=6.7, bandgap=1.7, fermi_level=5.85),
    dielectric=dict(tensor=[[9.5, 0, 0], [0, 9.5, 0], [0, 0, 10.4]]),
    supercell=dict(
        lattice_vectors={"a": [9.57, 0, 0], "b": [-4.785, 8.2877, 0], "c": [0, 0, 10.38]},
        defect_position=[0.16667, 0.33333, 0.0],
    ),
    defects=[
        dict(name="V_Ga", defect_type="vacancy", removed_atoms={"Ga": 1}, charge_states=[
            dict(charge=0, energy=-454.393, num_atoms=71, potential_alignment={"delta_v": 0.0}),
            dict(charge=-1, energy=-451.0689, num_atoms=71, potential_alignment={"delta_v": 0.02}),
        ]),
    ],
)


@pytest.fixture
def data(tmp_path):
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(_DATA), encoding="utf-8")
    return load_input(path)


@pytest.fixture
def grid():
    return np.linspace(-0.17, 1.87, 400)


# ===========================================================================
# effective_dos / carrier_concentrations
# ===========================================================================

def test_effective_dos_matches_independent_recomputation_from_constants():
    """Regression check that effective_dos actually implements its
    documented formula (independently re-derived here from the same
    CODATA constants, not by calling effective_dos)."""
    m_eff, T = 0.5, 250.0
    n_m3 = 2.0 * (2.0 * math.pi * (m_eff * ELECTRON_MASS_KG) * BOLTZMANN_J_PER_K * T
                  / PLANCK_CONSTANT_J_S ** 2) ** 1.5
    assert effective_dos(m_eff, T) == pytest.approx(n_m3 * 1e-6)


def test_effective_dos_matches_well_known_reference_value_at_300k():
    """Cross-check against the commonly-cited ~2.5e19 cm^-3 effective
    density of states for a free-electron-mass band at 300 K (e.g. Sze
    & Ng, 'Physics of Semiconductor Devices') -- loose tolerance since
    that's a rounded textbook figure, not a many-digit reference."""
    assert effective_dos(1.0, 300.0) == pytest.approx(2.5e19, rel=0.05)


def test_effective_dos_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        effective_dos(0.0, 300.0)
    with pytest.raises(ValueError):
        effective_dos(1.0, 0.0)


# ===========================================================================
# bisect_fermi_level / charge_neutrality_residual: analytic cases
# ===========================================================================

def test_intrinsic_fermi_level_symmetric_masses_is_midgap():
    vbm, cbm = 0.0, 2.0
    fermi_rel = bisect_fermi_level(
        lambda fr: charge_neutrality_residual(
            fr, [], vbm, cbm, 300.0, include_carriers=True,
            m_eff_e=1.0, m_eff_h=1.0),
        0.0, cbm - vbm, tol=1e-10)
    assert fermi_rel == pytest.approx((cbm - vbm) / 2, abs=1e-8)


def test_intrinsic_fermi_level_asymmetric_masses_matches_analytic_formula():
    """E_F,intrinsic = (vbm+cbm)/2 + (3kT/4)*ln(m_h*/m_e*) -- derived by
    setting n(E_F) = p(E_F) and solving for E_F analytically."""
    vbm, cbm = 0.0, 2.0
    T = 300.0
    m_eff_e, m_eff_h = 0.2, 1.0
    fermi_rel = bisect_fermi_level(
        lambda fr: charge_neutrality_residual(
            fr, [], vbm, cbm, T, include_carriers=True,
            m_eff_e=m_eff_e, m_eff_h=m_eff_h),
        0.0, cbm - vbm, tol=1e-10)
    kT = BOLTZMANN_EV_PER_K * T
    expected = (cbm - vbm) / 2 + (3 * kT / 4) * math.log(m_eff_h / m_eff_e)
    assert fermi_rel == pytest.approx(expected, abs=1e-6)


def test_bisect_matches_known_defect_crossing_point_no_carriers():
    """Single amphoteric defect, charge states q=-1,0,+1 (q=0
    contributes nothing -- included anyway for fidelity), equal site
    density/degeneracy: with carriers excluded, the neutrality point is
    exactly the analytic (0/-1)-vs-(0/+1)... i.e. the (-1/+1) CTL,
    E_F* = (a_(-1) - a_(+1)) / 2, the same y-intercept-crossing formula
    used by formation_energy.analytic_ctls."""
    vbm, cbm = 0.0, 2.0
    n_sites = 1e20
    defect_ctx_list = [
        dict(q=-1, ef_intercept=3.0, n_sites_cm3=n_sites, degeneracy=1),
        dict(q=0, ef_intercept=2.0, n_sites_cm3=n_sites, degeneracy=1),
        dict(q=1, ef_intercept=1.0, n_sites_cm3=n_sites, degeneracy=1),
    ]
    fermi_rel = bisect_fermi_level(
        lambda fr: charge_neutrality_residual(
            fr, defect_ctx_list, vbm, cbm, 300.0, include_carriers=False),
        0.0, cbm - vbm, tol=1e-10)
    assert fermi_rel == pytest.approx((3.0 - 1.0) / 2, abs=1e-6)


def test_charge_neutrality_residual_no_carriers_excludes_carrier_term():
    """include_carriers=False must drop the p-n term entirely, not just
    zero out its contribution incidentally."""
    vbm, cbm = 0.0, 2.0
    defect_ctx_list = [dict(q=-1, ef_intercept=1.0, n_sites_cm3=1e18, degeneracy=1)]
    with_carriers = charge_neutrality_residual(
        0.5, defect_ctx_list, vbm, cbm, 300.0, include_carriers=True)
    without_carriers = charge_neutrality_residual(
        0.5, defect_ctx_list, vbm, cbm, 300.0, include_carriers=False)
    assert with_carriers != pytest.approx(without_carriers)


def test_bisect_fermi_level_raises_when_no_sign_change_in_bracket():
    with pytest.raises(ValueError, match="does not change sign"):
        bisect_fermi_level(lambda x: 1.0, 0.0, 1.0)


def test_bisect_fermi_level_endpoint_exact_root_returns_immediately():
    assert bisect_fermi_level(lambda x: 0.0 if x == 0.5 else (1.0 - x), 0.0, 0.5) == 0.5


def test_bisect_fermi_level_inverted_bracket_raises():
    """e_hi <= e_lo must raise, not silently return a meaningless value:
    without this guard, (hi - lo) < tol is already true (negative width)
    on the very first loop iteration, so bisection would return the
    bracket midpoint without ever having actually converged on a root."""
    with pytest.raises(ValueError, match="inverted"):
        bisect_fermi_level(lambda x: 1.0 - x, 1.0, 0.0)


def test_bisect_fermi_level_empty_bracket_raises():
    with pytest.raises(ValueError, match="inverted"):
        bisect_fermi_level(lambda x: 1.0 - x, 0.5, 0.5)


# ===========================================================================
# defect_site_density / resolve_site_density
# ===========================================================================

def test_defect_site_density_auto_derivation_vacancy():
    density = defect_site_density(
        dict(removed_atoms={"Ga": 1}), {"Ga": 36, "N": 36}, 1000.0)
    assert density == pytest.approx(36 * 1e24 / 1000.0)


def test_defect_site_density_none_for_interstitial():
    assert defect_site_density(
        dict(added_atoms={"N": 1}), {"Ga": 36, "N": 36}, 1000.0) is None


def test_defect_site_density_none_for_multi_species_removal():
    assert defect_site_density(
        dict(removed_atoms={"Ga": 1, "N": 1}), {"Ga": 36, "N": 36}, 1000.0) is None


def test_resolve_site_density_explicit_overrides_auto():
    density = resolve_site_density(
        dict(removed_atoms={"Ga": 1}, site_density=1e19),
        {"Ga": 36, "N": 36}, 1000.0)
    assert density == 1e19


def test_resolve_site_density_raises_when_neither_available():
    with pytest.raises(ValueError, match="site density"):
        resolve_site_density(
            dict(name="X_i", added_atoms={"Ga": 1}), {"Ga": 36, "N": 36}, 1000.0)


# ===========================================================================
# solve_equilibrium: end-to-end orchestration
# ===========================================================================

def test_solve_equilibrium_end_to_end_auto_site_density(data, grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        all_results = [run_one_defect(d, data, 0.85, grid) for d in data["defects"]]
    result = solve_equilibrium(data, all_results, temperature_k=300.0)

    assert 0.0 <= result["fermi_level_rel_vbm"] <= 1.7
    assert result["electron_conc_cm3"] > 0
    assert result["hole_conc_cm3"] > 0
    # Only the non-zero charge state (q=-1) appears; q=0 is excluded.
    assert [c["q"] for c in result["defect_concentrations"]] == [-1]


def test_solve_equilibrium_reports_include_carriers_flag(data, grid):
    """Full-bracket bisection with carriers excluded and only an
    acceptor-like defect present has no bracketed root (nothing
    supplies positive charge) -- see the direct
    charge_neutrality_residual test above for --no-carriers' numeric
    effect. This checks the flag is threaded through and reported
    correctly for the (default, carriers-included) solvable case."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        all_results = [run_one_defect(d, data, 0.85, grid) for d in data["defects"]]
    result = solve_equilibrium(data, all_results, temperature_k=300.0,
                                include_carriers=True)
    assert result["include_carriers"] is True


def test_solve_equilibrium_raises_for_unresolvable_site_density(tmp_path, grid):
    data_dict = dict(_DATA)
    data_dict["defects"] = [dict(
        name="X_i", defect_type="interstitial", added_atoms={"Ga": 1},
        charge_states=[
            dict(charge=0, energy=-454.0, num_atoms=73,
                 potential_alignment={"delta_v": 0.0}),
            dict(charge=1, energy=-450.0, num_atoms=73,
                 potential_alignment={"delta_v": 0.0}),
        ],
    )]
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(data_dict), encoding="utf-8")
    interstitial_data = load_input(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        all_results = [run_one_defect(d, interstitial_data, 0.85, grid)
                        for d in interstitial_data["defects"]]
    with pytest.raises(ValueError, match="site density"):
        solve_equilibrium(interstitial_data, all_results, temperature_k=300.0)


def test_solve_equilibrium_skips_site_density_for_neutral_only_defect(tmp_path, grid):
    """A defect whose only charge state is q=0 never appears in the
    neutrality sum, so it must not be required to have a resolvable
    site density (interstitial, no explicit site_density given)."""
    data_dict = dict(_DATA)
    data_dict["defects"] = [dict(
        name="Perfect_i", defect_type="interstitial", added_atoms={"Ga": 1},
        charge_states=[
            dict(charge=0, energy=-454.0, num_atoms=73,
                 potential_alignment={"delta_v": 0.0}),
        ],
    )]
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(data_dict), encoding="utf-8")
    neutral_only_data = load_input(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        all_results = [run_one_defect(d, neutral_only_data, 0.85, grid)
                        for d in neutral_only_data["defects"]]
    result = solve_equilibrium(neutral_only_data, all_results, temperature_k=300.0)
    assert result["defect_concentrations"] == []
