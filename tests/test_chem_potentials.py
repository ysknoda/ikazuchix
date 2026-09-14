"""Unit tests for ikazuchix.chem_potentials."""

import warnings

import pytest

from ikazuchix.chem_potentials import (
    chem_pot_correction,
    check_chemical_potentials,
    competing_phase_bound,
)

# A self-consistent "clean" host: composition {A:2, B:2}, chosen so that
# host stability (Check 1) holds exactly and both delta_mu values sit
# comfortably within their allowed bounds (Checks 2/3) -- no warnings
# should fire for this dataset.
CLEAN_DATA = dict(
    bulk=dict(formula="AB", energy=-10.0, composition={"A": 2, "B": 2}),
    elemental_references={"A": -1.0, "B": -3.0},
    chemical_potentials={"A": 0.0, "B": -1.0},
)


def test_chem_pot_correction_vacancy():
    data = dict(CLEAN_DATA)
    data["bulk"] = dict(formula="AB", energy=-10.0, composition={"A": 4, "B": 4})
    cs_entry = {"composition": {"A": 3, "B": 4}}  # one A atom removed
    corr, delta_n = chem_pot_correction(data, cs_entry)
    assert delta_n == {"A": -1, "B": 0}
    assert corr == pytest.approx(-1.0)


def test_chem_pot_correction_interstitial():
    data = dict(CLEAN_DATA)
    data["bulk"] = dict(formula="AB", energy=-10.0, composition={"A": 4, "B": 4})
    cs_entry = {"composition": {"A": 5, "B": 4}}  # one A atom added
    corr, delta_n = chem_pot_correction(data, cs_entry)
    assert delta_n == {"A": 1, "B": 0}
    assert corr == pytest.approx(1.0)


def test_chem_pot_correction_is_species_order_independent():
    """Regression: the correction accumulates floats over the species,
    and floating-point addition is not associative, so the iteration
    order must be deterministic.

    It previously came from a set(), whose str iteration order varies
    between processes under Python's randomized string hashing -- so
    the same input could produce two different bit patterns from run to
    run, silently breaking the project's bit-for-bit reproducibility
    contract. A ternary defect with all three delta_n non-zero is used
    here because the shipped 2-species example happens to have one
    delta_n == 0, which masks the effect entirely.
    """
    import itertools

    data = dict(
        bulk=dict(formula="ABC", energy=-30.0,
                  composition={"A": 24, "B": 24, "C": 12}),
        elemental_references={"A": -3.031, "B": -8.328, "C": -4.9475},
        chemical_potentials={"A": -0.137, "B": -1.276, "C": -0.9231},
    )
    base_comp = {"A": 23, "B": 25, "C": 11}  # every delta_n != 0

    seen = set()
    for order in itertools.permutations(base_comp):
        cs_entry = {"composition": {sp: base_comp[sp] for sp in order}}
        corr, _ = chem_pot_correction(data, cs_entry)
        seen.add(corr.hex())
    assert len(seen) == 1, (
        f"chem_pot_correction produced {len(seen)} distinct bit patterns "
        f"across species orderings: {sorted(seen)}")


def test_clean_input_raises_no_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_chemical_potentials(CLEAN_DATA)
    assert caught == []


def test_host_stability_warning():
    data = dict(CLEAN_DATA)
    data["bulk"] = dict(formula="AB", energy=-100.0, composition={"A": 2, "B": 2})
    with pytest.warns(UserWarning, match="Host stability"):
        check_chemical_potentials(data)


def test_positive_delta_mu_warning():
    data = dict(CLEAN_DATA)
    data["chemical_potentials"] = {"A": 0.5, "B": -1.0}
    with pytest.warns(UserWarning, match="delta_mu_A"):
        check_chemical_potentials(data)


def test_lower_bound_violation_warning():
    data = dict(CLEAN_DATA)
    data["chemical_potentials"] = {"A": 0.0, "B": -5.0}  # far below the allowed range
    with pytest.warns(UserWarning, match="lower bound"):
        check_chemical_potentials(data)


def test_missing_species_in_elemental_references_warns():
    data = dict(CLEAN_DATA)
    data["defects"] = [dict(charge_states=[dict(composition={"A": 1, "C": 1})])]
    with pytest.warns(UserWarning, match="'C' has no entry in 'elemental_references'"):
        check_chemical_potentials(data)


def test_missing_species_in_chemical_potentials_warns():
    data = dict(CLEAN_DATA)
    data["elemental_references"] = dict(data["elemental_references"], C=-2.0)
    data["defects"] = [dict(charge_states=[dict(composition={"A": 1, "C": 1})])]
    with pytest.warns(UserWarning, match="'C' has no entry in 'chemical_potentials'"):
        check_chemical_potentials(data)


# ===========================================================================
# Feat-2: competing_phase_bound / Check 4
# ===========================================================================

def test_competing_phase_bound_computes_raw_formation_enthalpy():
    phase = dict(formula="A2B", energy=-5.0, composition={"A": 2, "B": 1})
    refs = {"A": -1.0, "B": -3.0}
    comp, dH_f = competing_phase_bound(phase, refs)
    assert comp == {"A": 2, "B": 1}
    assert dH_f == pytest.approx(-5.0 - (2 * -1.0 + 1 * -3.0))  # = 0.0


def test_competing_phase_within_bound_does_not_warn():
    data = dict(CLEAN_DATA)
    # dH_f(A) = energy - refs[A] = -0.5 - (-1.0) = 0.5; delta_mu_A=0.0 <= 0.5: OK.
    data["competing_phases"] = [dict(formula="A", energy=-0.5, composition={"A": 1})]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_chemical_potentials(data)
    assert not any("Competing phase" in str(w.message) for w in caught)


def test_competing_phase_violation_warns():
    data = dict(CLEAN_DATA)
    # dH_f(A) = -6.0 - (-1.0) = -5.0; delta_mu_A=0.0 > -5.0: violated.
    data["competing_phases"] = [dict(formula="A", energy=-6.0, composition={"A": 1})]
    with pytest.warns(UserWarning, match="Competing phase"):
        check_chemical_potentials(data)
