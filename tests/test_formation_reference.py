"""Tests for the optional 'bulk.formation_reference' field (docs/adr/0007).

It carries a k-converged host energy (any cell with the host's
stoichiometry) used for the host formation enthalpy -- host-stability
checks and the stability region -- while 'bulk.energy' stays the
reference for E_def - E_bulk.
"""

import copy
import warnings

import pytest
import yaml
from conftest import EXAMPLE_INPUT, run_cli

from ikazuchix.chem_potentials import (
    check_chemical_potentials,
    host_formation_reference,
)
from ikazuchix.io_utils import load_input
from ikazuchix.stability_region import stability_summary

# Binary host with elemental references 0: dH_f is the energy itself.
# The 4-atom supercell energy (-5.0) is deliberately inconsistent with
# the 2-atom converged reference (-3.0 -> -6.0 per 4 atoms), like a
# Gamma-only supercell vs a k-converged primitive cell.
_DATA = dict(
    bulk=dict(formula="AlN", energy=-5.0, composition={"Al": 2, "N": 2}),
    elemental_references={"Al": 0.0, "N": 0.0},
    chemical_potentials={"Al": 0.0, "N": -3.0},  # consistent with -3.0 per AlN
)


def _with_ref(**ref):
    data = copy.deepcopy(_DATA)
    data["bulk"]["formation_reference"] = ref
    return data


def test_host_formation_reference_defaults_to_bulk():
    energy, comp = host_formation_reference(_DATA)
    assert energy == -5.0 and comp == {"Al": 2, "N": 2}


def test_host_formation_reference_prefers_explicit_field():
    data = _with_ref(energy=-3.0, composition={"Al": 1, "N": 1})
    assert host_formation_reference(data) == (-3.0, {"Al": 1, "N": 1})


def test_host_stability_check_uses_formation_reference():
    # Without the reference: -5.0/4 vs mu sum -6.0/4 -> 0.25 eV/atom off -> warns.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_chemical_potentials(copy.deepcopy(_DATA))
    assert [w for w in caught if "Host stability" in str(w.message)]
    # With a consistent converged reference: no host-stability warning.
    data = _with_ref(energy=-3.0, composition={"Al": 1, "N": 1})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_chemical_potentials(data)
    assert not [w for w in caught if "Host stability" in str(w.message)]


def test_stability_summary_uses_formation_reference_enthalpy():
    data = _with_ref(energy=-3.0, composition={"Al": 1, "N": 1})
    result = stability_summary(data)
    assert result["dH_f_host"] == pytest.approx(-3.0)
    assert result["feasible"] is True
    assert stability_summary(copy.deepcopy(_DATA))["feasible"] is False


def test_ternary_polygon_uses_formation_reference():
    # Host HHeLi (refs 0): bulk supercell energy is unusable (-1.0 per
    # 2 f.u.), the converged reference gives dH_f = -3.0 per f.u.
    data = dict(
        bulk=dict(
            formula="HHeLi",
            energy=-1.0,
            composition={"H": 2, "He": 2, "Li": 2},
            formation_reference=dict(
                energy=-3.0, composition={"H": 1, "He": 1, "Li": 1}
            ),
        ),
        elemental_references={"H": 0.0, "He": 0.0, "Li": 0.0},
        chemical_potentials={"H": -1.0, "He": -1.0, "Li": -1.0},
    )
    result = stability_summary(data)
    xs = sorted(v[0] for v in result["region_vertices"])
    assert result["feasible"] is True
    assert xs[0] == pytest.approx(-3.0)  # triangle corner reaches dH_f = -3.0


def _write(tmp_path, mutate):
    raw = yaml.safe_load(EXAMPLE_INPUT.read_text(encoding="utf-8"))
    mutate(raw)
    p = tmp_path / "in.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def test_load_input_accepts_proportional_composition(tmp_path):
    def mutate(raw):
        raw["bulk"]["formation_reference"] = {
            "energy": raw["bulk"]["energy"] / 18,
            "composition": {"Ga": 2, "N": 2},
        }

    load_input(_write(tmp_path, mutate))


@pytest.mark.parametrize(
    "ref,match",
    [
        ({"energy": -10.0, "composition": {"Ga": 2, "N": 1}}, "stoichiometr"),
        ({"energy": -10.0, "composition": {"Ga": 1, "N": 1, "O": 1}}, "species"),
        ({"energy": "x", "composition": {"Ga": 1, "N": 1}}, "numeric"),
        ({"composition": {"Ga": 1, "N": 1}}, "energy"),
        ({"energy": -10.0}, "composition"),
    ],
)
def test_load_input_rejects_malformed_formation_reference(tmp_path, ref, match):
    def mutate(raw):
        raw["bulk"]["formation_reference"] = ref

    with pytest.raises((ValueError, KeyError), match=match):
        load_input(_write(tmp_path, mutate))


def test_formation_reference_does_not_change_defect_formation_energies(tmp_path):
    """E_def - E_bulk must keep using bulk.energy: adding a (consistent)
    formation_reference leaves every formation energy unchanged."""

    def mutate(raw):
        raw["bulk"]["formation_reference"] = {
            "energy": raw["bulk"]["energy"] / 36,
            "composition": {"Ga": 1, "N": 1},
        }

    base = tmp_path / "base"
    ref = tmp_path / "ref"
    assert run_cli(["-i", str(EXAMPLE_INPUT), "--out-dir", str(base)]).returncode == 0
    assert (
        run_cli(["-i", str(_write(tmp_path, mutate)), "--out-dir", str(ref)]).returncode
        == 0
    )
    for name in ("V_Ga_ef_vs_ef.txt", "V_N_ef_vs_ef.txt"):
        assert (base / "ef_vs_ef" / name).read_text() == (
            ref / "ef_vs_ef" / name
        ).read_text()
