"""Unit tests for ikazuchix.io_utils.load_input."""

import textwrap

import pytest
import yaml

from ikazuchix.io_utils import _atomic_write_text, load_input

_BASE = dict(
    bulk=dict(formula="XY", energy=-100.0, num_atoms=2, composition={"X": 1, "Y": 1}),
    elemental_references={"X": -1.0, "Y": -2.0},
    chemical_potentials={"X": 0.0, "Y": 0.0},
    band_structure=dict(vbm=0.0, cbm=2.0, bandgap=2.0, fermi_level=1.0),
    dielectric=dict(tensor=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    supercell=dict(
        lattice_vectors={"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]},
        defect_position=[0, 0, 0],
    ),
)


def _write_yaml(tmp_path, data, name="input.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _multi_defect(charge_kwargs=None):
    data = dict(_BASE)
    cs = dict(charge=0, energy=-50.0, num_atoms=1,
              potential_alignment={"delta_v": 0.0})
    if charge_kwargs:
        cs.update(charge_kwargs)
    data["defects"] = [dict(
        name="V_X", defect_type="vacancy", removed_atoms={"X": 1},
        charge_states=[cs],
    )]
    return data


def test_multi_defect_format_passthrough(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect())
    raw = load_input(path)
    assert len(raw["defects"]) == 1
    assert raw["defects"][0]["name"] == "V_X"
    assert raw["defects"][0]["charge_states"][0]["charge"] == 0


def test_legacy_single_defect_format_normalizes(tmp_path):
    data = dict(_BASE)
    data["defect"] = dict(name="V_X", defect_type="vacancy", removed_atoms={"X": 1})
    data["charge_states"] = [dict(charge=0, energy=-50.0, num_atoms=1,
                                   potential_alignment={"delta_v": 0.0})]
    path = _write_yaml(tmp_path, data)
    raw = load_input(path)
    assert len(raw["defects"]) == 1
    assert raw["defects"][0]["name"] == "V_X"
    assert raw["defects"][0]["charge_states"][0]["energy"] == -50.0


def test_composition_auto_completion_vacancy(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect())
    raw = load_input(path)
    comp = raw["defects"][0]["charge_states"][0]["composition"]
    assert comp == {"Y": 1}  # X: 1-1=0 (dropped), Y: 1-0=1


def test_composition_auto_completion_interstitial(tmp_path):
    data = dict(_BASE)
    cs = dict(charge=0, energy=-50.0, num_atoms=3,
              potential_alignment={"delta_v": 0.0})
    data["defects"] = [dict(
        name="X_i", defect_type="interstitial", added_atoms={"X": 1},
        charge_states=[cs],
    )]
    path = _write_yaml(tmp_path, data)
    raw = load_input(path)
    comp = raw["defects"][0]["charge_states"][0]["composition"]
    assert comp == {"X": 2, "Y": 1}


def test_explicit_composition_is_not_overwritten(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect(
        charge_kwargs={"composition": {"X": 99, "Y": 1}, "num_atoms": 100}))
    raw = load_input(path)
    comp = raw["defects"][0]["charge_states"][0]["composition"]
    assert comp == {"X": 99, "Y": 1}


@pytest.mark.parametrize("missing_key", [
    "bulk", "chemical_potentials", "elemental_references",
    "band_structure", "dielectric", "supercell",
])
def test_missing_required_key_raises_keyerror(tmp_path, missing_key):
    data = dict(_multi_defect())
    del data[missing_key]
    path = _write_yaml(tmp_path, data)
    with pytest.raises(KeyError):
        load_input(path)


def test_empty_defects_list_raises_valueerror(tmp_path):
    data = dict(_multi_defect())
    data["defects"] = []
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError):
        load_input(path)


def test_missing_defect_and_defects_raises_keyerror(tmp_path):
    data = dict(_BASE)
    path = _write_yaml(tmp_path, data)
    with pytest.raises(KeyError):
        load_input(path)


def test_num_atoms_mismatch_warns(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect(charge_kwargs={"num_atoms": 999}))
    with pytest.warns(UserWarning, match="num_atoms"):
        load_input(path)


def test_num_atoms_consistent_does_not_warn(tmp_path, recwarn):
    path = _write_yaml(tmp_path, _multi_defect())
    load_input(path)
    assert not any("num_atoms" in str(w.message) for w in recwarn.list)


def test_utf8_content_round_trips(tmp_path):
    data = dict(_multi_defect())
    data["bulk"] = dict(data["bulk"])
    data["bulk"]["formula"] = "GaN (α-phase)"  # alpha character, non-ASCII
    path = tmp_path / "input_utf8.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    raw = load_input(path)
    assert raw["bulk"]["formula"] == "GaN (α-phase)"


# ===========================================================================
# Val-1: legacy/multi-defect normalization symmetry
# ===========================================================================

def test_legacy_missing_name_raises(tmp_path):
    data = dict(_BASE)
    data["defect"] = dict(defect_type="vacancy", removed_atoms={"X": 1})
    data["charge_states"] = [dict(charge=0, energy=-50.0, num_atoms=1,
                                   potential_alignment={"delta_v": 0.0})]
    path = _write_yaml(tmp_path, data)
    with pytest.raises(KeyError):
        load_input(path)


def test_legacy_missing_defect_type_raises(tmp_path):
    data = dict(_BASE)
    data["defect"] = dict(name="V_X", removed_atoms={"X": 1})
    data["charge_states"] = [dict(charge=0, energy=-50.0, num_atoms=1,
                                   potential_alignment={"delta_v": 0.0})]
    path = _write_yaml(tmp_path, data)
    with pytest.raises(KeyError):
        load_input(path)


def test_both_defects_and_legacy_present_warns(tmp_path):
    data = _multi_defect()
    data["defect"] = dict(name="V_Y", defect_type="vacancy", removed_atoms={"Y": 1})
    data["charge_states"] = [dict(charge=0, energy=-50.0, num_atoms=1,
                                   potential_alignment={"delta_v": 0.0})]
    path = _write_yaml(tmp_path, data)
    with pytest.warns(UserWarning, match="legacy"):
        raw = load_input(path)
    assert raw["defects"][0]["name"] == "V_X"  # legacy keys ignored, not merged


def test_non_dict_top_level_raises(tmp_path):
    path = tmp_path / "input.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_input(path)


# ===========================================================================
# Val-2: nested required keys, charge normalization, duplicate detection
# ===========================================================================

def test_duplicate_charge_within_defect_raises(tmp_path):
    data = dict(_BASE)
    cs1 = dict(charge=0, energy=-50.0, num_atoms=1,
               potential_alignment={"delta_v": 0.0})
    cs2 = dict(charge=0, energy=-50.5, num_atoms=1,
               potential_alignment={"delta_v": 0.0})
    data["defects"] = [dict(
        name="V_X", defect_type="vacancy", removed_atoms={"X": 1},
        charge_states=[cs1, cs2],
    )]
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="duplicate"):
        load_input(path)


def test_charge_fractional_float_raises(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect(charge_kwargs={"charge": 0.5}))
    with pytest.raises(ValueError):
        load_input(path)


def test_charge_numeric_string_is_coerced(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect(charge_kwargs={"charge": "-1"}))
    raw = load_input(path)
    q = raw["defects"][0]["charge_states"][0]["charge"]
    assert q == -1
    assert isinstance(q, int)


def test_charge_state_missing_energy_raises(tmp_path):
    data = _multi_defect()
    del data["defects"][0]["charge_states"][0]["energy"]
    path = _write_yaml(tmp_path, data)
    with pytest.raises(KeyError, match="energy"):
        load_input(path)


def test_charge_state_non_numeric_energy_raises(tmp_path):
    path = _write_yaml(
        tmp_path, _multi_defect(charge_kwargs={"energy": "not_a_number"}))
    with pytest.raises(ValueError, match="energy"):
        load_input(path)


def test_charge_state_explicit_composition_negative_value_raises(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect(
        charge_kwargs={"composition": {"X": -1, "Y": 1}}))
    with pytest.raises(ValueError, match="non-negative"):
        load_input(path)


def test_removed_atoms_exceeding_bulk_raises(tmp_path):
    """Removing more atoms of a species than bulk.composition contains
    produced a negative auto-completed atom count, which then fed
    chem_pot_correction for every charge state -- silently, since the
    only signal was a num_atoms warning that is skipped entirely when
    'num_atoms' is omitted."""
    data = _multi_defect()
    data["defects"][0]["removed_atoms"] = {"X": 5}  # bulk has X: 1
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="negative atom count"):
        load_input(path)


def test_negative_removed_atoms_raises(tmp_path):
    """A '-' typo in removed_atoms silently ADDED an atom instead of
    removing one."""
    data = _multi_defect()
    data["defects"][0]["removed_atoms"] = {"X": -1}
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="non-negative"):
        load_input(path)


def test_negative_added_atoms_raises(tmp_path):
    data = _multi_defect()
    data["defects"][0]["added_atoms"] = {"X": -1}
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="non-negative"):
        load_input(path)


def test_auto_completed_composition_key_order_is_deterministic(tmp_path):
    """The auto-completed composition's key order becomes the order
    chem_pot_correction accumulates floats in, so it must be sorted
    rather than inherited from a set's (hash-randomized) iteration
    order."""
    data = dict(_BASE)
    data["bulk"] = dict(formula="XYZ", energy=-100.0,
                        composition={"Z": 2, "X": 3, "Y": 1})
    data["elemental_references"] = {"X": -1.0, "Y": -2.0, "Z": -3.0}
    data["chemical_potentials"] = {"X": 0.0, "Y": 0.0, "Z": 0.0}
    data["defects"] = [dict(
        name="V_X", defect_type="vacancy", removed_atoms={"X": 1},
        charge_states=[dict(charge=0, energy=-50.0,
                            potential_alignment={"delta_v": 0.0})],
    )]
    path = _write_yaml(tmp_path, data)
    comp = load_input(path)["defects"][0]["charge_states"][0]["composition"]
    assert list(comp) == sorted(comp)


def test_charge_state_explicit_composition_non_dict_raises(tmp_path):
    path = _write_yaml(
        tmp_path, _multi_defect(charge_kwargs={"composition": ["X", "Y"]}))
    with pytest.raises(ValueError, match="composition"):
        load_input(path)


def test_cbm_less_than_vbm_raises(tmp_path):
    data = dict(_multi_defect())
    data["band_structure"] = dict(vbm=2.0, cbm=1.0, bandgap=-1.0, fermi_level=1.5)
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="cbm"):
        load_input(path)


def test_band_structure_missing_key_raises(tmp_path):
    data = dict(_multi_defect())
    bs = dict(data["band_structure"])
    del bs["cbm"]
    data["band_structure"] = bs
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="band_structure"):
        load_input(path)


def test_empty_bulk_composition_raises(tmp_path):
    data = dict(_multi_defect())
    data["bulk"] = dict(data["bulk"])
    data["bulk"]["composition"] = {}
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="composition"):
        load_input(path)


def test_bulk_composition_negative_value_raises(tmp_path):
    """A sign typo in 'bulk.composition' (e.g. '-1' instead of '1') must
    be rejected at load time -- previously it propagated silently into
    every auto-completed defect composition (see load_input's
    auto-completion loop), corrupting chem_pot_correction() for every
    charge state without ever raising or warning."""
    data = dict(_multi_defect())
    data["bulk"] = dict(data["bulk"])
    data["bulk"]["composition"] = {"X": -1, "Y": 1}
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="non-negative"):
        load_input(path)


def test_bulk_missing_energy_raises(tmp_path):
    data = dict(_multi_defect())
    data["bulk"] = dict(data["bulk"])
    del data["bulk"]["energy"]
    path = _write_yaml(tmp_path, data)
    with pytest.raises(KeyError, match="energy"):
        load_input(path)


def test_bulk_non_numeric_energy_raises(tmp_path):
    data = dict(_multi_defect())
    data["bulk"] = dict(data["bulk"])
    data["bulk"]["energy"] = "not_a_number"
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="energy"):
        load_input(path)


def test_bulk_missing_formula_raises(tmp_path):
    data = dict(_multi_defect())
    data["bulk"] = dict(data["bulk"])
    del data["bulk"]["formula"]
    path = _write_yaml(tmp_path, data)
    with pytest.raises(KeyError, match="formula"):
        load_input(path)


def test_bulk_blank_formula_raises(tmp_path):
    data = dict(_multi_defect())
    data["bulk"] = dict(data["bulk"])
    data["bulk"]["formula"] = "   "
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="formula"):
        load_input(path)


def test_dielectric_tensor_wrong_shape_raises(tmp_path):
    data = dict(_multi_defect())
    data["dielectric"] = dict(tensor=[[1, 0], [0, 1]])
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="dielectric"):
        load_input(path)


def test_supercell_missing_defect_position_raises(tmp_path):
    data = dict(_multi_defect())
    sc = dict(data["supercell"])
    del sc["defect_position"]
    data["supercell"] = sc
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="defect_position"):
        load_input(path)


def test_defect_supercell_override_missing_lattice_vectors_raises(tmp_path):
    data = dict(_multi_defect())
    data["defects"][0]["supercell"] = dict(defect_position=[0, 0, 0])
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="lattice_vectors"):
        load_input(path)


# ===========================================================================
# Val-3: potential_alignment validation, removed/added-both-missing warning
# ===========================================================================

def test_potential_alignment_non_dict_raises(tmp_path):
    path = _write_yaml(
        tmp_path, _multi_defect(charge_kwargs={"potential_alignment": 0.0}))
    with pytest.raises(ValueError, match="potential_alignment"):
        load_input(path)


def test_potential_alignment_delta_v_null_raises(tmp_path):
    path = _write_yaml(
        tmp_path,
        _multi_defect(charge_kwargs={"potential_alignment": {"delta_v": None}}))
    with pytest.raises(ValueError, match="delta_v"):
        load_input(path)


def test_removed_and_added_both_missing_warns(tmp_path):
    data = dict(_BASE)
    cs = dict(charge=0, energy=-50.0, num_atoms=2,
              potential_alignment={"delta_v": 0.0})
    data["defects"] = [dict(name="Perfect", defect_type="vacancy",
                             charge_states=[cs])]
    path = _write_yaml(tmp_path, data)
    with pytest.warns(UserWarning, match="removed_atoms"):
        load_input(path)


# ===========================================================================
# Fix-2b: atomic writes
# ===========================================================================

def test_atomic_write_text_writes_content_and_leaves_no_temp_file(tmp_path):
    path = tmp_path / "out.txt"
    _atomic_write_text(str(path), "hello\nworld\n")
    assert path.read_text(encoding="utf-8") == "hello\nworld\n"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_write_text_overwrites_existing_file(tmp_path):
    path = tmp_path / "out.txt"
    path.write_text("old content", encoding="utf-8")
    _atomic_write_text(str(path), "new content")
    assert path.read_text(encoding="utf-8") == "new content"


# ===========================================================================
# Feat-1/Feat-2: optional new YAML fields
# ===========================================================================

def test_effective_mass_positive_is_accepted(tmp_path):
    data = dict(_multi_defect())
    data["band_structure"] = dict(data["band_structure"])
    data["band_structure"]["effective_mass_e"] = 0.2
    path = _write_yaml(tmp_path, data)
    raw = load_input(path)
    assert raw["band_structure"]["effective_mass_e"] == 0.2


def test_effective_mass_non_positive_raises(tmp_path):
    data = dict(_multi_defect())
    data["band_structure"] = dict(data["band_structure"])
    data["band_structure"]["effective_mass_h"] = -0.5
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="effective_mass_h"):
        load_input(path)


def test_degeneracy_non_positive_raises(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect(charge_kwargs={"degeneracy": 0}))
    with pytest.raises(ValueError, match="degeneracy"):
        load_input(path)


def test_degeneracy_positive_is_accepted(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect(charge_kwargs={"degeneracy": 2}))
    raw = load_input(path)
    assert raw["defects"][0]["charge_states"][0]["degeneracy"] == 2


def test_site_density_non_positive_raises(tmp_path):
    data = dict(_multi_defect())
    data["defects"][0]["site_density"] = -1e19
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="site_density"):
        load_input(path)


def test_competing_phases_default_empty_no_validation_triggered(tmp_path):
    path = _write_yaml(tmp_path, _multi_defect())
    raw = load_input(path)
    assert raw.get("competing_phases") in (None, [])


def test_competing_phases_missing_key_raises(tmp_path):
    data = dict(_multi_defect())
    data["competing_phases"] = [dict(formula="X2", energy=-1.0)]  # no composition
    path = _write_yaml(tmp_path, data)
    with pytest.raises(KeyError):
        load_input(path)


def test_competing_phases_non_numeric_energy_raises(tmp_path):
    data = dict(_multi_defect())
    data["competing_phases"] = [
        dict(formula="X2", energy="not_a_number", composition={"X": 2})]
    path = _write_yaml(tmp_path, data)
    with pytest.raises(ValueError, match="energy"):
        load_input(path)


def test_competing_phases_valid_entry_is_accepted(tmp_path):
    data = dict(_multi_defect())
    data["competing_phases"] = [
        dict(formula="X2", energy=-1.0, composition={"X": 2})]
    path = _write_yaml(tmp_path, data)
    raw = load_input(path)
    assert raw["competing_phases"][0]["formula"] == "X2"
