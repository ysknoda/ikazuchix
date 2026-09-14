"""Tests for ikazuchix.plotting: PNG smoke tests + the matplotlib-optional
fallback path (mirroring the existing pattern this refactor is asked
to replicate for numba/mpi4py -- see ikazuchix/plotting.py's try/except)."""

import importlib
import sys
import warnings

import numpy as np
import pytest
import yaml

from conftest import requires_matplotlib
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
def result(data):
    grid = np.linspace(-0.17, 1.87, 400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return run_one_defect(data["defects"][0], data, 0.85, grid)


def _assert_valid_png(path):
    assert path.exists() and path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@requires_matplotlib
def test_plot_single_defect_creates_valid_png(data, result, tmp_path):
    from ikazuchix.plotting import plot_single_defect
    out = tmp_path / "V_Ga_ef.png"
    plot_single_defect(data, result, fermi_ref=0.85, out_path=out)
    _assert_valid_png(out)


@requires_matplotlib
def test_plot_comparison_creates_valid_png(data, result, tmp_path):
    from ikazuchix.plotting import plot_comparison
    out = tmp_path / "comparison.png"
    plot_comparison(data, [result], fermi_ref=0.85, out_path=out)
    _assert_valid_png(out)


def _reload_plotting_without_matplotlib(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    import ikazuchix.plotting as plotting
    return importlib.reload(plotting)


def test_plot_single_defect_skips_gracefully_without_matplotlib(
        data, result, tmp_path, monkeypatch, capsys):
    plotting = _reload_plotting_without_matplotlib(monkeypatch)
    try:
        out = tmp_path / "should_not_exist.png"
        plotting.plot_single_defect(data, result, fermi_ref=0.85, out_path=out)
        assert not out.exists()
        assert "matplotlib not installed" in capsys.readouterr().out
    finally:
        importlib.reload(plotting)  # restore real matplotlib-backed state


def test_plot_comparison_skips_gracefully_without_matplotlib(
        data, result, tmp_path, monkeypatch, capsys):
    plotting = _reload_plotting_without_matplotlib(monkeypatch)
    try:
        out = tmp_path / "should_not_exist.png"
        plotting.plot_comparison(data, [result], fermi_ref=0.85, out_path=out)
        assert not out.exists()
        assert "matplotlib not installed" in capsys.readouterr().out
    finally:
        importlib.reload(plotting)


# ===========================================================================
# Feat-2: plot_stability_region
# ===========================================================================

_TERNARY_STABILITY_RESULT = dict(
    species=["A", "B", "C"], point_deltas={"A": -0.5, "B": -0.5, "C": -2.0},
    feasible=True, violations=[], dH_f_host=-3.0, n_competing_phases=1,
    region_species_x="A", region_species_y="B", region_species_z="C",
    region_vertices=[(-1.0, 0.0), (-3.0, 0.0), (0.0, -3.0), (0.0, -1.0)],
)
_BINARY_STABILITY_RESULT = dict(
    species=["Ga", "N"], point_deltas={"Ga": 0.0, "N": -1.5},
    feasible=True, violations=[], dH_f_host=-54.0, n_competing_phases=0,
)


@requires_matplotlib
def test_plot_stability_region_creates_valid_png_for_ternary(tmp_path):
    from ikazuchix.plotting import plot_stability_region
    out = tmp_path / "stability.png"
    plot_stability_region(dict(bulk=dict(formula="ABC")),
                           _TERNARY_STABILITY_RESULT, out)
    _assert_valid_png(out)


@requires_matplotlib
def test_plot_stability_region_creates_valid_png_for_empty_region(tmp_path):
    from ikazuchix.plotting import plot_stability_region
    out = tmp_path / "stability_empty.png"
    empty_result = dict(_TERNARY_STABILITY_RESULT, feasible=False,
                         violations=["competing phase 'AB': ..."],
                         region_vertices=[])
    plot_stability_region({}, empty_result, out)
    _assert_valid_png(out)


@requires_matplotlib
def test_plot_stability_region_skips_for_non_ternary_host(tmp_path, capsys):
    from ikazuchix.plotting import plot_stability_region
    out = tmp_path / "should_not_exist.png"
    plot_stability_region({}, _BINARY_STABILITY_RESULT, out)
    assert not out.exists()
    assert "ternary" in capsys.readouterr().out.lower()


def test_plot_stability_region_skips_gracefully_without_matplotlib(
        tmp_path, monkeypatch, capsys):
    plotting = _reload_plotting_without_matplotlib(monkeypatch)
    try:
        out = tmp_path / "should_not_exist.png"
        plotting.plot_stability_region({}, _TERNARY_STABILITY_RESULT, out)
        assert not out.exists()
        assert "matplotlib not installed" in capsys.readouterr().out
    finally:
        importlib.reload(plotting)
