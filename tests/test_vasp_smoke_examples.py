"""Regression tests on the real (small-cell, unconverged) VASP-derived
example inputs: examples/AlN_vasp_smoke.yaml (binary host, docs/adr/0003)
and examples/MgAl2O4_vasp_smoke.yaml (ternary host, docs/adr/0008).

The numbers are not physical results; they pin down that ikazuchix
keeps handling real DFT data correctly -- in particular the ternary
stability polygon, checked against an independent vertex enumeration.
"""

import contextlib
import io
import itertools
import math
import warnings

import numpy as np
import pytest
from conftest import REPO_ROOT, requires_matplotlib, run_cli

from ikazuchix.chem_potentials import check_chemical_potentials
from ikazuchix.constants import COULOMB_PREFACTOR
from ikazuchix.ewald import KumagaiObaCorrectionCalculator
from ikazuchix.geometry import build_dielectric_tensor, build_lattice_matrix
from ikazuchix.io_utils import load_input
from ikazuchix.stability_region import is_feasible_point, stability_summary

ALN = REPO_ROOT / "examples" / "AlN_vasp_smoke.yaml"
SPINEL = REPO_ROOT / "examples" / "MgAl2O4_vasp_smoke.yaml"

# Polygon vertices (delta_mu_O, delta_mu_Mg) [eV] at extraction time.
SPINEL_VERTICES = [
    (-5.153576, -0.312361),
    (-5.024733, -0.827731),
    (0.0, -5.852464),
    (0.0, -5.465937),
]


def _load(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_input(path)


def _dH(data, energy, comp):
    refs = data["elemental_references"]
    return energy - sum(n * refs[sp] for sp, n in comp.items())


def _independent_vertices(data):
    """All pairwise intersections of the constraint lines that satisfy
    every constraint -- an enumeration independent of the production
    Sutherland-Hodgman clipping."""
    fr = data["bulk"]["formation_reference"]
    comp = fr["composition"]
    dH = _dH(data, fr["energy"], comp)
    a, b, c = comp["O"], comp["Mg"], comp["Al"]  # x = O, y = Mg, Al eliminated
    rows = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (-a, -b, -dH)]  # x<=0, y<=0, Al<=0
    for p in data["competing_phases"]:
        n = p["composition"]
        dHp = _dH(data, p["energy"], n)
        nO, nMg, nAl = n.get("O", 0), n.get("Mg", 0), n.get("Al", 0)
        rows.append((nO - nAl * a / c, nMg - nAl * b / c, dHp - nAl * dH / c))
    pts = set()
    for (a1, b1, r1), (a2, b2, r2) in itertools.combinations(rows, 2):
        M = np.array([[a1, b1], [a2, b2]])
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x, y = np.linalg.solve(M, [r1, r2])
        if all(ax * x + ay * y <= r + 1e-9 for ax, ay, r in rows):
            pts.add((round(float(x), 6) + 0.0, round(float(y), 6) + 0.0))
    return sorted(pts)


def test_spinel_axes_are_ordered_by_atomic_number():
    r = stability_summary(_load(SPINEL))
    assert (r["region_species_x"], r["region_species_y"], r["region_species_z"]) == (
        "O",
        "Mg",
        "Al",
    )


def test_spinel_polygon_matches_independent_enumeration_and_pinned_values():
    data = _load(SPINEL)
    got = sorted(
        (round(x, 6) + 0.0, round(y, 6) + 0.0)
        for x, y in stability_summary(data)["region_vertices"]
    )
    indep = _independent_vertices(data)
    assert len(got) == len(indep) == 4
    for (gx, gy), (ix, iy), (px, py) in zip(got, indep, sorted(SPINEL_VERTICES)):
        assert gx == pytest.approx(ix, abs=1e-6) and gy == pytest.approx(iy, abs=1e-6)
        assert gx == pytest.approx(px, abs=1e-5) and gy == pytest.approx(py, abs=1e-5)


def test_spinel_vertices_and_centroid_feasible_outside_point_not():
    data = _load(SPINEL)
    fr = data["bulk"]["formation_reference"]
    comp = fr["composition"]
    dH = _dH(data, fr["energy"], comp)
    refs = data["elemental_references"]
    phases = data["competing_phases"]

    def point(x, y):
        return {
            "O": x,
            "Mg": y,
            "Al": (dH - comp["O"] * x - comp["Mg"] * y) / comp["Al"],
        }

    for x, y in SPINEL_VERTICES:
        ok, violations = is_feasible_point(
            point(x, y), comp, refs, dH, phases, tol=1e-5
        )
        assert ok, violations
    assert stability_summary(data)["feasible"] is True  # the file's own centroid
    ok, violations = is_feasible_point(point(0.0, 0.0), comp, refs, dH, phases)
    assert not ok
    assert len(violations) == 1 and "MgO" in violations[0]


@pytest.mark.parametrize("path", [ALN, SPINEL], ids=["AlN", "MgAl2O4"])
def test_formation_reference_passes_host_stability(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            check_chemical_potentials(_load(path))
    assert not [w for w in caught if "Host stability" in str(w.message)]


def test_aln_formation_enthalpy_from_primitive_cell():
    data = _load(ALN)
    fr = data["bulk"]["formation_reference"]
    assert _dH(data, fr["energy"], fr["composition"]) / 2 == pytest.approx(
        -2.812670, abs=1e-5
    )


def test_spinel_ko_matches_simple_cubic_madelung():
    """The 56-atom spinel supercell is simple cubic and its DFPT tensor
    isotropic, so E_PC(q=1) must equal -K*alpha/(2*eps*L), alpha = 2.837297."""
    data = _load(SPINEL)
    L = build_lattice_matrix(data["supercell"])
    eps = build_dielectric_tensor(data["dielectric"])
    side = float(np.linalg.norm(L[0]))
    ko = KumagaiObaCorrectionCalculator(L, eps, 1, [0.5, 0.5, 0.5], 0.0).compute()
    expected = -COULOMB_PREFACTOR * 2.837297 / (2.0 * eps[0, 0] * side)
    assert ko["E_PC"] == pytest.approx(expected, rel=1e-5)
    assert math.isclose(eps[0, 0], eps[2, 2]) and eps[0, 1] == 0.0


@requires_matplotlib
def test_spinel_end_to_end_cli(tmp_path):
    result = run_cli(
        [
            "-i",
            str(SPINEL),
            "--out-dir",
            str(tmp_path),
            "--solve-fermi",
            "--stability-diagram",
            "-p",
        ]
    )
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr  # Ewald converged at defaults
    summary = (tmp_path / "stability_summary.txt").read_text(encoding="utf-8")
    assert "feasible  : True" in summary
    assert (tmp_path / "plots" / "stability_region.png").stat().st_size > 0
    assert "Solved Fermi level" in (tmp_path / "equilibrium_summary.txt").read_text()


def test_stability_summary_written_before_solve_fermi_fails(tmp_path):
    """F6 (vasp_smoke/REPORT.md): at an infeasible chemical-potential
    point, formation energies go strongly negative and --solve-fermi
    cannot bracket a root and exits non-zero. The stability summary,
    which explains why, must already be on disk by then."""
    import yaml

    raw = yaml.safe_load(SPINEL.read_text(encoding="utf-8"))
    fr = raw["bulk"]["formation_reference"]
    refs = raw["elemental_references"]
    dH = fr["energy"] - sum(n * refs[sp] for sp, n in fr["composition"].items())
    raw["chemical_potentials"] = {"O": 0.0, "Mg": 0.0, "Al": dH / fr["composition"]["Al"]}
    inp = tmp_path / "outside.yaml"
    inp.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = tmp_path / "out"
    result = run_cli(["-i", str(inp), "--out-dir", str(out),
                      "--solve-fermi", "--stability-diagram"])
    assert result.returncode != 0
    assert "no root is bracketed" in result.stderr
    summary = (out / "stability_summary.txt").read_text(encoding="utf-8")
    assert "feasible  : False" in summary
    assert "MgO" in summary


@pytest.mark.parametrize("path,name,q", [
    (ALN, "O_N", 1), (ALN, "V_N", 1),
    (SPINEL, "Mg_Al", -1), (SPINEL, "Al_Mg", 1), (SPINEL, "V_Mg", -1), (SPINEL, "V_O", 1),
])
def test_point_charge_subtraction_recovers_neutral_offset(path, name, q):
    """docs/adr/0009 on real data: for a localized charged defect,
    subtracting the point-charge potential from the raw site-averaged
    delta_v leaves the neutral state's short-range offset -- which only
    holds with VASP's electron-potential-energy sign convention. The
    tolerance (0.1 eV) reflects the small, unconverged cells."""
    from ikazuchix.ewald import point_charge_potential

    data = _load(path)
    d = next(x for x in data["defects"] if x["name"] == name)
    dv = {cs["charge"]: cs["potential_alignment"]["delta_v"] for cs in d["charge_states"]}
    L = build_lattice_matrix(d["supercell"])
    eps = build_dielectric_tensor(data["dielectric"])
    phi = point_charge_potential(L, eps, q, d["supercell"]["defect_position"],
                                 d["alignment_sampling_sites"]).mean()
    assert dv[q] + phi == pytest.approx(dv[0], abs=0.1)
    # The opposite sign convention would miss by about 2|phi|:
    assert abs(dv[q] - phi - dv[0]) > abs(dv[q] + phi - dv[0])
