"""CLI regression tests: every pre-existing flag/dest/default must be
unchanged (requirement: full backward CLI compatibility), and the new
flags introduced by this refactor must have defaults that reproduce the
original behaviour unless explicitly given."""

import sys

import pytest

from conftest import EXAMPLE_INPUT, run_cli
from ikazuchix.cli import parse_args

# (flag, dest, default) for every flag that existed in the original
# monolithic script -- must never change.
_ORIGINAL_FLAGS = [
    ("-i", "input", None),  # required, no meaningful default
    ("--out-dir", "out_dir", "."),
    ("-o", "output", "results.txt"),
    ("-f", "fermi", None),
    ("--fermi-min", "fermi_min", None),
    ("--fermi-max", "fermi_max", None),
    ("-p", "plot", False),
    ("--comparison-plot", "comparison_plot", "comparison_plot.png"),
    ("-v", "verbose", False),
    ("--dry-run", "dry_run", False),
]

# (flag, dest, default) for every flag added by this refactor -- must
# default to values that reproduce the pre-refactor numerics exactly.
_NEW_FLAGS = [
    ("--real-cutoff", "real_cutoff", 3),
    ("--recip-cutoff", "recip_cutoff", 3),
    ("--ewald-eta", "ewald_eta", None),
    ("--adaptive-ewald", "adaptive_ewald", False),
    ("--ewald-tol", "ewald_tol", None),
    ("--max-ewald-shells", "max_ewald_shells", 15),
    ("--no-mpi", "no_mpi", False),
    ("--no-numba", "no_numba", False),
    # Feat-1 (self-consistent Fermi level / concentrations, opt-in):
    ("--solve-fermi", "solve_fermi", False),
    ("--temperature", "temperature", 300.0),
    ("--no-carriers", "no_carriers", False),
    ("--fermi-solve-tol", "fermi_solve_tol", 1e-8),
    # Feat-2 (chemical potential stability region, opt-in):
    ("--stability-diagram", "stability_diagram", False),
    ("--stability-plot", "stability_plot", "stability_region.png"),
]


def _parse(argv):
    old_argv = sys.argv
    sys.argv = ["ikazuchix"] + argv
    try:
        return parse_args()
    finally:
        sys.argv = old_argv


@pytest.mark.parametrize("flag,dest,default", _ORIGINAL_FLAGS)
def test_original_flag_default_unchanged(flag, dest, default):
    args = _parse(["-i", "dummy.yaml"])
    assert hasattr(args, dest), f"flag {flag} (dest={dest}) no longer exists"
    if dest != "input":
        assert getattr(args, dest) == default


@pytest.mark.parametrize("flag,dest,default", _NEW_FLAGS)
def test_new_flag_default_matches_original_behaviour(flag, dest, default):
    args = _parse(["-i", "dummy.yaml"])
    assert hasattr(args, dest)
    assert getattr(args, dest) == default


def test_input_is_required():
    old_argv = sys.argv
    sys.argv = ["ikazuchix"]
    try:
        with pytest.raises(SystemExit):
            parse_args()
    finally:
        sys.argv = old_argv


def test_help_exits_cleanly():
    result = run_cli(["--help"])
    assert result.returncode == 0
    assert "-i FILE, --input FILE" in result.stdout or "--input" in result.stdout
    for flag in ("--adaptive-ewald", "--no-mpi", "--no-numba",
                 "--real-cutoff", "--recip-cutoff"):
        assert flag in result.stdout


def test_no_new_flag_collides_with_an_existing_dest():
    all_dests = [dest for _, dest, _ in _ORIGINAL_FLAGS + _NEW_FLAGS]
    assert len(set(all_dests)) == len(all_dests)  # every dest is unique


# ===========================================================================
# Fix-3: positivity / ordering validation
# ===========================================================================

@pytest.mark.parametrize("flag", ["--real-cutoff", "--recip-cutoff",
                                   "--max-ewald-shells"])
@pytest.mark.parametrize("bad_value", ["0", "-1"])
def test_non_positive_ewald_shell_count_exits_with_error(flag, bad_value, capsys):
    with pytest.raises(SystemExit):
        _parse(["-i", "dummy.yaml", flag, bad_value])
    assert flag in capsys.readouterr().err


def test_ewald_eta_override_shown_in_report(tmp_path):
    """--ewald-eta must be reflected in the written report -- previously
    results.txt always displayed the auto-heuristic eta =
    sqrt(pi)/V^(1/3), mislabeled as if it were the value actually used,
    even when --ewald-eta had overridden it for the KO calculation
    itself."""
    out_dir = tmp_path / "eta_override_out"
    result = run_cli([
        "-i", str(EXAMPLE_INPUT), "--out-dir", str(out_dir),
        "--ewald-eta", "0.777777",
    ])
    assert result.returncode == 0, result.stderr
    text = (out_dir / "results.txt").read_text(encoding="utf-8")
    assert "Ewald eta (--ewald-eta)" in text
    assert "0.777777" in text
    assert "Ewald eta (auto)" not in text


def test_fermi_min_greater_than_fermi_max_exits_with_error():
    result = run_cli([
        "-i", str(EXAMPLE_INPUT), "--fermi-min", "2.0", "--fermi-max", "1.0",
        "--dry-run",
    ])
    assert result.returncode != 0
    assert "fermi" in result.stderr.lower()


def test_temperature_non_positive_exits_with_error():
    result = run_cli(["-i", "dummy.yaml", "--temperature", "0"])
    assert result.returncode != 0
    assert "--temperature" in result.stderr


def test_fermi_solve_tol_non_positive_exits_with_error():
    result = run_cli(["-i", "dummy.yaml", "--fermi-solve-tol", "-1"])
    assert result.returncode != 0
    assert "--fermi-solve-tol" in result.stderr


# ===========================================================================
# Feat-1: --solve-fermi end-to-end smoke test
# ===========================================================================

def test_solve_fermi_end_to_end_writes_equilibrium_summary(tmp_path):
    out_dir = tmp_path / "solve_fermi_out"
    result = run_cli([
        "-i", str(EXAMPLE_INPUT), "--out-dir", str(out_dir),
        "--solve-fermi", "--temperature", "300",
    ])
    assert result.returncode == 0, result.stderr
    summary = out_dir / "equilibrium_summary.txt"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "Solved Fermi level" in text
    assert "Electron concentration" in text


def test_solve_fermi_respects_explicit_fermi_min_max():
    """--fermi-min/--fermi-max must actually bound --solve-fermi's
    bisection bracket, as documented (README / their own --help text)
    -- not be silently ignored in favour of solve_equilibrium's
    internal [0, gap] default. A narrow range that does not bracket the
    (known, from the unconstrained solve) root must now be rejected."""
    solved = run_cli(["-i", str(EXAMPLE_INPUT), "--solve-fermi"])
    assert solved.returncode == 0, solved.stderr
    assert "+0.88" in solved.stdout  # sanity: known root for this fixture

    narrowed = run_cli([
        "-i", str(EXAMPLE_INPUT), "--solve-fermi",
        "--fermi-min", "0.1", "--fermi-max", "0.2",
    ])
    assert narrowed.returncode != 0
    assert "0.1000" in narrowed.stderr and "0.2000" in narrowed.stderr


def test_solve_fermi_default_bracket_unaffected_when_flags_omitted(tmp_path):
    """Regression guard: omitting --fermi-min/--fermi-max must keep
    using solve_equilibrium's own [0, gap] default (unchanged), not the
    wider, fermi_ref-dependent auto sweep range computed for the
    E_f(q, E_F) table/plot."""
    out_dir = tmp_path / "solve_fermi_default"
    result = run_cli([
        "-i", str(EXAMPLE_INPUT), "--out-dir", str(out_dir), "--solve-fermi",
    ])
    assert result.returncode == 0, result.stderr
    assert "+0.88" in result.stdout


# ===========================================================================
# Feat-2: --stability-diagram end-to-end smoke test
# ===========================================================================

def test_stability_diagram_end_to_end_writes_stability_summary(tmp_path):
    out_dir = tmp_path / "stability_out"
    result = run_cli([
        "-i", str(EXAMPLE_INPUT), "--out-dir", str(out_dir),
        "--stability-diagram",
    ])
    assert result.returncode == 0, result.stderr
    summary = out_dir / "stability_summary.txt"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "stability region summary" in text
    assert "Host species" in text
    # example_input.yaml's host (GaN) is binary, not ternary: no 2D region.
    assert "only computed for a ternary" in text
