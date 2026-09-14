"""Golden-output regression tests.

Compares the current package's output against the reference output
captured (in tests/golden/) from the original, unmodified monolithic
defect_formation_energy.py running on examples/example_input.yaml.
This is the primary safety net for the ikazuchix/ package split: it must
keep reproducing the same numbers by default, regardless of whether
numba/MPI acceleration happens to be active (parametrized below) --
those change performance only, never results.
"""

import subprocess
import sys

import pytest

from conftest import (
    ENTRY_POINT_ARGS,
    EXAMPLE_INPUT,
    GOLDEN_DIR,
    REPO_ROOT,
    assert_text_numerically_equal,
    requires_matplotlib,
    requires_mpirun,
    run_cli,
)

_ACCELERATION_MODES = [
    pytest.param([], id="numba-default"),
    pytest.param(["--no-numba"], id="no-numba"),
]


def _run(tmp_path, extra_args=()):
    result = run_cli(["-i", str(EXAMPLE_INPUT), "--out-dir", str(tmp_path),
                       "-p", *extra_args])
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.parametrize("extra_args", _ACCELERATION_MODES)
def test_results_txt_matches_golden(tmp_path, extra_args):
    _run(tmp_path, extra_args)
    assert_text_numerically_equal(tmp_path / "results.txt", GOLDEN_DIR / "results.txt")


@pytest.mark.parametrize("extra_args", _ACCELERATION_MODES)
def test_ctls_summary_matches_golden(tmp_path, extra_args):
    _run(tmp_path, extra_args)
    assert_text_numerically_equal(tmp_path / "ctls_summary.txt", GOLDEN_DIR / "ctls_summary.txt")


@pytest.mark.parametrize("extra_args", _ACCELERATION_MODES)
def test_ef_vs_ef_tables_match_golden(tmp_path, extra_args):
    _run(tmp_path, extra_args)
    for name in ("V_Ga_ef_vs_ef.txt", "V_N_ef_vs_ef.txt"):
        assert_text_numerically_equal(
            tmp_path / "ef_vs_ef" / name, GOLDEN_DIR / "ef_vs_ef" / name)


@pytest.mark.parametrize("extra_args", _ACCELERATION_MODES)
def test_summary_tables_match_golden(tmp_path, extra_args):
    _run(tmp_path, extra_args)
    for name in ("V_Ga_summary.txt", "V_N_summary.txt"):
        assert_text_numerically_equal(
            tmp_path / "summary" / name, GOLDEN_DIR / "summary" / name)


@requires_matplotlib
def test_plots_are_generated(tmp_path):
    _run(tmp_path)
    for name in ("V_Ga_ef.png", "V_N_ef.png", "comparison_plot.png"):
        p = tmp_path / "plots" / name
        assert p.exists() and p.stat().st_size > 0, f"missing or empty plot: {p}"
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not a valid PNG: {p}"


@pytest.mark.parametrize("extra_args", _ACCELERATION_MODES)
def test_convergence_warning_text_matches_golden(tmp_path, extra_args):
    result = _run(tmp_path, extra_args)
    golden_stderr = (GOLDEN_DIR / "stderr.log").read_text(encoding="utf-8")
    golden_messages = [
        line.split("RuntimeWarning: ", 1)[1]
        for line in golden_stderr.splitlines()
        if "RuntimeWarning:" in line
    ]
    new_messages = [
        line.split("RuntimeWarning: ", 1)[1]
        for line in result.stderr.splitlines()
        if "RuntimeWarning:" in line
    ]
    assert new_messages == golden_messages


def test_default_invocation_without_any_new_flags_matches_golden_byte_exact(tmp_path):
    """Explicit confirmation of the central contract behind every
    Val/Fix/Perf/Feat change in this improvement pass: with NONE of the
    flags they added (--solve-fermi, --stability-diagram,
    --adaptive-ewald, ...) given, output must be byte-for-byte
    identical to the golden reference -- not just numerically close --
    and none of the new opt-in output files may be created."""
    result = run_cli(["-i", str(EXAMPLE_INPUT), "--out-dir", str(tmp_path)])
    assert result.returncode == 0, result.stderr

    for rel_path in (
        "results.txt", "ctls_summary.txt",
        "ef_vs_ef/V_Ga_ef_vs_ef.txt", "ef_vs_ef/V_N_ef_vs_ef.txt",
        "summary/V_Ga_summary.txt", "summary/V_N_summary.txt",
    ):
        got    = (tmp_path / rel_path).read_bytes()
        golden = (GOLDEN_DIR / rel_path).read_bytes()
        assert got == golden, f"{rel_path}: byte-exact mismatch vs golden"

    assert not (tmp_path / "equilibrium_summary.txt").exists()
    assert not (tmp_path / "stability_summary.txt").exists()


@requires_mpirun
@pytest.mark.parametrize("n_ranks", [1, 2, 4])
def test_results_txt_matches_golden_under_mpirun(tmp_path, n_ranks):
    cmd = ["mpirun", "-n", str(n_ranks), sys.executable, *ENTRY_POINT_ARGS,
           "-i", str(EXAMPLE_INPUT), "--out-dir", str(tmp_path)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                             text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert_text_numerically_equal(tmp_path / "results.txt", GOLDEN_DIR / "results.txt")
