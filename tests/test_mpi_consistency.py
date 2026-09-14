"""Real multi-process MPI consistency tests.

Runs the actual CLI (as a subprocess) under `mpirun -n N` for several
values of N and checks the output files are identical to the
single-process golden reference. This is the only way to genuinely
exercise multi-rank behaviour from pytest (pytest itself is always one
process); the in-process rank/size-invariance check lives in
test_runner.py and does not require mpirun to be installed.

Skipped entirely when no mpirun/mpiexec launcher is on PATH (e.g. this
project's dependencies are "works with or without MPI" by design -- see
ikazuchix/mpi_utils.py), so the suite stays green without a system MPI
runtime installed.
"""

import subprocess
import sys

import pytest
import yaml

from conftest import (
    ENTRY_POINT_ARGS,
    EXAMPLE_INPUT,
    GOLDEN_DIR,
    REPO_ROOT,
    assert_text_numerically_equal,
    requires_matplotlib,
    requires_mpirun,
)


def _run_under_mpirun(n_ranks, out_dir, extra_args=()):
    cmd = ["mpirun", "-n", str(n_ranks), sys.executable, *ENTRY_POINT_ARGS,
           "-i", str(EXAMPLE_INPUT), "--out-dir", str(out_dir), *extra_args]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                           text=True, timeout=180)


@requires_mpirun
@pytest.mark.parametrize("n_ranks", [1, 2, 4])
def test_mpirun_output_matches_golden(tmp_path, n_ranks):
    out_dir = tmp_path / f"n{n_ranks}"
    result = _run_under_mpirun(n_ranks, out_dir)
    assert result.returncode == 0, result.stderr

    assert_text_numerically_equal(out_dir / "results.txt", GOLDEN_DIR / "results.txt")
    assert_text_numerically_equal(out_dir / "ctls_summary.txt", GOLDEN_DIR / "ctls_summary.txt")
    for name in ("V_Ga_ef_vs_ef.txt", "V_N_ef_vs_ef.txt"):
        assert_text_numerically_equal(
            out_dir / "ef_vs_ef" / name, GOLDEN_DIR / "ef_vs_ef" / name)
    for name in ("V_Ga_summary.txt", "V_N_summary.txt"):
        assert_text_numerically_equal(
            out_dir / "summary" / name, GOLDEN_DIR / "summary" / name)


@requires_mpirun
def test_mpirun_more_ranks_than_work_items(tmp_path):
    # example_input.yaml has 5 total (defect, charge_state) work items
    # (V_Ga: 3, V_N: 2); n_ranks=8 leaves several ranks with no work at
    # all, which must still complete cleanly and match golden.
    out_dir = tmp_path / "n8"
    result = _run_under_mpirun(8, out_dir)
    assert result.returncode == 0, result.stderr
    assert_text_numerically_equal(out_dir / "results.txt", GOLDEN_DIR / "results.txt")


@requires_mpirun
def test_mpirun_output_not_duplicated_across_ranks(tmp_path):
    """Rank-0-only I/O: stdout line count and the set of distinct
    RuntimeWarning messages on stderr must be independent of rank count
    (no duplicated [INFO] prints, no lost/duplicated warnings)."""
    result_1 = _run_under_mpirun(1, tmp_path / "n1")
    result_4 = _run_under_mpirun(4, tmp_path / "n4")
    assert result_1.returncode == 0 and result_4.returncode == 0

    assert len(result_1.stdout.splitlines()) == len(result_4.stdout.splitlines())

    def _warning_messages(stderr_text):
        return sorted(
            line.split("RuntimeWarning: ", 1)[1]
            for line in stderr_text.splitlines() if "RuntimeWarning:" in line
        )

    assert _warning_messages(result_1.stderr) == _warning_messages(result_4.stderr)


@requires_mpirun
def test_mpirun_dry_run(tmp_path):
    cmd = ["mpirun", "-n", "3", sys.executable, *ENTRY_POINT_ARGS,
           "-i", str(EXAMPLE_INPUT), "--dry-run"]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                             text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--dry-run: input validation complete") == 1


@requires_mpirun
@requires_matplotlib
def test_mpirun_with_plots(tmp_path):
    out_dir = tmp_path / "plots_n3"
    result = _run_under_mpirun(3, out_dir, extra_args=["-p"])
    assert result.returncode == 0, result.stderr
    for name in ("V_Ga_ef.png", "V_N_ef.png", "comparison_plot.png"):
        p = out_dir / "plots" / name
        assert p.exists() and p.stat().st_size > 0


# ===========================================================================
# Fix-1 / Fix-2a: MPI collective-boundary exception handling, live
# ===========================================================================

@requires_mpirun
def test_mpirun_bad_input_fails_cleanly_without_hanging(tmp_path):
    """A rank-0-only load_input() failure must not hang the other ranks
    waiting inside comm.bcast() (see main.py's load_exc/bcast-flag
    handling). The timeout turns an actual hang into a clear pytest
    failure instead of blocking the whole suite indefinitely."""
    bad_input = tmp_path / "bad_input.yaml"
    bad_input.write_text("not: a valid, ikazuchix: input\n", encoding="utf-8")
    cmd = ["mpirun", "-n", "4", sys.executable, *ENTRY_POINT_ARGS,
           "-i", str(bad_input)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                             text=True, timeout=60)
    assert result.returncode != 0


@requires_mpirun
def test_mpirun_single_work_item_failure_fails_cleanly_without_hanging(tmp_path):
    """A compute-time failure isolated to exactly ONE (defect,
    charge_state) work item -- which under mpirun -n 4 lands on only one
    rank, while the other 3 ranks' work items all succeed -- must not
    hang those other ranks waiting inside comm.gather() (see
    runner.run_all_defects' per-item try/except + DefectComputationError).

    The fault is an absurdly large (but schema-valid: a plain integer)
    charge, 10**200, chosen because it is deliberately NOT something
    load_input() should ever reject -- there is no principled upper
    bound on an integer charge value to validate against -- yet it
    still fails deep inside compute_charge_state_result(): q**2 (a huge
    Python int) overflows converting to a float when multiplied by
    COULOMB_PREFACTOR in KumagaiObaCorrectionCalculator._E_real(). This
    keeps the test exercising the DefectComputationError/per-item
    try-except path specifically (as opposed to a load_input()-time
    rejection, e.g. a non-numeric 'energy', which -- now that
    load_input() validates 'energy' -- would instead be caught earlier,
    at the raise_consistently() call around load_input() in main.py;
    that earlier collective boundary has its own hang-safety, covered
    by test_load_input_failure_* below)."""
    with open(EXAMPLE_INPUT, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for cs in data["defects"][0]["charge_states"]:  # V_Ga
        if cs["charge"] == -2:
            cs["charge"] = 10**200
    bad_input = tmp_path / "one_bad_charge_state.yaml"
    with open(bad_input, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)

    cmd = ["mpirun", "-n", "4", sys.executable, *ENTRY_POINT_ARGS,
           "-i", str(bad_input)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                             text=True, timeout=60)
    assert result.returncode != 0
    assert "DefectComputationError" in result.stderr
    assert "V_Ga" in result.stderr


@requires_mpirun
def test_mpirun_load_input_failure_fails_cleanly_without_hanging(tmp_path):
    """A load_input() failure -- deterministic and identical on every
    rank, since every rank calls load_input() independently before the
    comm.bcast() in main.py -- must also not hang: rank 0's raise would
    otherwise be caught only by rank 0, leaving every other rank
    blocked forever inside that same comm.bcast() (see
    mpi_utils.raise_consistently(), applied around load_input() in
    main.py). A non-numeric 'energy' is used here specifically because
    load_input() now validates it (see _validate_and_normalize_charge's
    sibling check for 'energy' in io_utils.load_input), so this
    exercises the *load_input*, not compute, collective boundary."""
    with open(EXAMPLE_INPUT, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    for cs in data["defects"][0]["charge_states"]:  # V_Ga
        if cs["charge"] == -2:
            cs["energy"] = "not_a_number"
    bad_input = tmp_path / "bad_energy.yaml"
    with open(bad_input, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)

    cmd = ["mpirun", "-n", "4", sys.executable, *ENTRY_POINT_ARGS,
           "-i", str(bad_input)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True,
                             text=True, timeout=60)
    assert result.returncode != 0
    assert "'energy' must be a numeric value" in result.stderr
