"""Shared pytest fixtures and helpers for the ikazuchix test suite."""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# CLI entry point is `python -m ikazuchix ...` (no top-level script) --
# this is the arg list to append after sys.executable. Requires cwd to
# be REPO_ROOT (or ikazuchix/ importable some other way) so `-m` module
# resolution can find the package.
ENTRY_POINT_ARGS = ["-m", "ikazuchix"]
EXAMPLE_INPUT = REPO_ROOT / "examples" / "example_input.yaml"
GOLDEN_DIR = REPO_ROOT / "tests" / "golden"


def _detect_numba():
    try:
        importlib.import_module("numba")
        return True
    except ImportError:
        return False


def _detect_mpi():
    try:
        from mpi4py import MPI  # noqa: F401

        return True
    except (ImportError, RuntimeError):
        return False


def _detect_matplotlib():
    try:
        importlib.import_module("matplotlib")
        return True
    except ImportError:
        return False


def _probe_mpirun_can_launch(launcher, timeout=60):
    """Whether `launcher` can actually start a job in this environment.

    shutil.which() only proves the binary exists. A launcher can be
    installed and still be unable to launch anything -- observed on a
    GitHub Actions runner, where every `mpirun` invocation exited 1
    instantly with completely empty stdout *and* stderr, turning the
    whole module into failures that said nothing about this package.

    The probe deliberately runs a bare no-op interpreter, never this
    package, so it separates the two cases that must not be conflated:
    "the environment cannot host an MPI job at all" (skip -- there is
    nothing here to test) versus "mpirun works but ikazuchix breaks
    under it" (fail loudly -- that is a real defect). If the probe
    succeeds, every assertion below runs exactly as before.

    Returns (ok, detail); detail is folded into the skip reason so a
    skipped run still records *why* the launcher was unusable.
    """
    cmd = [launcher, "-n", "1", sys.executable, "-c", ""]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode == 0:
        return True, ""
    output = (proc.stderr or proc.stdout or "").strip()
    first_line = output.splitlines()[0] if output else None
    return False, (
        f"`{' '.join(cmd)}` exited {proc.returncode}"
        + (f": {first_line}" if first_line else " with no output at all")
    )


HAS_NUMBA = _detect_numba()
HAS_MPI = _detect_mpi()
HAS_MATPLOTLIB = _detect_matplotlib()

MPIRUN = shutil.which("mpirun")
# Probed once at collection time (~0.2 s) rather than per test.
MPIRUN_USABLE, MPIRUN_DETAIL = (
    _probe_mpirun_can_launch(MPIRUN) if MPIRUN else (False, "not found on PATH")
)

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
requires_matplotlib = pytest.mark.skipif(
    not HAS_MATPLOTLIB, reason="matplotlib not installed"
)
# Three conditions matter, not one:
#  - mpirun on PATH;
#  - mpirun able to actually launch (see _probe_mpirun_can_launch);
#  - mpi4py usable in this Python environment -- launching a real
#    multi-rank mpirun job whose Python environment lacks mpi4py makes
#    every rank fall back to an independent single-process NullComm, so
#    N ranks redundantly compute *and race to write* the same output
#    files. That failure mode is exactly what --no-mpi's own warning
#    covers when triggered deliberately, but here it would just make
#    these specific tests flaky/wrong rather than cleanly skipped.
requires_mpirun = pytest.mark.skipif(
    not (MPIRUN and MPIRUN_USABLE and HAS_MPI),
    reason=(
        f"usable mpirun required (mpirun: {MPIRUN_DETAIL or 'ok'}; "
        f"mpi4py usable: {HAS_MPI})"
    ),
)


def run_cli(args, cwd=None, extra_env=None, timeout=120):
    """Run the CLI (``python -m ikazuchix ...``) as a subprocess.

    Returns a subprocess.CompletedProcess (text mode). ``args`` excludes
    the python executable and the -m ikazuchix prefix.
    """
    import os

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, *ENTRY_POINT_ARGS] + list(args)
    return subprocess.run(
        cmd,
        cwd=str(cwd or REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _is_float_token(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def assert_text_numerically_equal(path_a, path_b, abs_tol=1e-6, rel_tol=1e-9):
    """Compare two whitespace-tokenised text files.

    Numeric tokens are compared with a tolerance (to survive incidental
    formatting/precision differences); non-numeric tokens must match
    exactly. Line count and per-line token count must match.
    """
    lines_a = Path(path_a).read_text(encoding="utf-8").splitlines()
    lines_b = Path(path_b).read_text(encoding="utf-8").splitlines()
    assert len(lines_a) == len(lines_b), (
        f"{path_a} and {path_b} have different line counts "
        f"({len(lines_a)} vs {len(lines_b)})"
    )

    for lineno, (la, lb) in enumerate(zip(lines_a, lines_b), start=1):
        toks_a, toks_b = la.split(), lb.split()
        assert len(toks_a) == len(toks_b), (
            f"{path_a}:{lineno} token-count mismatch:\n  {la!r}\n  {lb!r}"
        )
        for ta, tb in zip(toks_a, toks_b):
            if _is_float_token(ta) and _is_float_token(tb):
                fa, fb = float(ta), float(tb)
                assert fa == pytest.approx(fb, abs=abs_tol, rel=rel_tol), (
                    f"{path_a}:{lineno} numeric mismatch: {fa} != {fb}\n"
                    f"  {la!r}\n  {lb!r}"
                )
            else:
                assert ta == tb, (
                    f"{path_a}:{lineno} token mismatch: {ta!r} != {tb!r}\n"
                    f"  {la!r}\n  {lb!r}"
                )


@pytest.fixture
def example_input():
    return EXAMPLE_INPUT


@pytest.fixture
def golden_dir():
    return GOLDEN_DIR
