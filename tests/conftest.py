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


HAS_NUMBA = _detect_numba()
HAS_MPI = _detect_mpi()
HAS_MATPLOTLIB = _detect_matplotlib()

requires_numba = pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
requires_matplotlib = pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
# Both conditions matter: mpirun alone is not enough -- launching a real
# multi-rank mpirun job whose Python environment lacks mpi4py makes every
# rank fall back to an independent single-process NullComm, so N ranks
# redundantly compute *and race to write* the same output files. That
# failure mode is exactly what --no-mpi's own warning covers when
# triggered deliberately, but here it would just make these specific
# tests flaky/wrong rather than cleanly skipped.
requires_mpirun = pytest.mark.skipif(
    shutil.which("mpirun") is None or not HAS_MPI,
    reason="mpirun not found on PATH, or mpi4py not usable in this Python environment")


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
        cmd, cwd=str(cwd or REPO_ROOT), env=env,
        capture_output=True, text=True, timeout=timeout,
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
        f"({len(lines_a)} vs {len(lines_b)})")

    for lineno, (la, lb) in enumerate(zip(lines_a, lines_b), start=1):
        toks_a, toks_b = la.split(), lb.split()
        assert len(toks_a) == len(toks_b), (
            f"{path_a}:{lineno} token-count mismatch:\n  {la!r}\n  {lb!r}")
        for ta, tb in zip(toks_a, toks_b):
            if _is_float_token(ta) and _is_float_token(tb):
                fa, fb = float(ta), float(tb)
                assert fa == pytest.approx(fb, abs=abs_tol, rel=rel_tol), (
                    f"{path_a}:{lineno} numeric mismatch: {fa} != {fb}\n"
                    f"  {la!r}\n  {lb!r}")
            else:
                assert ta == tb, (
                    f"{path_a}:{lineno} token mismatch: {ta!r} != {tb!r}\n"
                    f"  {la!r}\n  {lb!r}")


@pytest.fixture
def example_input():
    return EXAMPLE_INPUT


@pytest.fixture
def golden_dir():
    return GOLDEN_DIR
