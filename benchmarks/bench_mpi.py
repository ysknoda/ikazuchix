#!/usr/bin/env python3
"""Benchmark: serial vs. mpirun -n N wall-clock time on a multi-defect run.

Standalone script (not a pytest test) -- see benchmarks/bench_numba.py
for why timing comparisons are kept out of the pytest suite. Runs the
actual CLI as a subprocess (optionally under mpirun) against
examples/example_input.yaml, with an elevated Ewald cutoff so each
(defect, charge_state) work item is expensive enough for the
distribution to matter; correctness (identical results regardless of
rank count) is verified separately in tests/test_mpi_consistency.py.

Usage:
    python benchmarks/bench_mpi.py [--real-cutoff N] [--recip-cutoff N]
                                    [--ranks 1 2 4]
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRY_POINT_ARGS = ["-m", "ikazuchix"]
EXAMPLE_INPUT = REPO_ROOT / "examples" / "example_input.yaml"


def _run(n_ranks, real_cutoff, recip_cutoff, out_dir):
    base_cmd = [sys.executable, *ENTRY_POINT_ARGS,
                "-i", str(EXAMPLE_INPUT), "--out-dir", str(out_dir),
                "--real-cutoff", str(real_cutoff),
                "--recip-cutoff", str(recip_cutoff), "--no-numba"]
    if n_ranks == 1:
        cmd = base_cmd
    else:
        cmd = ["mpirun", "-n", str(n_ranks)] + base_cmd

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"run failed (n={n_ranks}): {result.stderr[-2000:]}")
    return elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-cutoff", type=int, default=16)
    p.add_argument("--recip-cutoff", type=int, default=16)
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4])
    args = p.parse_args()

    print(f"Real-/recip-space shell cutoff : {args.real_cutoff} / {args.recip_cutoff}")
    print("Numba                           : disabled for this benchmark "
          "(--no-numba), to isolate the MPI distribution effect from JIT "
          "speedup -- see bench_numba.py for that comparison.")
    print()

    have_mpirun = shutil.which("mpirun") is not None
    if not have_mpirun and any(r > 1 for r in args.ranks):
        print("mpirun not found on PATH: only n_ranks=1 (plain single-"
              "process execution) can be benchmarked here. Install an MPI "
              "runtime to see multi-rank scaling, e.g. "
              "`conda install -c conda-forge openmpi mpi4py`.")
        args.ranks = [r for r in args.ranks if r == 1] or [1]

    timings = {}
    with tempfile.TemporaryDirectory() as tmp:
        for n in args.ranks:
            out_dir = Path(tmp) / f"n{n}"
            elapsed = _run(n, args.real_cutoff, args.recip_cutoff, out_dir)
            timings[n] = elapsed
            print(f"n_ranks={n:<2d} : {elapsed:.3f} s")

    if 1 in timings:
        print()
        baseline = timings[1]
        for n, t in sorted(timings.items()):
            if n == 1:
                continue
            print(f"Speedup at n_ranks={n}: {baseline / t:.2f}x")
        print()
        print("Note: examples/example_input.yaml has only 5 (defect, charge_state)")
        print("work items in total, and every mpirun rank pays the same fixed")
        print("Python/numpy process-startup cost independently (it is not reduced")
        print("by adding ranks) -- so speedup here is modest and bounded by that")
        print("overhead, not by this package's distribution logic. Larger, more")
        print("realistic multi-defect screening studies (more work items, run")
        print("across multiple cluster nodes) see much better scaling; see")
        print("tests/test_mpi_consistency.py for the correctness side of this,")
        print("verified separately from performance.")


if __name__ == "__main__":
    main()
