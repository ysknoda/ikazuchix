#!/usr/bin/env python3
"""Benchmark: numba-JIT-accelerated Ewald kernels vs. the pure-Python path.

Standalone script (not a pytest test) -- timing comparisons are
inherently sensitive to machine load, so they are reported for human
inspection rather than asserted on. Both paths are verified elsewhere
(tests/test_ewald_kernels.py, tests/test_golden_regression.py) to
produce numerically identical results; this script only measures speed.

Usage:
    python benchmarks/bench_numba.py [--real-cutoff N] [--recip-cutoff N]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ikazuchix.ewald import KumagaiObaCorrectionCalculator
from ikazuchix.numba_shim import HAS_NUMBA, set_numba_enabled

L = np.array([[9.57, 0.0, 0.0],
              [-4.785, 8.2877, 0.0],
              [0.0, 0.0, 10.38]])
EPS = np.array([[9.5, 0.0, 0.0], [0.0, 9.5, 0.0], [0.0, 0.0, 10.4]])
POS = [0.16667, 0.33333, 0.0]


def _time_one(real_cutoff, recip_cutoff, q=2, delta_v=0.05):
    calc = KumagaiObaCorrectionCalculator(
        lattice=L, epsilon=EPS, charge=q, defect_frac=POS, delta_v=delta_v,
        real_cutoff=real_cutoff, recip_cutoff=recip_cutoff)
    t0 = time.perf_counter()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc.compute()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-cutoff", type=int, default=10)
    p.add_argument("--recip-cutoff", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    n_points = (2 * args.real_cutoff + 1) ** 3 - 1
    print(f"Real-/recip-space shell cutoff : {args.real_cutoff} / {args.recip_cutoff}")
    print(f"Real-space lattice points swept: {n_points}")
    print(f"numba installed                : {HAS_NUMBA}")
    print()

    if not HAS_NUMBA:
        print("numba is not installed in this environment -- only the "
              "pure-Python path can be benchmarked here. Install numba "
              "to see the JIT speedup (`pip install numba` or "
              "`conda install numba`).")

    # Pure-Python path (also what every run uses when numba is absent).
    set_numba_enabled(False)
    # Warm-up (first call pays Python import/attribute-lookup overhead
    # that has nothing to do with the algorithm itself).
    _time_one(2, 2)
    py_times = [_time_one(args.real_cutoff, args.recip_cutoff)
                for _ in range(args.repeats)]
    py_best = min(py_times)
    print(f"pure-Python : best of {args.repeats} = {py_best:.4f} s "
          f"(all: {[f'{t:.4f}' for t in py_times]})")

    if HAS_NUMBA:
        set_numba_enabled(True)
        # Warm-up: pays the one-time JIT compilation cost, which should
        # not be counted against the steady-state speed comparison.
        t_compile0 = time.perf_counter()
        _time_one(2, 2)
        t_compile = time.perf_counter() - t_compile0
        print(f"numba       : one-time JIT warm-up = {t_compile:.4f} s "
              "(paid once per process, cached to disk via cache=True "
              "across separate CLI invocations)")

        numba_times = [_time_one(args.real_cutoff, args.recip_cutoff)
                       for _ in range(args.repeats)]
        numba_best = min(numba_times)
        print(f"numba       : best of {args.repeats} = {numba_best:.4f} s "
              f"(all: {[f'{t:.4f}' for t in numba_times]})")
        print()
        print(f"Speedup (pure-Python / numba): {py_best / numba_best:.1f}x")

    set_numba_enabled(True)  # restore default


if __name__ == "__main__":
    main()
