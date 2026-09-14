#!/usr/bin/env python3
"""Benchmark: Perf-2 incremental (ring-kernel) --adaptive-ewald shell
growth, via EwaldShellCache.raw_*_upto(), vs. the pre-Perf-2 behaviour
of recomputing the full O(n^3) cube from scratch at every grown shell.

Standalone script (not a pytest test). Numerical equivalence (to
relative 1e-9, deliberately NOT bit-exact -- see
ikazuchix/ewald_kernels.py's ring-kernel module docstring) is verified
in tests/test_ewald.py; this script only measures speed at a high
target cutoff, where the saved inner-shell recomputation is largest.

Usage:
    python benchmarks/bench_perf2_adaptive_incremental.py [--ewald-tol EV] [--max-shells N]
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ikazuchix.ewald import KumagaiObaCorrectionCalculator, build_ewald_shell_cache

L = np.array([[9.57, 0.0, 0.0],
              [-4.785, 8.2877, 0.0],
              [0.0, 0.0, 10.38]])
EPS = np.array([[9.5, 0.0, 0.0], [0.0, 9.5, 0.0], [0.0, 0.0, 10.4]])
POS = [0.16667, 0.33333, 0.0]


def _compute(shell_cache, ewald_tol, max_shells):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc = KumagaiObaCorrectionCalculator(
            lattice=L, epsilon=EPS, charge=2, defect_frac=POS, delta_v=0.05,
            real_cutoff=1, recip_cutoff=1, adaptive=True,
            ewald_tol=ewald_tol, max_shells=max_shells, shell_cache=shell_cache)
        calc.compute()
        return calc.n_real_used, calc.n_recip_used


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ewald-tol", type=float, default=1e-10,
                    help="Tight tolerance to force growth to a large "
                         "cutoff (starting from real/recip-cutoff=1). "
                         "Default: 1e-10.")
    p.add_argument("--max-shells", type=int, default=25)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    print(f"Ewald tolerance : {args.ewald_tol:.0e} eV")
    print(f"Max shells      : {args.max_shells}")
    print()

    _compute(None, 1e-6, 5)  # warm-up (numba JIT, if active)

    uncached_times = []
    n_real = n_recip = None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        n_real, n_recip = _compute(None, args.ewald_tol, args.max_shells)
        uncached_times.append(time.perf_counter() - t0)
    print(f"Converged shell counts: real={n_real}, recip={n_recip}")
    uncached_best = min(uncached_times)
    print(f"full-recompute-per-step (no shell_cache)  : "
          f"best of {args.repeats} = {uncached_best:.4f} s "
          f"(all: {[f'{t:.4f}' for t in uncached_times]})")

    cached_times = []
    for _ in range(args.repeats):
        shell_cache = build_ewald_shell_cache(L, EPS)
        t0 = time.perf_counter()
        _compute(shell_cache, args.ewald_tol, args.max_shells)
        cached_times.append(time.perf_counter() - t0)
    cached_best = min(cached_times)
    print(f"incremental ring-kernel (with shell_cache): "
          f"best of {args.repeats} = {cached_best:.4f} s "
          f"(all: {[f'{t:.4f}' for t in cached_times]})")
    print()
    print(f"Speedup (full-recompute / incremental): "
          f"{uncached_best / cached_best:.1f}x")


if __name__ == "__main__":
    main()
