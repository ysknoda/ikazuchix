#!/usr/bin/env python3
"""Benchmark: Perf-1 EwaldShellCache -- one defect's raw Ewald shell
sums shared across its charge states, vs. each charge state
recomputing them independently (the pre-Perf-1 behaviour).

Standalone script (not a pytest test) -- timing comparisons are
inherently sensitive to machine load, so they are reported for human
inspection rather than asserted on. Numerical equivalence between the
cached and uncached paths is verified exactly (bit-for-bit) in
tests/test_ewald.py; this script only measures speed.

Usage:
    python benchmarks/bench_perf1_shell_cache.py [--n-charges N] [--real-cutoff N] [--recip-cutoff N]
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


def _compute_charge_states(n_charges, real_cutoff, recip_cutoff, shell_cache):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for q in range(1, n_charges + 1):
            calc = KumagaiObaCorrectionCalculator(
                lattice=L, epsilon=EPS, charge=q, defect_frac=POS,
                delta_v=0.02 * q, real_cutoff=real_cutoff,
                recip_cutoff=recip_cutoff, shell_cache=shell_cache)
            calc.compute()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-charges", type=int, default=6,
                    help="Number of (simulated) non-zero charge states "
                         "of one defect. Default: 6.")
    p.add_argument("--real-cutoff", type=int, default=6)
    p.add_argument("--recip-cutoff", type=int, default=6)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    print(f"Charge states simulated  : {args.n_charges}")
    print(f"Real-/recip-space cutoff : {args.real_cutoff} / {args.recip_cutoff}")
    print()

    _compute_charge_states(1, 2, 2, None)  # warm-up (numba JIT, if active)

    uncached_times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _compute_charge_states(args.n_charges, args.real_cutoff,
                                args.recip_cutoff, None)
        uncached_times.append(time.perf_counter() - t0)
    uncached_best = min(uncached_times)
    print(f"without shell_cache: best of {args.repeats} = {uncached_best:.4f} s "
          f"(all: {[f'{t:.4f}' for t in uncached_times]})")

    cached_times = []
    for _ in range(args.repeats):
        shell_cache = build_ewald_shell_cache(L, EPS)
        t0 = time.perf_counter()
        _compute_charge_states(args.n_charges, args.real_cutoff,
                                args.recip_cutoff, shell_cache)
        cached_times.append(time.perf_counter() - t0)
    cached_best = min(cached_times)
    print(f"with shell_cache   : best of {args.repeats} = {cached_best:.4f} s "
          f"(all: {[f'{t:.4f}' for t in cached_times]})")
    print()
    print(f"Speedup (uncached / cached): {uncached_best / cached_best:.1f}x "
          f"(expected to approach {args.n_charges}x as --n-charges grows, "
          "since the shared cache turns N kernel evaluations into 1)")


if __name__ == "__main__":
    main()
