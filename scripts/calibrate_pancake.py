"""Pilot: node counts / time / censoring per bound offset on the 12-pancake,
for both heuristics, so a human picks OFFSETS and CAP for generation."""

import argparse
import random
import time

import numpy as np

from effortpred.pancake import gap_h, rand_h, random_pancake_state
from effortpred.pancake_count import count_expansions_pancake

HEURISTICS = {"gap": gap_h, "rand": rand_h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--n-states", type=int, default=30)
    ap.add_argument("--max-offset", type=int, default=8)
    ap.add_argument("--cap", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=70)
    args = ap.parse_args()

    for name, h_fn in HEURISTICS.items():
        rng = random.Random(args.seed)
        states = [random_pancake_state(args.n, rng) for _ in range(args.n_states)]
        print(f"\n=== heuristic: {name} ===")
        print(f"{'offset':>6} {'median':>10} {'p90':>10} {'max':>10} "
              f"{'censored':>8} {'sec/state':>10}")
        for off in range(0, args.max_offset + 1):
            counts, censored, t0 = [], 0, time.perf_counter()
            for s in states:
                nodes, c = count_expansions_pancake(
                    s, h_fn(s) + off, h_fn, cap=args.cap)
                counts.append(nodes)
                censored += c
            dt = (time.perf_counter() - t0) / len(states)
            a = np.array(counts, dtype=float)
            print(f"{off:>6} {np.median(a):>10.0f} {np.percentile(a, 90):>10.0f} "
                  f"{a.max():>10.0f} {censored:>8} {dt:>10.3f}")


if __name__ == "__main__":
    main()
