"""Pilot run: measure node counts / wall time / censoring per bound offset so
a human can pick OFFSETS and CAP for the full generation run. Read the table,
then set --offsets/--cap when running generate_data.py."""

import argparse
import random
import time

import numpy as np

from effortpred.count import count_expansions
from effortpred.heuristic import manhattan
from effortpred.puzzle import random_walk_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=30)
    ap.add_argument("--max-offset", type=int, default=12)
    ap.add_argument("--cap", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    states = [random_walk_state(4, rng.randint(10, 70), rng)
              for _ in range(args.n_states)]

    print(f"{'offset':>6} {'median':>10} {'p90':>10} {'max':>10} "
          f"{'censored':>8} {'sec/state':>10}")
    for off in range(0, args.max_offset + 1, 2):
        counts, censored, t0 = [], 0, time.perf_counter()
        for s in states:
            nodes, c = count_expansions(s, 4, manhattan(s, 4) + off, cap=args.cap)
            counts.append(nodes)
            censored += c
        dt = (time.perf_counter() - t0) / len(states)
        a = np.array(counts, dtype=float)
        print(f"{off:>6} {np.median(a):>10.0f} {np.percentile(a, 90):>10.0f} "
              f"{a.max():>10.0f} {censored:>8} {dt:>10.3f}")


if __name__ == "__main__":
    main()
