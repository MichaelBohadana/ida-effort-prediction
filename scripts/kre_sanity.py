"""Large-set KRE sanity report on the 15-puzzle (the regime where Lecture 5
says KRE is accurate). Bounds must sit near/above the mean h (~37) of uniform
random states, else most states expand 0 nodes and the check is degenerate
(the original defaults 24/26 had exactly that flaw). Healthy result:
mean-ratio within roughly [0.5, 2].
All job parameters travel inside job tuples (multiprocessing-spawn safe)."""

import argparse
import multiprocessing as mp
import random

import numpy as np

from effortpred.count import count_expansions
from effortpred.distribution import HDistribution
from effortpred.kre import kre_predict
from effortpred.puzzle import random_solvable_state


def actual_one(job):
    state, bound, cap = job
    nodes, censored = count_expansions(state, 4, bound, cap=cap)
    return nodes, censored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=3000)
    ap.add_argument("--bounds", type=int, nargs="+", default=[36, 38])
    ap.add_argument("--cap", type=int, default=20_000_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    states = [random_solvable_state(4, rng) for _ in range(args.n_states)]
    dists = {
        "overall": HDistribution(np.load("results/hdist_overall.npy")),
        "equilibrium": HDistribution(np.load("results/hdist_equilibrium.npy")),
    }

    for bound in args.bounds:
        jobs = [(s, bound, args.cap) for s in states]
        with mp.Pool(args.workers) as pool:
            res = pool.map(actual_one, jobs, chunksize=8)
        nodes = np.array([r[0] for r in res], dtype=float)
        censored = sum(r[1] for r in res)
        print(f"\nbound={bound}: mean actual={nodes.mean():.1f}, "
              f"median={np.median(nodes):.1f}, censored={censored}/{len(states)}")
        for name, dist in dists.items():
            kre = np.array([kre_predict(s, 4, bound, dist) for s in states])
            ratio = kre.mean() / nodes.mean()
            print(f"  KRE[{name:11s}]: mean={kre.mean():.1f}  "
                  f"mean-ratio={ratio:.2f}")


if __name__ == "__main__":
    main()
