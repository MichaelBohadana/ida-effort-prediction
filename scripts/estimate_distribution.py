"""Estimate and save the 15-puzzle heuristic-value distributions:
- results/hdist_overall.npy      (uniform random solvable states)
- results/hdist_equilibrium.npy  (reweighted by the tree's blank-cell equilibrium)
Both use the SAME seed, hence the same underlying sample of states.
"""

import argparse
import random

import numpy as np

from effortpred.distribution import estimate_distribution
from effortpred.tree_size import blank_cell_equilibrium


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    d_overall = estimate_distribution(4, args.n_samples, random.Random(args.seed))
    np.save(f"{args.out_dir}/hdist_overall.npy", d_overall.cumulative)

    w = blank_cell_equilibrium(4)
    d_eq = estimate_distribution(
        4, args.n_samples, random.Random(args.seed), weights_by_blank_cell=w
    )
    np.save(f"{args.out_dir}/hdist_equilibrium.npy", d_eq.cumulative)

    print(f"overall:     mean h = {mean_of(d_overall):.2f}, support 0..{len(d_overall.cumulative) - 1}")
    print(f"equilibrium: mean h = {mean_of(d_eq):.2f}, support 0..{len(d_eq.cumulative) - 1}")


def mean_of(dist):
    c = dist.cumulative
    pmf = np.diff(np.concatenate([[0.0], c]))
    return float(np.sum(pmf * np.arange(len(pmf))))


if __name__ == "__main__":
    main()
