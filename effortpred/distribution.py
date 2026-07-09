"""Estimating P(x) = Pr[h(state) <= x], the heuristic-value distribution.

KRE needs the *equilibrium* distribution of h over brute-force tree nodes at
large depth. Following the course treatment (Lecture 5 sets D(x) = P(x)) we
estimate it from uniformly random solvable states. We also provide an
equilibrium-reweighted variant — stratify samples by blank cell, reweight by
the tree's blank-cell equilibrium (as in the original KRE paper) — and report
both in experiments.
"""

from itertools import permutations

import numpy as np

from .heuristic import manhattan
from .puzzle import is_solvable, random_solvable_state


class HDistribution:
    """Cumulative distribution over integer h values.

    cumulative[x] = P(h <= x). P(x) = 0 for x < 0 and 1 past the array."""

    def __init__(self, cumulative):
        self.cumulative = np.asarray(cumulative, dtype=float)

    def P(self, x):
        if x < 0:
            return 0.0
        x = int(x)
        if x >= len(self.cumulative):
            return 1.0
        return float(self.cumulative[x])


def estimate_distribution(n, n_samples, rng, weights_by_blank_cell=None):
    """Estimate P(x) from uniformly random solvable states.

    If weights_by_blank_cell is given (length n*n, sums to 1), samples are
    reweighted so blank-cell mass matches those weights (equilibrium variant).
    """
    hs = np.empty(n_samples, dtype=np.int64)
    blanks = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        s = random_solvable_state(n, rng)
        hs[i] = manhattan(s, n)
        blanks[i] = s.index(0)

    if weights_by_blank_cell is None:
        w = np.ones(n_samples)
    else:
        cell_counts = np.bincount(blanks, minlength=n * n).astype(float)
        target = np.asarray(weights_by_blank_cell, dtype=float)
        per_cell_w = np.divide(
            target, cell_counts / n_samples,
            out=np.zeros_like(target), where=cell_counts > 0,
        )
        w = per_cell_w[blanks]

    max_h = int(hs.max())
    pmf = np.bincount(hs, weights=w, minlength=max_h + 1)
    pmf = pmf / pmf.sum()
    return HDistribution(np.cumsum(pmf))


def exact_distribution_8puzzle():
    """Exact P(x) for the 3x3 puzzle by enumerating ALL solvable states.
    Used by tests and by the KRE correctness gate."""
    counts = {}
    for perm in permutations(range(9)):
        if is_solvable(perm, 3):
            h = manhattan(perm, 3)
            counts[h] = counts.get(h, 0) + 1
    max_h = max(counts)
    pmf = np.array([counts.get(x, 0) for x in range(max_h + 1)], dtype=float)
    pmf /= pmf.sum()
    return HDistribution(np.cumsum(pmf))
