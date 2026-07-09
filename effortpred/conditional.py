"""Conditional heuristic-value distributions p(v|vp) for CDP.

Matrix convention (matches Zahavi et al. 2010, §4.1.1 and Figure 6):
M[v, vp] = probability that a generated child has heuristic value v, given
its parent (the node being expanded) has value vp. Columns are normalized.

Adaptation to our parent-pruned tree: each sample fixes an "incoming flip"
(the move that hypothetically produced the parent) and tallies only the
children of the OTHER flips. The paper's basic 1-step model tallies all
children and needs its 2-step model p(v|vp, vgp) to correct for operator
pruning; with the pancake's uniform branching, excluding the incoming flip
during sampling achieves that correction directly.
"""

from itertools import permutations

import numpy as np

from .pancake import flip, random_pancake_state


def H_MAX_PANCAKE(n):
    return n


def sample_conditional_matrix(n, h_fn, n_samples, rng, h_max):
    M = np.zeros((h_max + 1, h_max + 1))
    for _ in range(n_samples):
        s = random_pancake_state(n, rng)
        incoming = rng.randint(2, n)   # randint is INCLUSIVE: samples all n-1 flips {2..n}
        vp = h_fn(s)
        for k in range(2, n + 1):
            if k == incoming:
                continue
            v = h_fn(flip(s, k))
            M[v, vp] += 1
    cols = M.sum(axis=0)
    return np.divide(M, cols, out=np.zeros_like(M), where=cols > 0)


def exact_conditional_matrix(n, h_fn, h_max):
    """Exact matrix by enumerating ALL states x incoming flips x children.
    Use only for n <= 8."""
    M = np.zeros((h_max + 1, h_max + 1))
    for perm in permutations(range(n)):
        vp = h_fn(perm)
        for incoming in range(2, n + 1):
            for k in range(2, n + 1):
                if k == incoming:
                    continue
                v = h_fn(flip(perm, k))
                M[v, vp] += 1
    cols = M.sum(axis=0)
    return np.divide(M, cols, out=np.zeros_like(M), where=cols > 0)
