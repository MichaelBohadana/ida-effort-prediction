"""The pancake puzzle domain + GAP heuristics.

State: a tuple permutation of 0..n-1; goal = (0, 1, ..., n-1). A move
"flip k" (2 <= k <= n) reverses the first k elements. Every permutation is
reachable. Parent pruning: never repeat the previous flip (it would undo it).
Branching is uniform: n-1 at the root, n-2 below.

Heuristics:
- gap_h: the GAP heuristic (Helmert 2010): count adjacencies (including the
  virtual "plate" below the bottom pancake) whose sizes differ by more than 1.
  Each flip changes exactly ONE adjacency, so gap_h changes by at most 1 per
  move: consistent (and admissible). Verified exhaustively on n=7 in tests.
- gap2_h: GAP ignoring adjacencies that involve pancake 0 or 1 (GAP-2)
  — the standard "GAP-x" weakening (a strictly weaker consistent, admissible
  heuristic). Verified exhaustively on n=7 in tests.
- rand_h(s): gap_h(s) or gap2_h(s), selected deterministically by
  zlib.crc32(bytes(s)) & 1. This is the "random selection of heuristics"
  method for constructing an INCONSISTENT admissible heuristic (Zahavi et
  al. 2010, §2.5): each individual heuristic is consistent, but the values
  actually consulted along a path can jump. crc32 (not Python's salted
  hash()) keeps it deterministic across runs.

LEMMA (documented in tests): GAP is SELF-DUAL — gap_h(dual(s)) == gap_h(s)
for every permutation — so the paper's "dual evaluation" trick cannot make
GAP inconsistent (it works for PDB lookups, which GAP is not). An earlier
design used dual-GAP as the inconsistent heuristic; the exhaustive n=7 gate
refuted it, and rand_h replaced it.
"""

import zlib
from itertools import permutations

import numpy as np

from .distribution import HDistribution


def pancake_goal(n):
    return tuple(range(n))


def flip(state, k):
    """Reverse the first k elements (2 <= k <= len(state))."""
    return state[k - 1::-1] + state[k:]


def successors_pancake(state, prev_k):
    """All (k, child) pairs for flips k = 2..n excluding prev_k (parent
    pruning). Pass prev_k=0 at the root / for full neighbor generation."""
    n = len(state)
    return [(k, flip(state, k)) for k in range(2, n + 1) if k != prev_k]


def dual_state(state):
    inv = [0] * len(state)
    for pos, pancake in enumerate(state):
        inv[pancake] = pos
    return tuple(inv)


def gap_h(state):
    n = len(state)
    h = 0
    for i in range(n - 1):
        if abs(state[i] - state[i + 1]) > 1:
            h += 1
    if state[n - 1] != n - 1:       # adjacency to the virtual plate (= n)
        h += 1
    return h


def gap2_h(state):
    """GAP ignoring adjacencies that involve pancake 0 or 1 (GAP-2)."""
    n = len(state)
    h = 0
    for i in range(n - 1):
        a, b = state[i], state[i + 1]
        if a >= 2 and b >= 2 and abs(a - b) > 1:
            h += 1
    if state[n - 1] != n - 1 and state[n - 1] >= 2:
        h += 1
    return h


def rand_h(state):
    """Random selection between gap_h and gap2_h, keyed deterministically by
    a crc32 of the state (Python's hash() is salted per process — unusable).
    Admissible (both components are); inconsistent (neighbors may consult
    different components)."""
    return gap_h(state) if zlib.crc32(bytes(state)) & 1 else gap2_h(state)


def random_pancake_state(n, rng):
    cells = list(range(n))
    rng.shuffle(cells)
    return tuple(cells)


def exact_pancake_distribution(n, h_fn):
    """Exact P(x) over ALL n! permutations (use only for n <= 8)."""
    counts = {}
    for perm in permutations(range(n)):
        h = h_fn(perm)
        counts[h] = counts.get(h, 0) + 1
    max_h = max(counts)
    pmf = np.array([counts.get(x, 0) for x in range(max_h + 1)], dtype=float)
    pmf /= pmf.sum()
    return HDistribution(np.cumsum(pmf))


def estimate_pancake_distribution(n, h_fn, n_samples, rng):
    hs = np.empty(n_samples, dtype=np.int64)
    for i in range(n_samples):
        hs[i] = h_fn(random_pancake_state(n, rng))
    pmf = np.bincount(hs)
    pmf = pmf / pmf.sum()
    return HDistribution(np.cumsum(pmf))
