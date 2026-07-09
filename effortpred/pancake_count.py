"""Bounded-DFS expansion counter for the pancake puzzle.

Same semantics as Phase 1's count.py (they must stay in lock-step):
- Parent-pruned brute-force tree (never repeat the previous flip); no other
  duplicate detection.
- A node is expanded (counted) iff f = g + h <= bound.
- No goal stop (KRE/CDP count all fertile nodes).

The heuristic is pluggable (h_fn), so the same counter produces labels under
the consistent gap_h AND the inconsistent rand_h. With an inconsistent
heuristic a fertile node may be unreachable (its ancestor was pruned) — the
DFS naturally implements that, which is exactly why KRE's fertile-node count
overestimates in that regime (Zahavi et al. 2010, §3.2.1).
"""

import sys

from .pancake import flip


class CapExceeded(Exception):
    pass


def count_expansions_pancake(state, bound, h_fn, cap=None):
    """Returns (count, censored). If cap is reached, censored=True and
    count == cap."""
    n = len(state)
    h0 = h_fn(state)
    if h0 > bound:
        return 0, False

    sys.setrecursionlimit(100000)
    counter = [0]

    def dfs(s, prev_k, g):
        counter[0] += 1
        if cap is not None and counter[0] >= cap:
            raise CapExceeded
        for k in range(2, n + 1):
            if k == prev_k:
                continue
            child = flip(s, k)
            if g + 1 + h_fn(child) <= bound:
                dfs(child, k, g + 1)

    try:
        dfs(state, 0, 0)
    except CapExceeded:
        return counter[0], True
    return counter[0], False
