"""Count the nodes one bounded DFS iteration of IDA* expands.

Semantics (fixed project-wide; matches the KRE model from Lecture 5):
- Tree = parent-pruned brute-force tree: children are all blank moves except
  the one undoing the parent move. No other duplicate detection (IDA* /
  linear-space semantics, exactly what KRE assumes).
- A node is *expanded* (counted) iff f(n) = g + h <= bound. Children with
  f > bound are generated but never expanded (not counted).
- We do NOT stop at the goal: KRE counts all fertile nodes (worst case).

Two implementations:
- count_expansions_reference: slow, obviously correct (immutable tuples,
  full h recomputation, parent identified by comparing whole states).
  Ground truth for tests only.
- count_expansions: fast (in-place board, incremental Manhattan deltas,
  parent identified by blank cell). Used for all data generation.
"""

import sys

from .heuristic import manhattan, md_table
from .puzzle import neighbors_table, successors


class CapExceeded(Exception):
    pass


def count_expansions_reference(state, n, bound):
    """Slow ground-truth counter. Tests only."""

    def rec(s, parent, g):
        h = manhattan(s, n)
        if g + h > bound:
            return 0
        total = 1  # this node is expanded
        for child in successors(s, n):
            if child == parent:
                continue
            total += rec(child, s, g + 1)
        return total

    return rec(state, None, 0)


def count_expansions(state, n, bound, cap=None):
    """Fast counter. Returns (count, censored). If `cap` is given and reached,
    aborts with censored=True and count == cap."""
    md = md_table(n)
    nbrs = neighbors_table(n)
    board = list(state)
    b0 = board.index(0)
    h0 = manhattan(state, n)
    if h0 > bound:
        return 0, False

    sys.setrecursionlimit(100000)
    counter = [0]

    def dfs(blank, prev, g, h):
        counter[0] += 1
        if cap is not None and counter[0] >= cap:
            raise CapExceeded
        for to in nbrs[blank]:
            if to == prev:
                continue
            tile = board[to]
            dh = md[tile][blank] - md[tile][to]
            if g + 1 + h + dh <= bound:
                board[blank] = tile
                board[to] = 0
                dfs(to, blank, g + 1, h + dh)
                board[to] = tile
                board[blank] = 0

    try:
        dfs(b0, -1, 0, h0)
    except CapExceeded:
        return counter[0], True
    return counter[0], False
