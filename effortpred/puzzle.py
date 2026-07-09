"""Generic N x N sliding-tile puzzle domain.

State representation: a tuple of length n*n. state[cell] = tile at that cell,
0 = the blank. Cells are indexed row-major: cell = row * n + col.
Goal state: (0, 1, 2, ..., n*n - 1) — the blank in the top-left corner, so the
goal cell of tile t is cell t.
A move slides a tile into the blank; we identify it by the cell the blank
moves TO.
"""

import random
from functools import lru_cache


@lru_cache(maxsize=None)
def goal_state(n):
    return tuple(range(n * n))


@lru_cache(maxsize=None)
def neighbors_table(n):
    """For each cell, the tuple of orthogonally adjacent cells."""
    table = []
    for cell in range(n * n):
        r, c = divmod(cell, n)
        adj = []
        if r > 0:
            adj.append(cell - n)
        if r < n - 1:
            adj.append(cell + n)
        if c > 0:
            adj.append(cell - 1)
        if c < n - 1:
            adj.append(cell + 1)
        table.append(tuple(adj))
    return tuple(table)


def apply_move(state, to_cell):
    """Slide the tile at to_cell into the blank. Returns the new state."""
    b = state.index(0)
    lst = list(state)
    lst[b], lst[to_cell] = lst[to_cell], 0
    return tuple(lst)


def successors(state, n):
    b = state.index(0)
    return [apply_move(state, to) for to in neighbors_table(n)[b]]


def permutation_parity(perm):
    """Parity (0 or 1) of a permutation of 0..len-1, via cycle decomposition."""
    seen = [False] * len(perm)
    parity = 0
    for i in range(len(perm)):
        if seen[i]:
            continue
        cycle_len = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            cycle_len += 1
        parity ^= (cycle_len - 1) & 1
    return parity


def is_solvable(state, n):
    """Solvable iff permutation parity == parity of the blank's Manhattan
    distance from its goal cell (cell 0).

    Invariant argument: each move swaps the blank with one tile — a
    transposition, flipping permutation parity — and moves the blank one step,
    flipping the parity of its Manhattan distance to cell 0. So
    (perm_parity XOR blank_dist_parity) is invariant along any move sequence,
    and it equals 0 at the goal. This rule is verified EXHAUSTIVELY against
    BFS reachability on the 8-puzzle in tests/test_puzzle.py.
    """
    b = state.index(0)
    r, c = divmod(b, n)
    blank_dist_parity = (r + c) & 1
    return permutation_parity(state) == blank_dist_parity


def random_walk_state(n, length, rng):
    """Random walk of `length` moves from the goal; solvable by construction.
    Never immediately undoes the previous move, so walks make progress."""
    state = goal_state(n)
    prev_blank = -1
    b = 0
    nbrs = neighbors_table(n)
    for _ in range(length):
        choices = [x for x in nbrs[b] if x != prev_blank]
        to = rng.choice(choices)
        state = apply_move(state, to)
        prev_blank, b = b, to
    return state


def random_solvable_state(n, rng):
    """Uniformly random solvable state (rejection-sample the parity rule)."""
    cells = list(range(n * n))
    while True:
        rng.shuffle(cells)
        state = tuple(cells)
        if is_solvable(state, n):
            return state
