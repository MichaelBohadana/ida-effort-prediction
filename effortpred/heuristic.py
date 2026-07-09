"""Manhattan-distance heuristic (admissible & consistent) + a linear-conflict
pair count used only as a *feature*.

The goal cell of tile t is cell t (goal = (0, 1, ..., n*n-1)); the blank never
contributes to h.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def md_table(n):
    """md_table(n)[tile][cell] = Manhattan distance of `tile` from its goal
    cell when located at `cell`. Row for tile 0 (the blank) is all zeros."""
    table = []
    for tile in range(n * n):
        row = []
        for cell in range(n * n):
            if tile == 0:
                row.append(0)
            else:
                r1, c1 = divmod(cell, n)
                r2, c2 = divmod(tile, n)
                row.append(abs(r1 - r2) + abs(c1 - c2))
        table.append(tuple(row))
    return tuple(table)


def manhattan(state, n):
    t = md_table(n)
    return sum(t[tile][cell] for cell, tile in enumerate(state))


def linear_conflict_pairs(state, n):
    """Number of linearly conflicting pairs (rows + columns): two tiles whose
    goals are both in the line they currently share, in reversed order.

    Used only as a FEATURE. This is the simple pair count — we deliberately
    skip the max-non-crossing refinement needed to make LC an admissible
    heuristic, because we never search with it."""
    pairs = 0
    for line in range(n):
        # row `line`: tiles in this row whose goal row is also `line`
        row_tiles = []  # (current col, goal col), in increasing current col
        for col in range(n):
            tile = state[line * n + col]
            if tile != 0 and tile // n == line:
                row_tiles.append((col, tile % n))
        pairs += sum(
            1
            for i in range(len(row_tiles))
            for j in range(i + 1, len(row_tiles))
            if row_tiles[i][1] > row_tiles[j][1]
        )
        # column `line`: tiles in this column whose goal column is `line`
        col_tiles = []  # (current row, goal row)
        for row in range(n):
            tile = state[row * n + line]
            if tile != 0 and tile % n == line:
                col_tiles.append((row, tile // n))
        pairs += sum(
            1
            for i in range(len(col_tiles))
            for j in range(i + 1, len(col_tiles))
            if col_tiles[i][1] > col_tiles[j][1]
        )
    return pairs
