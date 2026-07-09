"""Exact per-depth node counts of the parent-pruned brute-force tree.

The tree's shape depends only on the blank trajectory: a node's number of
children = number of neighbors of its blank cell, minus one for the parent
(except at the root). So exact counting needs only aggregates over
(current blank cell, previous blank cell) — O(cells^2 * depth) total, with
exact integer arithmetic. This replaces the b^i approximation in the KRE
formula with the true N_i. Verified against explicit enumeration in
tests/test_tree_size.py.
"""

from .puzzle import neighbors_table


def nodes_per_depth(start_blank, n, max_depth):
    """N[i] = exact number of depth-i nodes of the parent-pruned brute-force
    tree rooted at a state whose blank is at `start_blank`."""
    nbrs = neighbors_table(n)
    counts = {(start_blank, -1): 1}  # (current blank, previous blank) -> count
    result = [1]
    for _ in range(max_depth):
        nxt = {}
        for (cur, prev), c in counts.items():
            for to in nbrs[cur]:
                if to == prev:
                    continue
                key = (to, cur)
                nxt[key] = nxt.get(key, 0) + c
        counts = nxt
        result.append(sum(counts.values()))
    return result


def blank_cell_equilibrium(n, depth=200):
    """Fraction of tree nodes with the blank at each cell, at large depth.

    Starts from a uniform mix over all cells and renormalizes each step
    (float weights). Because the blank's cell parity alternates every move,
    we average two consecutive depths to damp the parity oscillation."""
    nbrs = neighbors_table(n)

    def step(counts):
        nxt = {}
        for (cur, prev), c in counts.items():
            for to in nbrs[cur]:
                if to == prev:
                    continue
                key = (to, cur)
                nxt[key] = nxt.get(key, 0.0) + c
        total = sum(nxt.values())
        return {k: v / total for k, v in nxt.items()}

    counts = {(c, -1): 1.0 / (n * n) for c in range(n * n)}
    for _ in range(depth):
        counts = step(counts)
    counts_next = step(counts)

    frac = [0.0] * (n * n)
    for cdict in (counts, counts_next):
        for (cur, _prev), c in cdict.items():
            frac[cur] += c / 2.0
    return frac
