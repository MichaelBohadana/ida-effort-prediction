import random

from effortpred.puzzle import successors, random_solvable_state
from effortpred.tree_size import nodes_per_depth, blank_cell_equilibrium


def _enumerate_counts(state, n, max_depth):
    """Explicit parent-pruned enumeration — independent of the recurrence."""
    counts = [0] * (max_depth + 1)

    def rec(s, parent, depth):
        counts[depth] += 1
        if depth == max_depth:
            return
        for c in successors(s, n):
            if c == parent:
                continue
            rec(c, s, depth + 1)

    rec(state, None, 0)
    return counts


def test_recurrence_matches_enumeration():
    rng = random.Random(9)
    for n in (3, 4):
        for _ in range(5):
            s = random_solvable_state(n, rng)
            assert nodes_per_depth(s.index(0), n, 7) == _enumerate_counts(s, n, 7)


def test_hand_case():
    # 3x3, blank in corner (cell 0, 2 neighbors): depth 1 has 2 nodes.
    # Each depth-1 node has blank at an edge cell (3 neighbors), one of which
    # is the parent -> 2 children each -> depth 2 has 4 nodes.
    assert nodes_per_depth(0, 3, 2) == [1, 2, 4]


def test_equilibrium_fractions():
    frac = blank_cell_equilibrium(4)
    assert abs(sum(frac) - 1.0) < 1e-9
    # symmetry: all four corners identical, all four center cells identical
    corners = [frac[c] for c in (0, 3, 12, 15)]
    centers = [frac[c] for c in (5, 6, 9, 10)]
    assert max(corners) - min(corners) < 1e-6
    assert max(centers) - min(centers) < 1e-6
    # the center of the board hosts the blank more often than a corner
    assert centers[0] > corners[0]
