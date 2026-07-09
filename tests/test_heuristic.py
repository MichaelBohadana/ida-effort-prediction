import random
from collections import deque

from effortpred.puzzle import (
    goal_state, successors, apply_move, neighbors_table, random_walk_state,
)
from effortpred.heuristic import manhattan, md_table, linear_conflict_pairs


def test_manhattan_hand_cases():
    assert manhattan(goal_state(3), 3) == 0
    # tile 1 at cell 0 (goal cell 1): distance 1; blank contributes 0
    assert manhattan((1, 0, 2, 3, 4, 5, 6, 7, 8), 3) == 1
    # tile 8 at cell 0 (goal cell 8 = (2,2)): distance 4; everything else home
    assert manhattan((8, 1, 2, 3, 4, 5, 6, 7, 0), 3) == 4


def test_manhattan_changes_by_exactly_one_per_move():
    """Moving one tile one step changes its distance by exactly +-1, so h
    changes by exactly +-1. This is the consistency property we rely on."""
    rng = random.Random(1)
    for _ in range(300):
        s = random_walk_state(4, rng.randint(0, 60), rng)
        h = manhattan(s, 4)
        for c in successors(s, 4):
            assert abs(manhattan(c, 4) - h) == 1


def _bfs_distances_3x3():
    start = goal_state(3)
    dist = {start: 0}
    q = deque([start])
    while q:
        s = q.popleft()
        for c in successors(s, 3):
            if c not in dist:
                dist[c] = dist[s] + 1
                q.append(c)
    return dist


def test_manhattan_admissible_exhaustive_8puzzle():
    """h(s) <= true optimal distance for ALL 181,440 reachable states."""
    dist = _bfs_distances_3x3()
    assert len(dist) == 181440
    for s, d in dist.items():
        assert manhattan(s, 3) <= d


def test_incremental_delta_matches_recompute():
    """The delta-table update used inside the fast DFS must agree with a
    full recomputation, for every legal move of many random states."""
    md = md_table(4)
    rng = random.Random(2)
    for _ in range(200):
        s = random_walk_state(4, rng.randint(0, 60), rng)
        b = s.index(0)
        for to in neighbors_table(4)[b]:
            tile = s[to]
            dh = md[tile][b] - md[tile][to]
            assert manhattan(apply_move(s, to), 4) == manhattan(s, 4) + dh


def test_linear_conflict_hand_cases():
    assert linear_conflict_pairs(goal_state(3), 3) == 0
    # row 0 holds tiles 2 and 1 (both goal-row 0) in reversed order: 1 pair
    assert linear_conflict_pairs((0, 2, 1, 3, 4, 5, 6, 7, 8), 3) == 1
