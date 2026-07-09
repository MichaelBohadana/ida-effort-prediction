import random
from collections import deque
from itertools import permutations

from effortpred.puzzle import (
    goal_state, neighbors_table, apply_move, successors,
    is_solvable, random_walk_state, random_solvable_state,
)


def test_goal_and_neighbors_3x3():
    assert goal_state(3) == (0, 1, 2, 3, 4, 5, 6, 7, 8)
    nbrs = neighbors_table(3)
    assert set(nbrs[0]) == {1, 3}          # corner: 2 neighbors
    assert set(nbrs[4]) == {1, 3, 5, 7}    # center: 4 neighbors
    assert set(nbrs[7]) == {4, 6, 8}       # edge: 3 neighbors


def test_apply_move():
    s = goal_state(3)                      # blank at cell 0
    s2 = apply_move(s, 1)                  # slide tile 1 into the blank
    assert s2 == (1, 0, 2, 3, 4, 5, 6, 7, 8)
    # moving back restores the original state
    assert apply_move(s2, 0) == s


def test_successors_counts():
    assert len(successors(goal_state(3), 3)) == 2   # blank in corner
    center = apply_move(apply_move(goal_state(3), 1), 4)  # blank now at cell 4
    assert center.index(0) == 4
    assert len(successors(center, 3)) == 4


def _bfs_reachable_3x3():
    start = goal_state(3)
    seen = {start}
    q = deque([start])
    while q:
        s = q.popleft()
        for c in successors(s, 3):
            if c not in seen:
                seen.add(c)
                q.append(c)
    return seen


def test_solvability_rule_exhaustive_8puzzle():
    """The authority test: the parity rule must agree with true BFS
    reachability for ALL 9! = 362,880 permutations."""
    reachable = _bfs_reachable_3x3()
    assert len(reachable) == 362880 // 2
    for perm in permutations(range(9)):
        assert is_solvable(perm, 3) == (perm in reachable)


def test_random_states_solvable():
    rng = random.Random(0)
    for _ in range(50):
        assert is_solvable(random_walk_state(4, rng.randint(1, 80), rng), 4)
        assert is_solvable(random_solvable_state(4, rng), 4)
