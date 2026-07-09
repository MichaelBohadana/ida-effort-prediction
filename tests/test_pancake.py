import random
from collections import deque
from itertools import permutations

import numpy as np

from effortpred.pancake import (
    pancake_goal, flip, successors_pancake, dual_state, gap_h, gap2_h,
    rand_h, random_pancake_state, exact_pancake_distribution,
    estimate_pancake_distribution,
)


def test_flip_hand_cases():
    assert flip((0, 1, 2, 3), 2) == (1, 0, 2, 3)
    assert flip((0, 1, 2, 3), 3) == (2, 1, 0, 3)
    assert flip((0, 1, 2, 3), 4) == (3, 2, 1, 0)
    # flipping twice restores the state
    s = (3, 0, 2, 1)
    for k in (2, 3, 4):
        assert flip(flip(s, k), k) == s


def test_successors_parent_pruning():
    s = pancake_goal(5)
    root_children = successors_pancake(s, prev_k=0)
    assert len(root_children) == 4                      # flips 2..5
    assert sorted(k for k, _ in root_children) == [2, 3, 4, 5]
    deeper = successors_pancake(s, prev_k=3)
    assert len(deeper) == 3                             # flip 3 excluded
    assert all(k != 3 for k, _ in deeper)


def test_dual_state():
    # dual = inverse permutation; dual[pancake] = its position
    assert dual_state((2, 0, 1, 3)) == (1, 2, 0, 3)
    assert dual_state(pancake_goal(6)) == pancake_goal(6)
    s = (3, 1, 4, 0, 2)
    assert dual_state(dual_state(s)) == s


def test_gap_hand_cases():
    assert gap_h(pancake_goal(5)) == 0
    # (1,0,2,3,4): only adjacency (0,2) is a gap; plate ok -> h = 1
    assert gap_h((1, 0, 2, 3, 4)) == 1
    # (4,3,2,1,0): internal adjacencies all differ by 1 (no gaps);
    # bottom pancake 0 != 4 -> plate gap -> h = 1
    assert gap_h((4, 3, 2, 1, 0)) == 1


def test_gap2_hand_cases():
    assert gap2_h(pancake_goal(5)) == 0
    # (1,0,2,3,4): the (1,0) and (0,2) adjacencies involve pancakes < 2 ->
    # ignored; everything else in place -> h = 0 (weaker than gap_h's 1)
    assert gap2_h((1, 0, 2, 3, 4)) == 0
    # (4,2,3,1,0): (4,2) is a gap not involving 0/1 -> counted; (2,3) fine;
    # (3,1),(1,0) involve pancakes < 2 -> ignored; bottom is 0 < 2 -> plate
    # ignored -> h = 1
    assert gap2_h((4, 2, 3, 1, 0)) == 1


def _bfs_distances_pancake(n):
    start = pancake_goal(n)
    dist = {start: 0}
    q = deque([start])
    while q:
        s = q.popleft()
        for _k, c in successors_pancake(s, prev_k=0):   # full neighbors for BFS
            if c not in dist:
                dist[c] = dist[s] + 1
                q.append(c)
    return dist


def test_all_permutations_reachable_n7():
    dist = _bfs_distances_pancake(7)
    assert len(dist) == 5040                            # 7! — every perm reachable


def test_gap_admissible_and_consistent_exhaustive_n7():
    """GATE: GAP must be admissible for ALL 5,040 states and consistent
    (|h(s) - h(child)| <= 1) across ALL edges."""
    dist = _bfs_distances_pancake(7)
    for s, d in dist.items():
        h = gap_h(s)
        assert h <= d
        for _k, c in successors_pancake(s, prev_k=0):
            assert abs(gap_h(c) - h) <= 1


def test_gap_is_self_dual_exhaustive_n7():
    """LEMMA (documented on purpose): gap_h(dual(s)) == gap_h(s) for every
    state. Non-gap adjacencies of s are exactly the consecutive-value pairs
    that are position-adjacent — the same set seen from the dual side — and
    the plate terms coincide (s[n-1] == n-1  <=>  dual[n-1] == n-1). This is
    why dual evaluation CANNOT produce an inconsistent heuristic from GAP,
    and why rand_h (random selection) is used instead."""
    for perm in permutations(range(7)):
        assert gap_h(dual_state(perm)) == gap_h(perm)


def test_gap2_and_rand_admissible_exhaustive_and_rand_inconsistent_n7():
    """GATE: gap2_h and rand_h admissible for ALL 5,040 states; rand_h is
    genuinely INCONSISTENT (some edge changes it by more than 1); gap2_h is
    genuinely weaker than gap_h somewhere. If admissibility FAILS, report
    BLOCKED — do not weaken this test."""
    dist = _bfs_distances_pancake(7)
    witness_inconsistent = False
    witness_weaker = False
    for s, d in dist.items():
        assert gap2_h(s) <= d
        h = rand_h(s)
        assert h <= d
        if gap2_h(s) < gap_h(s):
            witness_weaker = True
        for _k, c in successors_pancake(s, prev_k=0):
            if abs(rand_h(c) - h) > 1:
                witness_inconsistent = True
    assert witness_inconsistent, "rand_h looks consistent — defeats its purpose"
    assert witness_weaker, "gap2_h never differs from gap_h — defeats its purpose"


def test_exact_distribution_n7_properties():
    d = exact_pancake_distribution(7, gap_h)
    c = d.cumulative
    assert np.all(np.diff(c) >= -1e-12)
    assert abs(c[-1] - 1.0) < 1e-12
    # gap2 is a genuinely different heuristic: its distribution must differ
    d2 = exact_pancake_distribution(7, gap2_h)
    assert not np.allclose(
        np.resize(d.cumulative, max(len(d.cumulative), len(d2.cumulative))),
        np.resize(d2.cumulative, max(len(d.cumulative), len(d2.cumulative))),
    )


def test_sampled_distribution_matches_exact_n7():
    exact = exact_pancake_distribution(7, gap_h)
    est = estimate_pancake_distribution(7, gap_h, 50_000, random.Random(21))
    hi = max(len(exact.cumulative), len(est.cumulative))
    assert max(abs(exact.P(x) - est.P(x)) for x in range(hi)) < 0.01
