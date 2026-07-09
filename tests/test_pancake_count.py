import random

from effortpred.pancake import (
    pancake_goal, gap_h, rand_h, flip, random_pancake_state,
)
from effortpred.pancake_count import count_expansions_pancake


def _reference_count(state, bound, h_fn):
    """Slow, obviously-correct enumerator, independent of the implementation:
    explicit child construction via flip(), parent tracked by flip index."""
    def rec(s, prev_k, g):
        h = h_fn(s)
        if g + h > bound:
            return 0
        total = 1
        for k in range(2, len(s) + 1):
            if k == prev_k:
                continue
            total += rec(flip(s, k), k, g + 1)
        return total
    return rec(state, 0, 0)


def test_goal_bound_zero():
    assert count_expansions_pancake(pancake_goal(6), 0, gap_h) == (1, False)


def test_bound_below_h_is_zero():
    rng = random.Random(30)
    s = random_pancake_state(7, rng)
    h = gap_h(s)
    if h > 0:
        assert count_expansions_pancake(s, h - 1, gap_h) == (0, False)


def test_matches_reference_consistent():
    rng = random.Random(31)
    for _ in range(60):
        s = random_pancake_state(6, rng)
        h = gap_h(s)
        for off in (0, 1, 2, 3, 4):
            ref = _reference_count(s, h + off, gap_h)
            fast, censored = count_expansions_pancake(s, h + off, gap_h)
            assert not censored and fast == ref


def test_matches_reference_inconsistent():
    """The counter must also be exact under the INCONSISTENT rand_h heuristic —
    where pruning can cut off descendants that 'look' fertile."""
    rng = random.Random(32)
    for _ in range(60):
        s = random_pancake_state(6, rng)
        h = rand_h(s)
        for off in (0, 1, 2, 3):
            ref = _reference_count(s, h + off, rand_h)
            assert count_expansions_pancake(s, h + off, rand_h) == (ref, False)


def test_odd_offsets_do_add_nodes_sometimes():
    """Anti-regression for a Phase 1 assumption that does NOT carry over:
    GAP can stay the same across a flip, so f-parity is NOT invariant and
    odd bound increments genuinely matter for the pancake."""
    rng = random.Random(33)
    found = False
    for _ in range(80):
        s = random_pancake_state(7, rng)
        h = gap_h(s)
        if count_expansions_pancake(s, h + 1, gap_h)[0] > \
           count_expansions_pancake(s, h, gap_h)[0]:
            found = True
            break
    assert found


def test_cap_censors():
    rng = random.Random(34)
    s = random_pancake_state(9, rng)
    h = gap_h(s)
    full = count_expansions_pancake(s, h + 5, gap_h)[0]
    assert full > 10
    capped, censored = count_expansions_pancake(s, h + 5, gap_h, cap=10)
    assert censored is True and capped == 10
