import random

from effortpred.puzzle import goal_state, random_walk_state
from effortpred.heuristic import manhattan
from effortpred.count import count_expansions, count_expansions_reference


def test_goal_bound_zero():
    # Root has f = 0 <= 0: expanded. Its children have f = 2 > 0: not.
    assert count_expansions(goal_state(3), 3, 0) == (1, False)
    assert count_expansions_reference(goal_state(3), 3, 0) == 1


def test_bound_below_h_is_zero():
    rng = random.Random(3)
    s = random_walk_state(3, 20, rng)
    h = manhattan(s, 3)
    assert h > 0
    assert count_expansions(s, 3, h - 1) == (0, False)
    assert count_expansions_reference(s, 3, h - 1) == 0


def test_fast_matches_reference_8puzzle():
    rng = random.Random(4)
    for _ in range(150):
        s = random_walk_state(3, rng.randint(0, 30), rng)
        h = manhattan(s, 3)
        for off in (0, 1, 2, 4, 6):
            ref = count_expansions_reference(s, 3, h + off)
            fast, censored = count_expansions(s, 3, h + off)
            assert not censored
            assert fast == ref


def test_fast_matches_reference_15puzzle_small():
    rng = random.Random(5)
    for _ in range(30):
        s = random_walk_state(4, rng.randint(0, 25), rng)
        h = manhattan(s, 4)
        for off in (0, 2, 4):
            ref = count_expansions_reference(s, 4, h + off)
            assert count_expansions(s, 4, h + off) == (ref, False)


def test_monotone_in_bound():
    rng = random.Random(6)
    for _ in range(30):
        s = random_walk_state(4, rng.randint(0, 30), rng)
        h = manhattan(s, 4)
        counts = [count_expansions(s, 4, h + off)[0] for off in (0, 2, 4, 6)]
        assert counts == sorted(counts)


def test_odd_bound_equals_even_bound_below():
    """f-parity: every node's f has the parity of h(start), so raising the
    bound by an odd amount adds nothing. Justifies even-only offsets."""
    rng = random.Random(7)
    for _ in range(50):
        s = random_walk_state(4, rng.randint(0, 30), rng)
        h = manhattan(s, 4)
        assert count_expansions(s, 4, h + 1)[0] == count_expansions(s, 4, h)[0]
        assert count_expansions(s, 4, h + 3)[0] == count_expansions(s, 4, h + 2)[0]


def test_cap_censors():
    rng = random.Random(8)
    s = random_walk_state(4, 60, rng)
    h = manhattan(s, 4)
    full = count_expansions(s, 4, h + 6)[0]
    assert full > 10
    capped, censored = count_expansions(s, 4, h + 6, cap=10)
    assert censored is True
    assert capped == 10
