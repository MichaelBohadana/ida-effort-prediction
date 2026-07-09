import random

import numpy as np

from effortpred.pancake import (
    pancake_goal, successors_pancake, random_pancake_state,
    exact_pancake_distribution, gap_h,
)
from effortpred.pancake_count import count_expansions_pancake
from effortpred.pancake_tree import nodes_per_depth_pancake, kre_predict_pancake
from effortpred.distribution import HDistribution


def _enumerate_counts(state, max_depth):
    counts = [0] * (max_depth + 1)

    def rec(s, prev_k, depth):
        counts[depth] += 1
        if depth == max_depth:
            return
        for k, c in successors_pancake(s, prev_k):
            rec(c, k, depth + 1)

    rec(state, 0, 0)
    return counts


def test_closed_form_matches_enumeration():
    for n in (4, 5, 7):
        expected = _enumerate_counts(pancake_goal(n), 5)
        assert nodes_per_depth_pancake(n, 5) == expected


def test_hand_case():
    # n=7: [1, 6, 6*5, 6*25, ...]
    assert nodes_per_depth_pancake(7, 3) == [1, 6, 30, 150]


def test_kre_trivial_heuristic_is_brute_force():
    dist = HDistribution([1.0])          # P(x) = 1 for all x >= 0
    assert kre_predict_pancake(7, 6, dist) == sum(nodes_per_depth_pancake(7, 6))


def test_kre_pancake_large_set_gate():
    """GATE (mirror of the Phase 1 KRE gate): with the EXACT P(x) and exact
    tree sizes, KRE's single prediction must be in the ballpark of the MEAN
    actual count over many uniform random states, for the consistent gap_h.
    Do not weaken; debug instead (checklist: counter semantics, N_i closed
    form, P indexing, off-by-one in the sum)."""
    dist = exact_pancake_distribution(7, gap_h)
    rng = random.Random(40)
    states = [random_pancake_state(7, rng) for _ in range(600)]
    bound = 8
    actual = np.array(
        [count_expansions_pancake(s, bound, gap_h)[0] for s in states],
        dtype=float,
    )
    pred = kre_predict_pancake(7, bound, dist)
    ratio = pred / actual.mean()
    print(f"\npancake KRE/actual mean ratio at bound {bound}: {ratio:.3f} "
          f"(mean actual {actual.mean():.1f}, KRE {pred:.1f})")
    assert 0.4 < ratio < 2.5
