import random

import numpy as np

from effortpred.count import count_expansions
from effortpred.distribution import HDistribution, exact_distribution_8puzzle
from effortpred.kre import kre_predict
from effortpred.puzzle import random_solvable_state
from effortpred.tree_size import nodes_per_depth


def test_trivial_heuristic_gives_brute_force_size():
    """With h == 0 (P(x) = 1 for all x >= 0), every node within the depth
    bound is fertile, so KRE must equal the whole brute-force tree size."""
    dist = HDistribution([1.0])
    s = random_solvable_state(4, random.Random(12))
    d = 9
    assert kre_predict(s, 4, d, dist) == sum(nodes_per_depth(s.index(0), 4, d))


def test_kre_zero_bound_from_goalish_distribution():
    # With bound 0, only the root can be fertile: prediction = P(0).
    dist = exact_distribution_8puzzle()
    s = random_solvable_state(3, random.Random(13))
    assert kre_predict(s, 3, 0, dist) == dist.P(0)


def test_kre_ballpark_on_8puzzle_large_set():
    """CORRECTNESS GATE (do not weaken — debug instead; checklist in the plan).

    With the EXACT heuristic distribution and exact tree sizes, KRE's mean
    prediction over a large set of uniform random start states must be in the
    right ballpark of the true mean node count — this is exactly the regime
    where Lecture 5 says KRE works. A gross implementation error (wrong
    counting semantics, depth/bound off-by-one, wrong distribution) lands far
    outside [0.4, 2.5]."""
    dist = exact_distribution_8puzzle()
    rng = random.Random(14)
    states = [random_solvable_state(3, rng) for _ in range(800)]
    bound = 14
    actual = np.array(
        [count_expansions(s, 3, bound)[0] for s in states], dtype=float
    )
    pred = np.array([kre_predict(s, 3, bound, dist) for s in states])
    ratio = pred.mean() / actual.mean()
    print(f"\nKRE/actual mean ratio at bound {bound}: {ratio:.3f} "
          f"(mean actual {actual.mean():.1f}, mean KRE {pred.mean():.1f})")
    assert 0.4 < ratio < 2.5
