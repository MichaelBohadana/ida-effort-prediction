import random

import numpy as np

from effortpred.distribution import (
    HDistribution, estimate_distribution, exact_distribution_8puzzle,
)


def test_hdistribution_lookup():
    d = HDistribution([0.25, 0.5, 1.0])
    assert d.P(-1) == 0.0
    assert d.P(0) == 0.25
    assert d.P(1) == 0.5
    assert d.P(2) == 1.0
    assert d.P(999) == 1.0


def test_exact_8puzzle_properties():
    d = exact_distribution_8puzzle()
    c = d.cumulative
    assert np.all(np.diff(c) >= -1e-12)          # monotone
    assert abs(c[-1] - 1.0) < 1e-12              # ends at 1
    assert d.P(0) > 0                            # the goal state has h = 0


def test_sampler_matches_exact_8puzzle():
    """The uniform sampler's estimated P must agree with the exact P
    computed by full enumeration (max CDF gap < 0.01 at 50k samples)."""
    exact = exact_distribution_8puzzle()
    est = estimate_distribution(3, 50_000, random.Random(10))
    hi = max(len(exact.cumulative), len(est.cumulative))
    diff = max(abs(exact.P(x) - est.P(x)) for x in range(hi))
    assert diff < 0.01


def test_reweighted_distribution_valid():
    w = [1.0 / 9] * 9
    d = estimate_distribution(3, 20_000, random.Random(11), weights_by_blank_cell=w)
    c = d.cumulative
    assert np.all(np.diff(c) >= -1e-12)
    assert abs(c[-1] - 1.0) < 1e-9
