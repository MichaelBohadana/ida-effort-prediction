import random

import numpy as np

from effortpred.cdp import cdp1_predict
from effortpred.conditional import exact_conditional_matrix
from effortpred.pancake import (
    exact_pancake_distribution, random_pancake_state, gap_h, rand_h,
)
from effortpred.pancake_count import count_expansions_pancake
from effortpred.pancake_tree import kre_predict_pancake, nodes_per_depth_pancake


def test_bound_zero():
    cond = np.array([[1.0]])
    assert cdp1_predict(0, 0, cond, b_root=6, b_rest=5) == 1.0   # root fertile
    assert cdp1_predict(1, 0, cond, b_root=6, b_rest=5) == 0.0   # h > bound


def test_degenerate_heuristic_gives_brute_force():
    """With h == 0 everywhere (cond = [[1.0]], h_start = 0), every generated
    node has v=0 <= d-i, so CDP must equal the full brute-force tree size —
    the same identity the KRE trivial test uses."""
    cond = np.array([[1.0]])
    n, d = 7, 6
    expected = float(sum(nodes_per_depth_pancake(n, d)))
    got = cdp1_predict(0, d, cond, b_root=n - 1, b_rest=n - 2)
    assert abs(got - expected) < 1e-6


def _gate_data(h_fn, n, n_states, offsets, seed):
    cond = exact_conditional_matrix(n, h_fn, h_max=n)
    dist = exact_pancake_distribution(n, h_fn)
    rng = random.Random(seed)
    rows = []
    for _ in range(n_states):
        s = random_pancake_state(n, rng)
        h0 = h_fn(s)
        for off in offsets:
            bound = h0 + off
            actual = count_expansions_pancake(s, bound, h_fn)[0]
            if actual == 0:
                continue
            kre = max(kre_predict_pancake(n, bound, dist), 1.0)
            cdp = max(cdp1_predict(h0, bound, cond, n - 1, n - 2), 1.0)
            rows.append((actual, kre, cdp))
    return np.array(rows, dtype=float)


def _median_factor(pred, actual):
    r = np.log10(pred) - np.log10(actual)
    return 10 ** np.median(np.abs(r))


def test_cdp_gate_consistent():
    """THE GATE, part 1 (consistent gap_h): per single start states, CDP's
    median factor error beats KRE's, and CDP is absolutely sane."""
    rows = _gate_data(gap_h, n=7, n_states=150, offsets=(2, 4), seed=60)
    actual, kre, cdp = rows[:, 0], rows[:, 1], rows[:, 2]
    mf_kre = _median_factor(kre, actual)
    mf_cdp = _median_factor(cdp, actual)
    ratio_cdp = cdp.mean() / actual.mean()
    print(f"\n[consistent] median-factor KRE={mf_kre:.2f} CDP={mf_cdp:.2f}; "
          f"CDP mean-ratio={ratio_cdp:.2f} over {len(rows)} rows")
    assert mf_cdp < mf_kre
    assert 0.2 < ratio_cdp < 5.0


def test_cdp_gate_inconsistent():
    """THE GATE, part 2 (inconsistent rand_h): same relative claim, PLUS the
    paper's signature: KRE systematically OVERESTIMATES under inconsistency
    (it counts fertile-but-unreachable nodes), while CDP does not."""
    rows = _gate_data(rand_h, n=7, n_states=150, offsets=(2, 4), seed=61)
    actual, kre, cdp = rows[:, 0], rows[:, 1], rows[:, 2]
    mf_kre = _median_factor(kre, actual)
    mf_cdp = _median_factor(cdp, actual)
    mean_ratio_kre = kre.mean() / actual.mean()
    ratio_cdp = cdp.mean() / actual.mean()
    print(f"\n[inconsistent] median-factor KRE={mf_kre:.2f} CDP={mf_cdp:.2f}; "
          f"KRE mean-ratio={mean_ratio_kre:.2f}, CDP mean-ratio={ratio_cdp:.2f}")
    assert mf_cdp < mf_kre
    assert mean_ratio_kre > 1.0          # KRE overestimates (paper §3.2.1)
    assert 0.2 < ratio_cdp < 5.0


def test_out_of_support_h_start_raises():
    """h_start <= bound but outside the matrix support must FAIL LOUDLY, not
    silently return 0 (which would drop CDP_1's root term — found by the
    cross-model audit)."""
    import pytest
    cond = np.array([[1.0]])          # supports h values {0} only
    with pytest.raises(ValueError):
        cdp1_predict(2, 5, cond, b_root=6, b_rest=5)
