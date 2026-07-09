import numpy as np

from effortpred.metrics import cluster_bootstrap_se, eval_log10_predictions


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = eval_log10_predictions(y, y)
    assert m["rmse_log10"] == 0.0
    assert m["median_factor"] == 1.0
    assert m["within_2x"] == 1.0
    assert m["spearman"] == 1.0


def test_off_by_one_decade():
    y = np.array([1.0, 2.0, 3.0])
    p = y + 1.0                       # predictions 10x too high everywhere
    m = eval_log10_predictions(y, p)
    assert abs(m["rmse_log10"] - 1.0) < 1e-12
    assert abs(m["median_factor"] - 10.0) < 1e-9
    assert m["within_2x"] == 0.0
    assert m["within_10x"] == 1.0     # 10x counts as within a factor of 10
    assert m["spearman"] == 1.0       # ordering preserved


def test_bootstrap_se_positive_and_deterministic():
    rng = np.random.default_rng(0)
    states = np.repeat([f"s{i}" for i in range(40)], 5)
    y = rng.normal(3, 1, size=200)
    p = y + rng.normal(0, 0.3, size=200)
    se1 = cluster_bootstrap_se(states, y, p, "rmse_log10", n_boot=200, seed=1)
    se2 = cluster_bootstrap_se(states, y, p, "rmse_log10", n_boot=200, seed=1)
    assert se1 > 0
    assert se1 == se2
