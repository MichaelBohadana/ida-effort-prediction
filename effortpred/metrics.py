"""Evaluation metrics in log10 space (search effort spans orders of
magnitude) + cluster-bootstrap standard errors."""

import numpy as np
from scipy.stats import spearmanr


def eval_log10_predictions(y_true_log10, y_pred_log10):
    y_true_log10 = np.asarray(y_true_log10, dtype=float)
    y_pred_log10 = np.asarray(y_pred_log10, dtype=float)
    err = y_pred_log10 - y_true_log10
    abs_err = np.abs(err)
    return {
        "rmse_log10": float(np.sqrt(np.mean(err ** 2))),
        "median_factor": float(10 ** np.median(abs_err)),
        "within_2x": float(np.mean(abs_err <= np.log10(2))),
        "within_5x": float(np.mean(abs_err <= np.log10(5))),
        "within_10x": float(np.mean(abs_err <= 1.0)),
        "spearman": float(spearmanr(y_true_log10, y_pred_log10).statistic),
    }


def cluster_bootstrap_se(states, y_true, y_pred, metric_key, n_boot=1000, seed=0):
    """SE of a metric via bootstrap over STATES (clusters), not rows: rows of
    the same state at different bounds are correlated, so a row-level
    bootstrap would understate the SE."""
    rng = np.random.default_rng(seed)
    states = np.asarray(states)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    uniq = np.unique(states)
    idx_by_state = {s: np.flatnonzero(states == s) for s in uniq}
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_state[s] for s in chosen])
        vals.append(eval_log10_predictions(y_true[idx], y_pred[idx])[metric_key])
    return float(np.std(vals, ddof=1))
