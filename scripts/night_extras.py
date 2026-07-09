"""Overnight additive experiments (2026-07-02 night run). Creates ONLY new
result files — never touches canonical outputs.

1. MLP permutation importance (manual: shuffle each feature column on the
   held-out test set, measure RMSE increase; n_repeats shuffles per feature).
   Outputs: results/mlp_importance_{heuristic}.csv + .png
2. Prediction-latency microbenchmark: per-row wall time of GBM / MLP / KRE /
   CDP_1 predictions on the test rows. Output: results/prediction_latency.csv
"""

import argparse
import random
import time

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from effortpred.cdp import cdp1_predict
from effortpred.conditional import H_MAX_PANCAKE, sample_conditional_matrix
from effortpred.models import fit_gbm, fit_mlp, split_by_state
from effortpred.pancake import estimate_pancake_distribution, gap_h, rand_h
from effortpred.pancake_features import PANCAKE_FEATURE_NAMES
from effortpred.pancake_tree import kre_predict_pancake

N = 12
HEURISTICS = {"gap": gap_h, "rand": rand_h}


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def mlp_importance(heuristic_name, seed, n_repeats, rng_seed):
    df = pd.read_csv(f"results/pancake_labels_{heuristic_name}.csv")
    df = df[df["censored"] == 0].copy()
    df["y"] = np.log10(df["nodes"].astype(float))
    train, val, test = split_by_state(df, seed=seed)
    X_tr = train[PANCAKE_FEATURE_NAMES].values.astype(float)
    X_va = val[PANCAKE_FEATURE_NAMES].values.astype(float)
    X_te = test[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_tr, y_va, y_te = train["y"].values, val["y"].values, test["y"].values

    predict = fit_mlp(X_tr, y_tr, X_va, y_va, seed=seed)
    base = rmse(predict(X_te), y_te)

    rng = np.random.default_rng(rng_seed)
    rows = []
    for j, name in enumerate(PANCAKE_FEATURE_NAMES):
        deltas = []
        for _ in range(n_repeats):
            Xp = X_te.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            deltas.append(rmse(predict(Xp), y_te) - base)
        rows.append({"feature": name,
                     "importance_mean": float(np.mean(deltas)),
                     "importance_std": float(np.std(deltas, ddof=1))})
    out = pd.DataFrame(rows).sort_values("importance_mean")
    out.to_csv(f"results/mlp_importance_{heuristic_name}.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.barh(out["feature"], out["importance_mean"], xerr=out["importance_std"])
    plt.xlabel("permutation importance (increase in rmse_log10)")
    plt.title(f"MLP feature importance ({heuristic_name})")
    plt.tight_layout()
    plt.savefig(f"results/mlp_importance_{heuristic_name}.png", dpi=150)
    plt.close()
    print(f"[{heuristic_name}] MLP base rmse {base:.4f}; top features:")
    print(out.tail(3).to_string(index=False))
    return base


def latency(heuristic_name, seed, args):
    h_fn = HEURISTICS[heuristic_name]
    df = pd.read_csv(f"results/pancake_labels_{heuristic_name}.csv")
    df = df[df["censored"] == 0].copy()
    df["y"] = np.log10(df["nodes"].astype(float))
    train, val, test = split_by_state(df, seed=seed)
    X_tr = train[PANCAKE_FEATURE_NAMES].values.astype(float)
    X_va = val[PANCAKE_FEATURE_NAMES].values.astype(float)
    X_te = test[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_tr, y_va = train["y"].values, val["y"].values

    gbm = fit_gbm(X_tr, y_tr, seed=seed)
    mlp = fit_mlp(X_tr, y_tr, X_va, y_va, seed=seed)
    dist = estimate_pancake_distribution(N, h_fn, args.n_dist_samples,
                                         random.Random(100))
    cond = sample_conditional_matrix(N, h_fn, args.n_cond_samples,
                                     random.Random(200),
                                     h_max=H_MAX_PANCAKE(N))
    h_col = f"h_{heuristic_name}"
    pairs = list(zip(test[h_col].astype(int), test["bound"].astype(int)))

    rows = []
    t0 = time.perf_counter()
    gbm.predict(X_te)
    rows.append(("gbm_batch", (time.perf_counter() - t0) / len(X_te)))
    t0 = time.perf_counter()
    mlp(X_te)
    rows.append(("mlp_batch", (time.perf_counter() - t0) / len(X_te)))
    t0 = time.perf_counter()
    for h0, bound in pairs:
        kre_predict_pancake(N, bound, dist)          # deliberately uncached
    rows.append(("kre_per_row", (time.perf_counter() - t0) / len(pairs)))
    t0 = time.perf_counter()
    for h0, bound in pairs:
        cdp1_predict(h0, bound, cond, N - 1, N - 2)
    rows.append(("cdp1_per_row", (time.perf_counter() - t0) / len(pairs)))

    out = pd.DataFrame(rows, columns=["method", "seconds_per_prediction"])
    out["heuristic"] = heuristic_name
    print(out.to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--n-dist-samples", type=int, default=200_000)
    ap.add_argument("--n-cond-samples", type=int, default=200_000)
    args = ap.parse_args()

    lat = []
    for h in ("gap", "rand"):
        mlp_importance(h, args.seed, args.n_repeats, rng_seed=args.seed + 500)
        lat.append(latency(h, args.seed, args))
    pd.concat(lat).to_csv("results/prediction_latency.csv", index=False)
    print("night_extras done")


if __name__ == "__main__":
    main()
