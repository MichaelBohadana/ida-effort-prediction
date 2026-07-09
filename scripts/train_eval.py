"""Fit all predictors on the labeled data, evaluate on held-out states,
compare against KRE, and produce the report tables/figures."""

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from effortpred.distribution import HDistribution
from effortpred.features import FEATURE_NAMES
from effortpred.kre import kre_predict
from effortpred.metrics import cluster_bootstrap_se, eval_log10_predictions
from effortpred.models import (
    GapBaseline, MeanBaseline, fit_gbm, fit_mlp, split_by_state,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/labels.csv")
    ap.add_argument("--dist", default="results/hdist_overall.npy")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data).sort_values(["state", "bound"]).reset_index(drop=True)
    n_censored = int(df["censored"].sum())
    df = df[df["censored"] == 0].copy()
    print(f"dropped {n_censored} censored rows (report this in the paper)")
    df["y"] = np.log10(df["nodes"].astype(float))

    train, val, test = split_by_state(df, seed=args.seed)
    print(f"split sizes: train={len(train)} val={len(val)} test={len(test)} rows")

    X_tr = train[FEATURE_NAMES].values.astype(float)
    X_va = val[FEATURE_NAMES].values.astype(float)
    X_te = test[FEATURE_NAMES].values.astype(float)
    y_tr, y_va, y_te = train["y"].values, val["y"].values, test["y"].values

    preds = {}
    preds["mean_baseline"] = MeanBaseline().fit(X_tr, y_tr).predict(X_te)
    preds["gap_baseline"] = (
        GapBaseline().fit(train["gap"].values, y_tr).predict(test["gap"].values)
    )
    gbm = fit_gbm(X_tr, y_tr, seed=args.seed)
    preds["gbm"] = gbm.predict(X_te)
    preds["mlp"] = fit_mlp(X_tr, y_tr, X_va, y_va, seed=args.seed)(X_te)

    # KRE on the same test rows. N_i depends only on (blank cell, bound).
    dist = HDistribution(np.load(args.dist))
    kre_cache = {}
    kre_log = []
    for state_str, bound in zip(test["state"], test["bound"]):
        state = tuple(int(x) for x in state_str.split())
        key = (state.index(0), int(bound))
        if key not in kre_cache:
            kre_cache[key] = kre_predict(state, 4, int(bound), dist)
        kre_log.append(np.log10(max(kre_cache[key], 1.0)))
    preds["kre"] = np.array(kre_log)

    rows = []
    for name, p in preds.items():
        m = eval_log10_predictions(y_te, p)
        m["rmse_log10_se"] = cluster_bootstrap_se(
            test["state"].values, y_te, p, "rmse_log10", seed=args.seed
        )
        m["spearman_se"] = cluster_bootstrap_se(
            test["state"].values, y_te, p, "spearman", seed=args.seed
        )
        m["median_factor_se"] = cluster_bootstrap_se(
            test["state"].values, y_te, p, "median_factor", seed=args.seed
        )
        rows.append({"method": name, **m})
    table = pd.DataFrame(rows)
    table.to_csv(f"{args.out_dir}/eval_table.csv", index=False)
    print(table.to_string(index=False))

    # predicted-vs-actual scatter per method
    for name, p in preds.items():
        plt.figure(figsize=(5, 5))
        plt.scatter(y_te, p, s=4, alpha=0.4)
        lim = [min(y_te.min(), p.min()), max(y_te.max(), p.max())]
        plt.plot(lim, lim, "k--", lw=1)
        plt.xlabel("actual log10(nodes)")
        plt.ylabel("predicted log10(nodes)")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(f"{args.out_dir}/scatter_{name}.png", dpi=150)
        plt.close()

    # permutation feature importance for the GBM (on held-out data)
    imp = permutation_importance(gbm, X_te, y_te, n_repeats=20,
                                 random_state=args.seed)
    order = np.argsort(imp.importances_mean)
    plt.figure(figsize=(6, 4))
    plt.barh([FEATURE_NAMES[i] for i in order], imp.importances_mean[order],
             xerr=imp.importances_std[order])
    plt.xlabel("permutation importance (drop in R^2)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/feature_importance.png", dpi=150)
    plt.close()
    print(f"wrote eval_table.csv, scatter_*.png, feature_importance.png to {args.out_dir}/")


if __name__ == "__main__":
    main()
