"""Tail-error analysis for canonical models on gap_n12 and rand_n12.

Retrain canonical models (seed 0, full features, standard split_by_state) on
each dataset.  For every method (gbm, mlp, kre, cdp1) compute per-test-row
absolute log10 error  |log10 pred - log10 actual|.

Report per method:
  - median absolute log10 error
  - p90 absolute log10 error
  - p95 absolute log10 error
  - max absolute log10 error
  - max factor error  (10 ^ p95, i.e. worst-5%-tile factor)
  - n_under10x_all: count of >10x UNDERpredictions (pred < actual/10,
    i.e. signed error < -1 in log10 units) overall
  - n_under10x_top_decile: same but restricted to rows where actual
    log10(nodes) >= the 90th percentile of actual log10(nodes) on the test set

Output: results/tail_errors.csv  +  printed table.
"""

import random
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effortpred.cdp import cdp1_predict
from effortpred.conditional import H_MAX_PANCAKE, sample_conditional_matrix
from effortpred.models import fit_gbm, fit_mlp, split_by_state
from effortpred.pancake import estimate_pancake_distribution, gap_h, rand_h
from effortpred.pancake_features import PANCAKE_FEATURE_NAMES
from effortpred.pancake_tree import kre_predict_pancake

DATASETS = [
    {
        "name": "gap_n12",
        "file": "results/pancake_labels_gap.csv",
        "h_col": "h_gap",
        "h_fn": gap_h,
        "n": 12,
    },
    {
        "name": "rand_n12",
        "file": "results/pancake_labels_rand.csv",
        "h_col": "h_rand",
        "h_fn": rand_h,
        "n": 12,
    },
]


def run_dataset(ds, all_rows):
    name = ds["name"]
    h_col = ds["h_col"]
    h_fn = ds["h_fn"]
    n = ds["n"]

    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}  (n={n})")
    print(f"{'=' * 60}")

    df = pd.read_csv(ds["file"])
    n_cens = int(df["censored"].sum())
    df = df[df["censored"] == 0].copy()
    df["y"] = np.log10(df["nodes"].astype(float))
    print(f"Dropped {n_cens} censored rows; {len(df)} rows remain")

    # Standard canonical split (seed 0)
    train, val, test = split_by_state(df, seed=0)
    X_tr = train[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_tr = train["y"].values
    X_va = val[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_va = val["y"].values
    X_te = test[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_te = test["y"].values

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # ---- Learned models ----
    print("Fitting GBM...")
    gbm = fit_gbm(X_tr, y_tr, seed=0)
    pred_gbm = gbm.predict(X_te)

    print("Fitting MLP...")
    pred_mlp_fn = fit_mlp(X_tr, y_tr, X_va, y_va, seed=0)
    pred_mlp = pred_mlp_fn(X_te)

    # ---- Analytic: KRE + CDP1 ----
    print(f"Sampling KRE distribution (n={n}, 200k)...")
    dist = estimate_pancake_distribution(n, h_fn, 200_000, random.Random(100))

    print(f"Sampling CDP_1 conditional matrix (n={n}, 200k)...")
    cond = sample_conditional_matrix(
        n, h_fn, 200_000, random.Random(200), h_max=H_MAX_PANCAKE(n)
    )

    h_vals = test[h_col].astype(int).values
    bound_vals = test["bound"].astype(int).values

    kre_cache = {}
    pred_kre = np.empty(len(test))
    pred_cdp1 = np.empty(len(test))

    for i, (h0, bound) in enumerate(zip(h_vals, bound_vals)):
        if bound not in kre_cache:
            kre_cache[bound] = np.log10(max(kre_predict_pancake(n, bound, dist), 1.0))
        pred_kre[i] = kre_cache[bound]
        pred_cdp1[i] = np.log10(max(cdp1_predict(h0, bound, cond, n - 1, n - 2), 1.0))

    preds = {
        "gbm": pred_gbm,
        "mlp": pred_mlp,
        "kre": pred_kre,
        "cdp1": pred_cdp1,
    }

    # ---- Compute tail metrics ----
    # Top-decile threshold: 90th percentile of ACTUAL log10(nodes) in test set
    top_decile_thresh = float(np.percentile(y_te, 90))
    is_top_decile = y_te >= top_decile_thresh

    print(
        f"\nTop-decile threshold: 10^{top_decile_thresh:.2f} "
        f"= {10**top_decile_thresh:.0f} nodes "
        f"({is_top_decile.sum()} rows = top {100*(1-90/100):.0f}%)"
    )

    for method, pred in preds.items():
        signed_err = pred - y_te        # log10(pred/actual)
        abs_err = np.abs(signed_err)

        median_abs = float(np.median(abs_err))
        p90_abs = float(np.percentile(abs_err, 90))
        p95_abs = float(np.percentile(abs_err, 95))
        max_abs = float(np.max(abs_err))
        max_factor_p95 = float(10 ** p95_abs)

        # >10x underprediction: pred < actual/10, i.e. signed_err < -1
        under10x = (signed_err < -1.0)
        n_under_all = int(under10x.sum())
        n_under_top = int((under10x & is_top_decile).sum())

        all_rows.append(
            {
                "dataset": name,
                "method": method,
                "n_test": len(y_te),
                "median_abs_log10": round(median_abs, 4),
                "p90_abs_log10": round(p90_abs, 4),
                "p95_abs_log10": round(p95_abs, 4),
                "max_abs_log10": round(max_abs, 4),
                "factor_at_p95": round(max_factor_p95, 2),
                "n_under10x_all": n_under_all,
                "n_under10x_top_decile": n_under_top,
                "top_decile_thresh_log10": round(top_decile_thresh, 3),
            }
        )

    # ---- Print table ----
    table_rows = [r for r in all_rows if r["dataset"] == name]
    df_out = pd.DataFrame(table_rows)
    print(f"\nTail errors for {name}:")
    cols = [
        "method",
        "median_abs_log10",
        "p90_abs_log10",
        "p95_abs_log10",
        "max_abs_log10",
        "factor_at_p95",
        "n_under10x_all",
        "n_under10x_top_decile",
    ]
    print(df_out[cols].to_string(index=False))


def main():
    all_rows = []
    for ds in DATASETS:
        run_dataset(ds, all_rows)

    out_df = pd.DataFrame(all_rows)
    out_path = "results/tail_errors.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n\nWrote {out_path}")

    # Final headline summary
    print("\n=== HEADLINE TABLE ===")
    headline_cols = [
        "dataset",
        "method",
        "factor_at_p95",
        "n_under10x_all",
        "n_under10x_top_decile",
    ]
    print(out_df[headline_cols].to_string(index=False))


if __name__ == "__main__":
    main()
