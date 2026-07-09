"""Multi-seed split robustness: does the train/test split choice change
the method ranking?

For each heuristic and each split seed, train GBM+MLP on both feature sets
and compare against the analytic predictors (KRE, CDP_1). Analytic inputs are
sampled ONCE with fixed RNG seeds (distribution: seed 100, conditional: seed
200) so the only variation is the train/test partition.

Usage:
    python scripts/robustness_seeds.py [--heuristics gap rand]
                                       [--seeds 0 1 2 3 4]
                                       [--n 12]
                                       [--n-dist-samples 200000]
                                       [--n-cond-samples 200000]
                                       [--out results/robustness_seeds.csv]
"""

import argparse
import random
import sys
import os

import numpy as np
import pandas as pd

# Make sure project root is importable when run as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from effortpred.cdp import cdp1_predict
from effortpred.conditional import H_MAX_PANCAKE, sample_conditional_matrix
from effortpred.metrics import eval_log10_predictions
from effortpred.models import fit_gbm, fit_mlp, split_by_state
from effortpred.pancake import estimate_pancake_distribution, gap_h, rand_h
from effortpred.pancake_features import PANCAKE_FEATURE_NAMES
from effortpred.pancake_tree import kre_predict_pancake

HEURISTICS = {"gap": gap_h, "rand": rand_h}


def run_heuristic(heuristic_name, args):
    h_fn = HEURISTICS[heuristic_name]
    n = args.n
    h_col = f"h_{heuristic_name}"

    print(f"\n{'='*60}")
    print(f"  Heuristic: {heuristic_name.upper()}  (n={n})")
    print(f"{'='*60}")

    # --- Load data ---
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "results",
        f"pancake_labels_{heuristic_name}.csv"
    )
    df = pd.read_csv(csv_path)
    n_censored = int(df["censored"].sum())
    df = df[df["censored"] == 0].copy()
    if n_censored:
        print(f"  Dropped {n_censored} censored rows")
    df["y"] = np.log10(df["nodes"].astype(float))

    # Verify n against actual state size (guard against wrong --n).
    state_len = len(df["state"].iloc[0].split())
    assert state_len == n, (
        f"--n {n} does not match dataset state size {state_len}")

    # --- Compute analytic inputs ONCE with fixed seeds ---
    # Using fixed seeds means the only source of variation across split seeds
    # is the train/test partition — exactly what we want to measure.
    print(f"  Sampling P(x) distribution (n_samples={args.n_dist_samples}, rng-seed=100) ...")
    dist = estimate_pancake_distribution(
        n, h_fn, args.n_dist_samples, random.Random(100))

    print(f"  Sampling conditional matrix (n_samples={args.n_cond_samples}, rng-seed=200) ...")
    cond = sample_conditional_matrix(
        n, h_fn, args.n_cond_samples,
        random.Random(200), h_max=H_MAX_PANCAKE(n))

    # --- KRE cache: keyed by bound (state-blind) ---
    bounds_all = sorted(df["bound"].unique())
    kre_cache = {}
    for b in bounds_all:
        raw = kre_predict_pancake(n, b, dist)
        kre_cache[b] = float(np.log10(max(raw, 1.0)))

    # --- CDP cache: keyed by (h0, bound) ---
    pairs_all = df[[h_col, "bound"]].drop_duplicates()
    cdp_cache = {}
    for _, row in pairs_all.iterrows():
        h0, bound = int(row[h_col]), int(row["bound"])
        raw = cdp1_predict(h0, bound, cond, n - 1, n - 2)
        cdp_cache[(h0, bound)] = float(np.log10(max(raw, 1.0)))

    # --- Feature sets ---
    full_features = PANCAKE_FEATURE_NAMES
    minimal_features = [h_col, "bound"]

    # --- Per-seed loop ---
    all_rows = []
    for seed in args.seeds:
        print(f"\n  Seed {seed} ...")
        train, val, test = split_by_state(df, seed=seed)

        X_tr_full = train[full_features].values.astype(float)
        X_va_full = val[full_features].values.astype(float)
        X_te_full = test[full_features].values.astype(float)
        X_tr_min = train[minimal_features].values.astype(float)
        X_va_min = val[minimal_features].values.astype(float)
        X_te_min = test[minimal_features].values.astype(float)
        y_tr = train["y"].values
        y_va = val["y"].values
        y_te = test["y"].values

        # GBM — full
        gbm_full = fit_gbm(X_tr_full, y_tr, seed=seed)
        pred_gbm_full = gbm_full.predict(X_te_full)

        # GBM — minimal
        gbm_min = fit_gbm(X_tr_min, y_tr, seed=seed)
        pred_gbm_min = gbm_min.predict(X_te_min)

        # MLP — full
        mlp_full_fn = fit_mlp(X_tr_full, y_tr, X_va_full, y_va, seed=seed)
        pred_mlp_full = mlp_full_fn(X_te_full)

        # MLP — minimal
        mlp_min_fn = fit_mlp(X_tr_min, y_tr, X_va_min, y_va, seed=seed)
        pred_mlp_min = mlp_min_fn(X_te_min)

        # Analytic predictors on the same test rows
        pred_kre = np.array([kre_cache[int(b)] for b in test["bound"]])
        pred_cdp = np.array([cdp_cache[(int(row[h_col]), int(row["bound"]))]
                             for _, row in test[[h_col, "bound"]].iterrows()])

        preds_by_method = {
            "gbm_full":    pred_gbm_full,
            "mlp_full":    pred_mlp_full,
            "gbm_minimal": pred_gbm_min,
            "mlp_minimal": pred_mlp_min,
            "cdp1":        pred_cdp,
            "kre":         pred_kre,
        }

        for method_name, pred in preds_by_method.items():
            rmse = eval_log10_predictions(y_te, pred)["rmse_log10"]
            all_rows.append({
                "heuristic": heuristic_name,
                "seed":      seed,
                "method":    method_name,
                "rmse_log10": rmse,
            })
            print(f"    {method_name:15s}  rmse_log10={rmse:.4f}")

    return all_rows


def print_summary(df_results, heuristic_name):
    """Print per-method mean/std/min/max over seeds, and an ordering verdict."""
    subset = df_results[df_results["heuristic"] == heuristic_name].copy()
    pivot = subset.groupby("method")["rmse_log10"].agg(
        mean="mean", std="std", min="min", max="max"
    ).reset_index()
    pivot = pivot.sort_values("mean")

    print(f"\n{'='*60}")
    print(f"  Summary for {heuristic_name.upper()}")
    print(f"{'='*60}")
    print(pivot.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Ordering verdict ---
    # Expected: (mlp_full or gbm_full) < cdp1 < kre
    # and minimal-learned < cdp1
    n_seeds = subset["seed"].nunique()
    by_seed = subset.pivot(index="seed", columns="method", values="rmse_log10")

    violations = []
    for seed_val, row in by_seed.iterrows():
        # mlp_full < cdp1
        if not row["mlp_full"] < row["cdp1"]:
            violations.append(f"seed={seed_val}: mlp_full ({row['mlp_full']:.4f}) >= cdp1 ({row['cdp1']:.4f})")
        # gbm_full < cdp1
        if not row["gbm_full"] < row["cdp1"]:
            violations.append(f"seed={seed_val}: gbm_full ({row['gbm_full']:.4f}) >= cdp1 ({row['cdp1']:.4f})")
        # cdp1 < kre
        if not row["cdp1"] < row["kre"]:
            violations.append(f"seed={seed_val}: cdp1 ({row['cdp1']:.4f}) >= kre ({row['kre']:.4f})")
        # mlp_minimal < cdp1
        if not row["mlp_minimal"] < row["cdp1"]:
            violations.append(f"seed={seed_val}: mlp_minimal ({row['mlp_minimal']:.4f}) >= cdp1 ({row['cdp1']:.4f})")
        # gbm_minimal < cdp1
        if not row["gbm_minimal"] < row["cdp1"]:
            violations.append(f"seed={seed_val}: gbm_minimal ({row['gbm_minimal']:.4f}) >= cdp1 ({row['cdp1']:.4f})")

    print()
    if not violations:
        print(f"  VERDICT [{heuristic_name}]: Ordering mlp/gbm < cdp1 < kre AND minimal-learned < cdp1 "
              f"HOLDS on ALL {n_seeds} seeds. Split choice does NOT change conclusions.")
    else:
        print(f"  VERDICT [{heuristic_name}]: Ordering VIOLATED on {len(violations)} case(s):")
        for v in violations:
            print(f"    - {v}")

    return by_seed


def main():
    ap = argparse.ArgumentParser(
        description="Multi-seed split robustness for pancake effort prediction")
    ap.add_argument("--heuristics", nargs="+", default=["gap", "rand"],
                    choices=["gap", "rand"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--n-dist-samples", type=int, default=200_000)
    ap.add_argument("--n-cond-samples", type=int, default=200_000)
    ap.add_argument("--out", default="results/robustness_seeds.csv")
    args = ap.parse_args()

    # Safety: refuse to overwrite existing canonical eval files.
    protected = [
        "results/pancake_eval_gap.csv",
        "results/pancake_eval_rand.csv",
        "results/pancake_eval_gap_minimal.csv",
        "results/pancake_eval_rand_minimal.csv",
        "results/pancake_labels_gap.csv",
        "results/pancake_labels_rand.csv",
    ]
    if args.out in protected:
        raise ValueError(f"Output path {args.out!r} would overwrite a protected file.")

    all_rows = []
    for h in args.heuristics:
        rows = run_heuristic(h, args)
        all_rows.extend(rows)

    df_results = pd.DataFrame(all_rows)

    # Write long-format CSV.
    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path} ({len(df_results)} rows)")

    # Print summary tables and verdicts.
    for h in args.heuristics:
        print_summary(df_results, h)

    print("\nDone.")


if __name__ == "__main__":
    main()
