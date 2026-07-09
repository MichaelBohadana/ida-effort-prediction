"""Phase 3c — probe-feature tier evaluation.

For each heuristic dataset (gap, rand) evaluate four feature tiers with
GBM and MLP, compared against the analytic KRE and CDP_1 baselines:

  a. minimal       — [h_<active>, "bound"]
  b. minimal_probe — minimal + the 6 probe features of the ACTIVE heuristic
  c. full          — the 13 PANCAKE_FEATURE_NAMES
  d. full_probe    — 13 + all 12 probe features

Analytic inputs use the same fixed seeds as robustness_seeds.py so results
are directly comparable:
  P(x)  distribution: random.Random(100), 200k samples
  p(v|vp) conditional: random.Random(200), 200k samples

The split is canonical: df sorted by ["state","bound"], split_by_state seed=0.

HONESTY RULE: no required direction.  If probe features add nothing, that is
the finding — report it plainly.

Usage:
    python scripts/probe_eval.py [--n 12] [--n-dist-samples 200000]
                                  [--n-cond-samples 200000]
                                  [--out results/probe_eval.csv]
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from effortpred.cdp import cdp1_predict
from effortpred.conditional import H_MAX_PANCAKE, sample_conditional_matrix
from effortpred.metrics import eval_log10_predictions
from effortpred.models import fit_gbm, fit_mlp, split_by_state
from effortpred.pancake import estimate_pancake_distribution, gap_h, rand_h
from effortpred.pancake_features import PANCAKE_FEATURE_NAMES
from effortpred.pancake_tree import kre_predict_pancake
from effortpred.probe_features import PROBE_FEATURE_NAMES, extract_probe_features

HEURISTICS = {"gap": gap_h, "rand": rand_h}

# Safety guard: these paths must never be written by this script.
_PROTECTED_PREFIXES = (
    "results/pancake_eval_",
    "results/pancake_labels_",
)


def _safe_out(path):
    for p in _PROTECTED_PREFIXES:
        if path.startswith(p):
            raise ValueError(
                f"Output path {path!r} would overwrite a protected file. "
                "Refusing to write."
            )
    return path


def _parse_state(s: str) -> tuple:
    return tuple(int(x) for x in s.split())


def run_heuristic(heuristic_name: str, args) -> list[dict]:
    h_fn = HEURISTICS[heuristic_name]
    n = args.n
    h_col = f"h_{heuristic_name}"

    print(f"\n{'='*64}")
    print(f"  Heuristic: {heuristic_name.upper()}   (n={n})")
    print(f"{'='*64}")

    # --- Load data ---
    csv_path = os.path.join(
        os.path.dirname(__file__), "..", "results",
        f"pancake_labels_{heuristic_name}.csv",
    )
    df = pd.read_csv(csv_path)
    n_total = len(df)
    n_censored = int(df["censored"].sum())
    df = df[df["censored"] == 0].copy()
    print(f"  Loaded {n_total} rows, dropped {n_censored} censored, {len(df)} remain.")

    # Canonical sort (same as train_eval_pancake.py).
    df = df.sort_values(["state", "bound"]).reset_index(drop=True)
    df["y"] = np.log10(df["nodes"].astype(float))

    # Verify n.
    state_len = len(df["state"].iloc[0].split())
    assert state_len == n, (
        f"--n {n} does not match dataset state size {state_len}")

    # --- Compute probe features for every row ---
    print("  Computing probe features for all rows ...")
    t0 = time.perf_counter()
    probe_rows = [
        extract_probe_features(_parse_state(row["state"]), int(row["bound"]))
        for _, row in df.iterrows()
    ]
    t_all = time.perf_counter() - t0
    print(f"  Done ({len(df)} rows in {t_all:.2f}s)")

    probe_df = pd.DataFrame(probe_rows, index=df.index)
    df_aug = pd.concat([df, probe_df], axis=1)

    # --- Analytic inputs with fixed seeds (matching robustness_seeds.py) ---
    print(f"  Sampling P(x) distribution (n_samples={args.n_dist_samples}, seed=100) ...")
    dist = estimate_pancake_distribution(
        n, h_fn, args.n_dist_samples, random.Random(100))

    print(f"  Sampling conditional matrix (n_samples={args.n_cond_samples}, seed=200) ...")
    cond = sample_conditional_matrix(
        n, h_fn, args.n_cond_samples,
        random.Random(200), h_max=H_MAX_PANCAKE(n))

    # --- Canonical split: seed=0 ---
    train, val, test = split_by_state(df_aug, seed=0)
    y_tr = train["y"].values
    y_va = val["y"].values
    y_te = test["y"].values

    # --- Feature tiers ---
    # Only the 6 probe features for the ACTIVE heuristic (not both).
    active_probe = [name for name in PROBE_FEATURE_NAMES
                    if f"probe_{heuristic_name}_" in name]
    assert len(active_probe) == 6, active_probe

    tiers = {
        "minimal":       [h_col, "bound"],
        "minimal_probe": [h_col, "bound"] + active_probe,
        "full":          list(PANCAKE_FEATURE_NAMES),
        "full_probe":    list(PANCAKE_FEATURE_NAMES) + list(PROBE_FEATURE_NAMES),
    }

    # --- KRE cache (state-blind: keyed by bound) ---
    kre_cache = {}
    for b in df_aug["bound"].unique():
        raw = kre_predict_pancake(n, int(b), dist)
        kre_cache[int(b)] = float(np.log10(max(raw, 1.0)))

    # --- CDP cache (keyed by (h0, bound)) ---
    pairs_all = df_aug[[h_col, "bound"]].drop_duplicates()
    cdp_cache = {}
    for _, row in pairs_all.iterrows():
        h0, bound = int(row[h_col]), int(row["bound"])
        raw = cdp1_predict(h0, bound, cond, n - 1, n - 2)
        cdp_cache[(h0, bound)] = float(np.log10(max(raw, 1.0)))

    pred_kre = np.array([kre_cache[int(b)] for b in test["bound"]])
    pred_cdp = np.array([
        cdp_cache[(int(row[h_col]), int(row["bound"]))]
        for _, row in test[[h_col, "bound"]].iterrows()
    ])

    kre_rmse = eval_log10_predictions(y_te, pred_kre)["rmse_log10"]
    cdp_rmse = eval_log10_predictions(y_te, pred_cdp)["rmse_log10"]

    # --- Train & evaluate per tier ---
    all_rows = []
    print(f"\n  {'tier':<15} {'method':<6}  {'rmse_log10':>12}")
    print(f"  {'-'*38}")
    for tier_name, feats in tiers.items():
        X_tr = train[feats].values.astype(float)
        X_va = val[feats].values.astype(float)
        X_te = test[feats].values.astype(float)

        gbm = fit_gbm(X_tr, y_tr, seed=0)
        pred_gbm = gbm.predict(X_te)
        rmse_gbm = eval_log10_predictions(y_te, pred_gbm)["rmse_log10"]

        mlp_fn = fit_mlp(X_tr, y_tr, X_va, y_va, seed=0)
        pred_mlp = mlp_fn(X_te)
        rmse_mlp = eval_log10_predictions(y_te, pred_mlp)["rmse_log10"]

        print(f"  {tier_name:<15} {'GBM':<6}  {rmse_gbm:>12.4f}")
        print(f"  {tier_name:<15} {'MLP':<6}  {rmse_mlp:>12.4f}")

        all_rows.append({
            "heuristic": heuristic_name,
            "tier": tier_name,
            "method": "gbm",
            "rmse_log10": rmse_gbm,
        })
        all_rows.append({
            "heuristic": heuristic_name,
            "tier": tier_name,
            "method": "mlp",
            "rmse_log10": rmse_mlp,
        })

    print(f"  {'-'*38}")
    print(f"  {'KRE':<22}  {kre_rmse:>12.4f}")
    print(f"  {'CDP_1':<22}  {cdp_rmse:>12.4f}")

    all_rows.append({
        "heuristic": heuristic_name,
        "tier": "analytic",
        "method": "kre",
        "rmse_log10": kre_rmse,
    })
    all_rows.append({
        "heuristic": heuristic_name,
        "tier": "analytic",
        "method": "cdp1",
        "rmse_log10": cdp_rmse,
    })

    return all_rows


def time_probe_features(n_calls: int = 1_000):
    """Return mean microseconds per extract_probe_features call."""
    import random as _random
    rng = _random.Random(999)
    n = 12
    states_bounds = [
        (tuple(rng.sample(range(n), n)), rng.randint(1, 20))
        for _ in range(n_calls)
    ]
    t0 = time.perf_counter()
    for state, bound in states_bounds:
        extract_probe_features(state, bound)
    elapsed = time.perf_counter() - t0
    return (elapsed / n_calls) * 1e6  # microseconds


def main():
    ap = argparse.ArgumentParser(
        description="Probe-feature tier evaluation for n-pancake effort prediction")
    ap.add_argument("--heuristics", nargs="+", default=["gap", "rand"],
                    choices=["gap", "rand"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--n-dist-samples", type=int, default=200_000)
    ap.add_argument("--n-cond-samples", type=int, default=200_000)
    ap.add_argument("--out", default="results/probe_eval.csv")
    args = ap.parse_args()

    # Safety check on output path.
    out_path = _safe_out(args.out)

    # Timing.
    print("Timing extract_probe_features (1,000 calls on n=12) ...")
    us_per_call = time_probe_features(1_000)
    print(f"  Mean time: {us_per_call:.1f} µs/call\n")

    all_rows = []
    for h in args.heuristics:
        rows = run_heuristic(h, args)
        all_rows.extend(rows)

    df_results = pd.DataFrame(all_rows)

    # Write CSV.
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df_results.to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path} ({len(df_results)} rows)")

    # Print summary tables.
    print()
    for h in args.heuristics:
        sub = df_results[df_results["heuristic"] == h].copy()
        pivot = sub.pivot_table(
            index="tier", columns="method", values="rmse_log10", aggfunc="first"
        ).reset_index()
        print(f"\n=== {h.upper()} — rmse_log10 by tier ===")
        print(pivot.to_string(index=False))

    print(f"\nTiming: {us_per_call:.1f} µs per extract_probe_features call (n=12)")
    print("\nDone.")


if __name__ == "__main__":
    main()
