"""Growth-ratio extrapolation to an unseen deeper offset (Freeze-Blocker 1).

For each dataset with >= 4 offsets:
  - gap_n12 (offsets 0-3, holdout offset=3)
  - gap_n10 (offsets 0-4, holdout offset=4)
  - rand_n10 (offsets 0-3, holdout offset=3)

Double holdout:
  1. State-grouped split (split_by_state seed 0) on the raw df.
  2. Offset holdout: training uses TRAIN states' ratios at shallow pairs only;
     evaluation uses TEST states' ratio at the DEEPEST pair only.

Target: log10 N(s, o) - log10 N(s, o-1)  [in log10 units]
Features: PANCAKE_FEATURE_NAMES from the deeper row (offset o).

Models: GBM(seed 0), MLP(seed 0, val = val-states' shallow ratios).
Baselines:
  const  -- mean training ratio
  kre    -- log10 KRE(bound_o) - log10 KRE(bound_{o-1}) per state
             (dist sampled once, random.Random(100), 200 k)
  cdp1   -- log10 CDP_1(h_s, bound_o) - log10 CDP_1(h_s, bound_{o-1}) per state
             (cond sampled once, random.Random(200), 200 k)

Metrics:
  rmse_ratio  -- RMSE of the ratio prediction (log10 units)
  rmse_abs    -- RMSE of the reconstructed absolute log10 N
                 (= actual log10 N at o-1 + predicted ratio)

Canonical interpolation RMSE at the holdout offset is printed for
CONTEXT ONLY -- those models saw all offsets during training (different protocol).

Output: results/extrapolation_test.csv + per-dataset printed verdicts.
"""

import random
import sys
import os

import numpy as np
import pandas as pd

# allow running from the project root
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
        "holdout_offset": 3,
        "by_offset_file": "results/pancake_eval_gap_by_offset.csv",
    },
    {
        "name": "gap_n10",
        "file": "results/pancake_labels_gap_n10.csv",
        "h_col": "h_gap",
        "h_fn": gap_h,
        "n": 10,
        "holdout_offset": 4,
        "by_offset_file": "results/pancake_eval_gap_n10_by_offset.csv",
    },
    {
        "name": "rand_n10",
        "file": "results/pancake_labels_rand_n10.csv",
        "h_col": "h_rand",
        "h_fn": rand_h,
        "n": 10,
        "holdout_offset": 3,
        "by_offset_file": "results/pancake_eval_rand_n10_by_offset.csv",
    },
]


def build_ratio_rows(state_df, h_col, holdout_offset, is_eval):
    """Build ratio rows from state_df for the appropriate pairs.

    If is_eval=True: use only the deepest pair (holdout_offset-1, holdout_offset).
    If is_eval=False: use all shallow pairs (consecutive, all < holdout_offset).

    Returns a DataFrame with columns: state, ratio, log10_curr, log10_prev,
    *PANCAKE_FEATURE_NAMES (from deeper row).
    """
    state_df = state_df.copy()
    offsets_present = sorted(state_df["offset"].unique())

    if is_eval:
        pairs = [(holdout_offset - 1, holdout_offset)]
    else:
        shallow = [o for o in offsets_present if o < holdout_offset]
        pairs = [(shallow[i], shallow[i + 1]) for i in range(len(shallow) - 1)]

    parts = []
    for o_prev, o_curr in pairs:
        df_curr = state_df[(state_df["offset"] == o_curr) & (state_df["censored"] == 0)]
        df_prev = state_df[(state_df["offset"] == o_prev) & (state_df["censored"] == 0)]
        if df_curr.empty or df_prev.empty:
            continue
        merged = pd.merge(
            df_curr[["state", "nodes"] + PANCAKE_FEATURE_NAMES].rename(
                columns={"nodes": "nodes_curr"}
            ),
            df_prev[["state", "nodes"]].rename(columns={"nodes": "nodes_prev"}),
            on="state",
        )
        merged["log10_curr"] = np.log10(merged["nodes_curr"].astype(float))
        merged["log10_prev"] = np.log10(merged["nodes_prev"].astype(float))
        merged["ratio"] = merged["log10_curr"] - merged["log10_prev"]
        parts.append(merged)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def run_dataset(ds, all_rows):
    name = ds["name"]
    h_col = ds["h_col"]
    h_fn = ds["h_fn"]
    n = ds["n"]
    holdout_offset = ds["holdout_offset"]

    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}  (n={n}, holdout_offset={holdout_offset})")
    print(f"{'=' * 60}")

    df = pd.read_csv(ds["file"])
    df["offset"] = df["bound"] - df[h_col]

    # Drop censored rows before split (keep structure simple)
    n_cens = int(df["censored"].sum())
    df = df[df["censored"] == 0].copy()
    print(f"Dropped {n_cens} censored rows; {len(df)} rows remain")

    # State-grouped split on the raw df (before ratio construction)
    train_df, val_df, test_df = split_by_state(df, seed=0)
    print(
        f"States: train={train_df['state'].nunique()}, "
        f"val={val_df['state'].nunique()}, "
        f"test={test_df['state'].nunique()}"
    )

    shallow_pairs_desc = list(range(0, holdout_offset))
    if len(shallow_pairs_desc) >= 2:
        pair_list = [(shallow_pairs_desc[i], shallow_pairs_desc[i + 1])
                     for i in range(len(shallow_pairs_desc) - 1)]
    else:
        pair_list = []
    print(f"Shallow pairs: {pair_list}, Eval pair: ({holdout_offset-1}, {holdout_offset})")

    # Build ratio datasets
    train_ratio = build_ratio_rows(train_df, h_col, holdout_offset, is_eval=False)
    val_ratio = build_ratio_rows(val_df, h_col, holdout_offset, is_eval=False)
    eval_ratio = build_ratio_rows(test_df, h_col, holdout_offset, is_eval=True)

    print(
        f"Ratio rows: train={len(train_ratio)}, val={len(val_ratio)}, eval={len(eval_ratio)}"
    )

    if eval_ratio.empty:
        print(f"WARNING: no eval rows for {name} -- skipping")
        return

    X_tr = train_ratio[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_tr = train_ratio["ratio"].values
    X_va = val_ratio[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_va = val_ratio["ratio"].values
    X_te = eval_ratio[PANCAKE_FEATURE_NAMES].values.astype(float)
    y_te = eval_ratio["ratio"].values

    # ---- Fit learned models ----
    print("Fitting GBM...")
    gbm = fit_gbm(X_tr, y_tr, seed=0)
    pred_gbm = gbm.predict(X_te)

    print("Fitting MLP...")
    pred_mlp_fn = fit_mlp(X_tr, y_tr, X_va, y_va, seed=0)
    pred_mlp = pred_mlp_fn(X_te)

    # ---- Constant baseline ----
    mean_ratio = float(np.mean(y_tr))
    pred_const = np.full(len(y_te), mean_ratio)

    # ---- Analytic baselines ----
    print(f"Sampling KRE distribution (n={n}, 200k)...")
    dist = estimate_pancake_distribution(n, h_fn, 200_000, random.Random(100))

    print(f"Sampling CDP_1 conditional matrix (n={n}, 200k)...")
    cond = sample_conditional_matrix(
        n, h_fn, 200_000, random.Random(200), h_max=H_MAX_PANCAKE(n)
    )

    # Per-state KRE and CDP1 ratios
    # eval_ratio features come from the DEEPER row (offset=holdout_offset)
    # h_col is a state property (independent of bound)
    # 'bound' in eval_ratio is the deeper bound = h(s) + holdout_offset
    h_vals = eval_ratio[h_col].values.astype(int)
    bound_deep = eval_ratio["bound"].values.astype(int)
    bound_prev = bound_deep - 1

    kre_cache = {}
    pred_kre = np.empty(len(eval_ratio))
    pred_cdp1 = np.empty(len(eval_ratio))

    for i, (h_s, b_deep, b_prev) in enumerate(zip(h_vals, bound_deep, bound_prev)):
        # KRE (state-independent for pancake - varies only through bound)
        if b_deep not in kre_cache:
            kre_cache[b_deep] = np.log10(max(kre_predict_pancake(n, b_deep, dist), 1.0))
        if b_prev not in kre_cache:
            kre_cache[b_prev] = np.log10(max(kre_predict_pancake(n, b_prev, dist), 1.0))
        pred_kre[i] = kre_cache[b_deep] - kre_cache[b_prev]

        # CDP1 (state-aware through h_s)
        cdp_deep = np.log10(max(cdp1_predict(h_s, b_deep, cond, n - 1, n - 2), 1.0))
        cdp_prev = np.log10(max(cdp1_predict(h_s, b_prev, cond, n - 1, n - 2), 1.0))
        pred_cdp1[i] = cdp_deep - cdp_prev

    # ---- Metrics ----
    log10_prev = eval_ratio["log10_prev"].values
    log10_curr = eval_ratio["log10_curr"].values  # = y_te + log10_prev (true abs)

    preds = {
        "gbm": pred_gbm,
        "mlp": pred_mlp,
        "const_mean": pred_const,
        "kre": pred_kre,
        "cdp1": pred_cdp1,
    }

    for method, pred in preds.items():
        rmse_ratio = float(np.sqrt(np.mean((pred - y_te) ** 2)))
        pred_abs = log10_prev + pred
        rmse_abs = float(np.sqrt(np.mean((pred_abs - log10_curr) ** 2)))
        all_rows.append(
            {
                "dataset": name,
                "method": method,
                "rmse_ratio_log10": round(rmse_ratio, 4),
                "rmse_abs_log10": round(rmse_abs, 4),
                "holdout_offset": holdout_offset,
                "n_eval_rows": len(y_te),
            }
        )

    # ---- Canonical interpolation context (different protocol) ----
    canonical_rmse = {}
    if ds.get("by_offset_file"):
        try:
            by_off = pd.read_csv(ds["by_offset_file"])
            deep_row = by_off[by_off["offset"] == holdout_offset]
            for m in ["gbm", "mlp", "kre", "cdp1"]:
                sub = deep_row[deep_row["method"] == m]["rmse_log10"]
                if len(sub) > 0:
                    canonical_rmse[m] = float(sub.iloc[0])
        except Exception as e:
            print(f"  [context] could not load canonical by-offset: {e}")

    # ---- Print table ----
    result_df = pd.DataFrame([r for r in all_rows if r["dataset"] == name])
    print(f"\nRatio RMSE (log10 units) on held-out deepest pair:")
    print(result_df[["method", "rmse_ratio_log10", "rmse_abs_log10"]].to_string(index=False))

    if canonical_rmse:
        print(
            f"\n[CONTEXT -- DIFFERENT PROTOCOL: canonical models trained on all offsets]"
        )
        print(f"  Absolute log10 RMSE at offset {holdout_offset}:")
        for m, v in canonical_rmse.items():
            print(f"    {m}: {v:.4f}")

    # ---- Verdict ----
    def rmse(pred):
        return float(np.sqrt(np.mean((pred - y_te) ** 2)))

    best_learned = min(rmse(pred_gbm), rmse(pred_mlp))
    const_rmse = rmse(pred_const)
    kre_rmse = rmse(pred_kre)
    cdp1_rmse = rmse(pred_cdp1)

    beats_const = best_learned < const_rmse
    beats_kre = best_learned < kre_rmse
    beats_cdp1 = best_learned < cdp1_rmse

    verdict = (
        f"VERDICT [{name}]: "
        f"best_learned_ratio_rmse={best_learned:.4f} | "
        f"const={const_rmse:.4f} | kre={kre_rmse:.4f} | cdp1={cdp1_rmse:.4f} || "
        f"beats_const={beats_const} beats_kre={beats_kre} beats_cdp1={beats_cdp1}"
    )
    print(f"\n{verdict}")
    return verdict


def main():
    all_rows = []
    verdicts = []
    for ds in DATASETS:
        v = run_dataset(ds, all_rows)
        if v:
            verdicts.append(v)

    out_df = pd.DataFrame(all_rows)
    out_path = "results/extrapolation_test.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n\nWrote {out_path}")
    print("\n=== SUMMARY ===")
    for v in verdicts:
        print(v)


if __name__ == "__main__":
    main()
