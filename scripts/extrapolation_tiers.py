#!/usr/bin/env python3
"""Tiered extrapolation experiment (Phase 3e).

Three feature tiers:
  minimal: [h_{active}, "bound"]  (2 features)
  full:    13 PANCAKE_FEATURE_NAMES
  probe:   full + 5 active-heuristic probe features (excluding n_fertile)
           probe_{active}_{min_child, mean_child, max_child, n_improving, n_worsening}

Two directions:
  forward: train on shallow pairs (offsets < holdout_offset), eval on deepest pair.
           IDENTICAL double-holdout protocol to scripts/extrapolation_test.py.
  reverse: train on deep pairs (offsets >= 1), eval on (0,1) pair.

Baselines (tier-independent, computed once per direction/dataset):
  const:   mean training ratio (from full-tier training rows — same rows for all tiers)
  kre:     log10 KRE(bound_deep) - log10 KRE(bound_prev) per state
  cdp1:    log10 CDP_1(h_s, bound_deep) - log10 CDP_1(h_s, bound_prev) per state
  Samplers: random.Random(100) / random.Random(200), 200 k each.

Cluster-bootstrap SEs: 1 000 resamples over eval STATES, numpy default_rng(0).

Consistency gate: forward / full-tier / GBM RMSE must reproduce
  results/extrapolation_test.csv values (0.1079 / 0.0544 / 0.1211) to 4 dp.
  Script writes no output and exits with error if a mismatch is detected.

Output: results/extrapolation_tiers.csv
  Columns: direction, dataset, tier, method, rmse, se, n_eval
"""

import os
import sys
import random
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from effortpred.cdp import cdp1_predict
from effortpred.conditional import H_MAX_PANCAKE, sample_conditional_matrix
from effortpred.models import fit_gbm, fit_mlp, split_by_state
from effortpred.pancake import estimate_pancake_distribution, gap_h, rand_h
from effortpred.pancake_features import PANCAKE_FEATURE_NAMES
from effortpred.pancake_tree import kre_predict_pancake
from effortpred.probe_features import extract_probe_features

OUTPUT_PATH = "results/extrapolation_tiers.csv"

# Ground truth for consistency gate (from results/extrapolation_test.csv)
_EXPECTED_FWD_FULL_GBM = {
    "gap_n12": 0.1079,
    "gap_n10": 0.0544,
    "rand_n10": 0.1211,
}

DATASETS = [
    {
        "name": "gap_n12",
        "file": "results/pancake_labels_gap.csv",
        "h_col": "h_gap",
        "h_fn": gap_h,
        "n": 12,
        "holdout_offset": 3,
    },
    {
        "name": "gap_n10",
        "file": "results/pancake_labels_gap_n10.csv",
        "h_col": "h_gap",
        "h_fn": gap_h,
        "n": 10,
        "holdout_offset": 4,
    },
    {
        "name": "rand_n10",
        "file": "results/pancake_labels_rand_n10.csv",
        "h_col": "h_rand",
        "h_fn": rand_h,
        "n": 10,
        "holdout_offset": 3,
    },
]

# Probe suffixes that do NOT depend on bound (n_fertile is excluded)
_PROBE_SUFFIXES = ("min_child", "mean_child", "max_child", "n_improving", "n_worsening")


def _probe_cols(h_active: str) -> list:
    """5 bound-independent probe feature column names for *h_active* ('gap' or 'rand')."""
    return [f"probe_{h_active}_{s}" for s in _PROBE_SUFFIXES]


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _parse_state(s: str) -> tuple:
    return tuple(int(x) for x in s.split())


def _add_probe_features(df: pd.DataFrame, h_active: str) -> pd.DataFrame:
    """Add 5 bound-independent probe features to df (cached per unique state).

    Calls extract_probe_features with bound=0; the 5 selected keys are
    independent of bound (only n_fertile depends on it, and it is excluded).
    """
    cols = _probe_cols(h_active)
    cache: dict = {}
    rows = []
    for state_str in df["state"]:
        if state_str not in cache:
            feats = extract_probe_features(_parse_state(state_str), bound=0)
            cache[state_str] = [feats[c] for c in cols]
        rows.append(cache[state_str])
    probe_df = pd.DataFrame(rows, columns=cols, index=df.index)
    return pd.concat([df, probe_df], axis=1)


# ---------------------------------------------------------------------------
# Ratio row construction
# ---------------------------------------------------------------------------

def _build_ratio_rows(
    state_df: pd.DataFrame, pairs: list, feat_cols: list
) -> pd.DataFrame:
    """Merge consecutive-offset pairs into growth-ratio rows.

    Features are taken from the DEEPER (o_curr) row.
    Returns: state, ratio, log10_curr, log10_prev, *feat_cols.
    """
    parts = []
    for o_prev, o_curr in pairs:
        curr = state_df[state_df["offset"] == o_curr]
        prev = state_df[state_df["offset"] == o_prev]
        if curr.empty or prev.empty:
            continue
        avail = [c for c in feat_cols if c in curr.columns]
        merged = pd.merge(
            curr[["state", "nodes"] + avail].rename(columns={"nodes": "nodes_curr"}),
            prev[["state", "nodes"]].rename(columns={"nodes": "nodes_prev"}),
            on="state",
        )
        merged["log10_curr"] = np.log10(merged["nodes_curr"].astype(float))
        merged["log10_prev"] = np.log10(merged["nodes_prev"].astype(float))
        merged["ratio"] = merged["log10_curr"] - merged["log10_prev"]
        parts.append(merged)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Metrics and cluster-bootstrap SE
# ---------------------------------------------------------------------------

def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _cluster_bootstrap_se(
    y_true,
    y_pred,
    states,
    n_boot: int = 1000,
    rng=None,
) -> float:
    """Bootstrap RMSE SE by resampling STATES (cluster-level) with replacement.

    Uses a shared *rng* (numpy Generator) for full reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    states = np.asarray(states)
    unique = np.unique(states)
    n_st = len(unique)
    idx_map: dict = defaultdict(list)
    for i, s in enumerate(states):
        idx_map[s].append(i)
    idx_arr = {s: np.array(v) for s, v in idx_map.items()}
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.choice(unique, size=n_st, replace=True)
        idx = np.concatenate([idx_arr[s] for s in chosen])
        e = y_true[idx] - y_pred[idx]
        boots[b] = np.sqrt(np.mean(e ** 2))
    return float(np.std(boots, ddof=1))


# ---------------------------------------------------------------------------
# Analytic baselines
# ---------------------------------------------------------------------------

def _analytic_preds(
    eval_df: pd.DataFrame, h_col: str, n: int, dist, cond
):
    """KRE and CDP_1 growth-ratio predictions for all rows in eval_df.

    Works for both FORWARD and REVERSE directions:
      bound_deep = eval_df["bound"]  (feature row is the deeper offset)
      bound_prev = bound_deep - 1
    """
    h_vals = eval_df[h_col].values.astype(int)
    bound_deep = eval_df["bound"].values.astype(int)
    kre_cache: dict = {}
    pred_kre = np.empty(len(eval_df), dtype=float)
    pred_cdp1 = np.empty(len(eval_df), dtype=float)
    for i, (hs, bd) in enumerate(zip(h_vals, bound_deep)):
        bp = bd - 1
        for b in (bd, bp):
            if b not in kre_cache:
                kre_cache[b] = np.log10(max(kre_predict_pancake(n, b, dist), 1.0))
        pred_kre[i] = kre_cache[bd] - kre_cache[bp]
        cdd = np.log10(max(cdp1_predict(hs, bd, cond, n - 1, n - 2), 1.0))
        cdp = np.log10(max(cdp1_predict(hs, bp, cond, n - 1, n - 2), 1.0))
        pred_cdp1[i] = cdd - cdp
    return pred_kre, pred_cdp1


# ---------------------------------------------------------------------------
# Core: run one direction for one dataset
# ---------------------------------------------------------------------------

def _run_direction(
    direction: str,
    ds: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dist,
    cond,
    tiers: dict,
    all_feat_cols: list,
    boot_rng,
) -> list:
    name = ds["name"]
    h_col = ds["h_col"]
    n = ds["n"]
    holdout_offset = ds["holdout_offset"]

    offsets_all = sorted(
        set(train_df["offset"]) | set(val_df["offset"]) | set(test_df["offset"])
    )

    if direction == "forward":
        # Replicate extrapolation_test.py exactly
        shallow = [o for o in offsets_all if o < holdout_offset]
        train_pairs = [(shallow[i], shallow[i + 1]) for i in range(len(shallow) - 1)]
        eval_pairs = [(holdout_offset - 1, holdout_offset)]
    else:  # reverse
        all_consec = [
            (offsets_all[i], offsets_all[i + 1])
            for i in range(len(offsets_all) - 1)
        ]
        eval_pairs = [(0, 1)]
        train_pairs = [p for p in all_consec if p != (0, 1)]

    print(
        f"  [{direction}] train_pairs={train_pairs}  eval_pair={eval_pairs[0]}"
    )

    tr_rows = _build_ratio_rows(train_df, train_pairs, all_feat_cols)
    va_rows = _build_ratio_rows(val_df, train_pairs, all_feat_cols)
    te_rows = _build_ratio_rows(test_df, eval_pairs, all_feat_cols)

    print(
        f"    ratio rows: train={len(tr_rows)}, val={len(va_rows)}, eval={len(te_rows)}"
    )

    if te_rows.empty:
        print(f"    WARNING: no eval rows — skipping {name}/{direction}")
        return []

    y_tr = tr_rows["ratio"].values
    y_te = te_rows["ratio"].values
    te_states = te_rows["state"].values

    results = []

    # ---- Baselines (tier-independent) ----
    pred_const = np.full(len(y_te), float(np.mean(y_tr)))
    pred_kre, pred_cdp1 = _analytic_preds(te_rows, h_col, n, dist, cond)

    for m_name, pred in [
        ("const", pred_const),
        ("kre", pred_kre),
        ("cdp1", pred_cdp1),
    ]:
        r = _rmse(y_te, pred)
        se = _cluster_bootstrap_se(y_te, pred, te_states, rng=boot_rng)
        results.append(
            {
                "direction": direction,
                "dataset": name,
                "tier": "baseline",
                "method": m_name,
                "rmse": round(r, 4),
                "se": round(se, 4),
                "n_eval": len(y_te),
            }
        )
        print(f"    baseline/{m_name}: {r:.4f} ± {se:.4f}")

    # ---- Per-tier learned models ----
    va_empty = va_rows.empty

    for tier_name, tier_cols in tiers.items():
        avail = [c for c in tier_cols if c in tr_rows.columns]
        X_tr = tr_rows[avail].values.astype(float)
        X_te = te_rows[avail].values.astype(float)

        if va_empty:
            # Fallback (should not occur in practice): use tail of train as val
            cut = max(1, len(X_tr) - int(0.15 * len(X_tr)))
            X_va_fit = X_tr[cut:]
            y_va_fit = y_tr[cut:]
            X_tr_fit = X_tr[:cut]
            y_tr_fit = y_tr[:cut]
        else:
            X_va_fit = va_rows[avail].values.astype(float)
            y_va_fit = va_rows["ratio"].values
            X_tr_fit = X_tr
            y_tr_fit = y_tr

        print(f"    Fitting GBM  [{tier_name}, {len(avail)} feats]...")
        gbm = fit_gbm(X_tr_fit, y_tr_fit, seed=0)
        pred_gbm = gbm.predict(X_te)

        print(f"    Fitting MLP  [{tier_name}]...")
        mlp_fn = fit_mlp(X_tr_fit, y_tr_fit, X_va_fit, y_va_fit, seed=0)
        pred_mlp = mlp_fn(X_te)

        for m_name, pred in [("gbm", pred_gbm), ("mlp", pred_mlp)]:
            r = _rmse(y_te, pred)
            se = _cluster_bootstrap_se(y_te, pred, te_states, rng=boot_rng)
            results.append(
                {
                    "direction": direction,
                    "dataset": name,
                    "tier": tier_name,
                    "method": m_name,
                    "rmse": round(r, 4),
                    "se": round(se, 4),
                    "n_eval": len(y_te),
                }
            )
            print(f"    {tier_name}/{m_name}: {r:.4f} ± {se:.4f}")

    return results


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------

def _run_dataset(ds: dict, boot_rng) -> list:
    name = ds["name"]
    h_col = ds["h_col"]
    h_fn = ds["h_fn"]
    n = ds["n"]
    holdout_offset = ds["holdout_offset"]

    print(f"\n{'=' * 62}")
    print(f"Dataset: {name}  (n={n}, holdout_offset={holdout_offset})")
    print(f"{'=' * 62}")

    df = pd.read_csv(ds["file"])
    df["offset"] = df["bound"] - df[h_col]

    n_cens = int(df["censored"].sum())
    df = df[df["censored"] == 0].copy()
    print(f"Dropped {n_cens} censored rows; {len(df)} rows remain")

    # Active heuristic: "gap" from "h_gap", "rand" from "h_rand"
    h_active = h_col[2:]

    print("Computing probe features (cached per unique state)...")
    df = _add_probe_features(df, h_active)

    train_df, val_df, test_df = split_by_state(df, seed=0)
    print(
        f"States: train={train_df['state'].nunique()}, "
        f"val={val_df['state'].nunique()}, "
        f"test={test_df['state'].nunique()}"
    )

    probe_feature_cols = _probe_cols(h_active)
    all_feat_cols = list(PANCAKE_FEATURE_NAMES) + probe_feature_cols

    tiers = {
        "minimal": [h_col, "bound"],
        "full": list(PANCAKE_FEATURE_NAMES),
        "probe": list(PANCAKE_FEATURE_NAMES) + probe_feature_cols,
    }

    print(f"\nSampling KRE distribution (n={n}, 200 k)...")
    dist = estimate_pancake_distribution(n, h_fn, 200_000, random.Random(100))
    print(f"Sampling CDP_1 conditional matrix (n={n}, 200 k)...")
    cond = sample_conditional_matrix(
        n, h_fn, 200_000, random.Random(200), h_max=H_MAX_PANCAKE(n)
    )

    all_rows = []
    for direction in ("forward", "reverse"):
        rows = _run_direction(
            direction, ds,
            train_df, val_df, test_df,
            dist, cond,
            tiers, all_feat_cols,
            boot_rng,
        )
        all_rows.extend(rows)

    return all_rows


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_table(df_res: pd.DataFrame, direction: str, dataset: str) -> None:
    sub = df_res[
        (df_res["direction"] == direction) & (df_res["dataset"] == dataset)
    ]
    if sub.empty:
        return
    print(f"\n--- {direction.upper()} / {dataset} ---")
    header = f"  {'tier':<10} {'method':<8} {'rmse':>8} {'se':>8} {'n_eval':>7}"
    print(header)
    print(f"  {'-' * 46}")
    for tier in ("baseline", "minimal", "full", "probe"):
        t_sub = sub[sub["tier"] == tier].sort_values("method")
        for _, row in t_sub.iterrows():
            print(
                f"  {row['tier']:<10} {row['method']:<8} "
                f"{row['rmse']:>8.4f} {row['se']:>8.4f} {row['n_eval']:>7}"
            )


def _compute_verdicts(df_res: pd.DataFrame) -> list:
    """Print and return the four required verdict lines."""

    def get(direction, dataset, tier, method):
        sub = df_res[
            (df_res["direction"] == direction)
            & (df_res["dataset"] == dataset)
            & (df_res["tier"] == tier)
            & (df_res["method"] == method)
        ]
        if sub.empty:
            return None, None
        return float(sub["rmse"].iloc[0]), float(sub["se"].iloc[0])

    lines = []
    print("\n=== VERDICTS ===")

    # (a) Does probe beat full on consistent gap datasets (forward)?
    print("\n(a) probe vs full — consistent-heuristic FORWARD gap datasets:")
    for ds_name in ("gap_n12", "gap_n10"):
        p_gbm, _ = get("forward", ds_name, "probe", "gbm")
        p_mlp, _ = get("forward", ds_name, "probe", "mlp")
        f_gbm, _ = get("forward", ds_name, "full", "gbm")
        f_mlp, _ = get("forward", ds_name, "full", "mlp")
        if any(x is None for x in (p_gbm, p_mlp, f_gbm, f_mlp)):
            continue
        probe_best = min(p_gbm, p_mlp)
        full_best = min(f_gbm, f_mlp)
        beats = probe_best < full_best
        line = (
            f"[{ds_name}] probe_best={probe_best:.4f} "
            f"full_best={full_best:.4f} "
            f"probe_beats_full={beats}"
        )
        print(f"  {line}")
        lines.append(f"(a) {line}")

    # (b) Does probe close/beat the gap to CDP_1?
    print("\n(b) probe vs CDP_1 gap — consistent-heuristic FORWARD gap datasets:")
    for ds_name in ("gap_n12", "gap_n10"):
        p_gbm, p_gbm_se = get("forward", ds_name, "probe", "gbm")
        p_mlp, p_mlp_se = get("forward", ds_name, "probe", "mlp")
        cdp1_r, cdp1_se = get("forward", ds_name, "baseline", "cdp1")
        f_gbm, _ = get("forward", ds_name, "full", "gbm")
        if any(x is None for x in (p_gbm, p_mlp, cdp1_r)):
            continue
        if p_gbm <= p_mlp:
            probe_best, probe_se = p_gbm, p_gbm_se
        else:
            probe_best, probe_se = p_mlp, p_mlp_se
        gap = probe_best - cdp1_r
        full_gap = (f_gbm - cdp1_r) if f_gbm is not None else float("nan")
        # "significantly above" = gap > 2*max(se_probe, se_cdp1)
        sig_above = gap > 2 * max(probe_se or 0.0, cdp1_se or 0.0)
        line = (
            f"[{ds_name}] probe_best={probe_best:.4f}±{probe_se:.4f} "
            f"cdp1={cdp1_r:.4f}±{cdp1_se:.4f} "
            f"probe_minus_cdp1={gap:+.4f} "
            f"full_minus_cdp1={full_gap:+.4f} "
            f"probe_significantly_above_cdp1={sig_above}"
        )
        print(f"  {line}")
        lines.append(f"(b) {line}")

    # (c) Does minimal extrapolate at all (vs constant baseline, forward)?
    print("\n(c) minimal vs constant baseline — FORWARD all datasets:")
    for ds_name in ("gap_n12", "gap_n10", "rand_n10"):
        m_gbm, _ = get("forward", ds_name, "minimal", "gbm")
        m_mlp, _ = get("forward", ds_name, "minimal", "mlp")
        const_r, _ = get("forward", ds_name, "baseline", "const")
        if any(x is None for x in (m_gbm, m_mlp, const_r)):
            continue
        min_best = min(m_gbm, m_mlp)
        beats = min_best < const_r
        line = (
            f"[{ds_name}] minimal_best={min_best:.4f} "
            f"const={const_r:.4f} "
            f"minimal_beats_const={beats}"
        )
        print(f"  {line}")
        lines.append(f"(c) {line}")

    # (d) Any reversal of prior conclusions (forward full-tier GBM vs baselines)?
    print("\n(d) Prior conclusion check — FORWARD full-tier GBM vs all baselines:")
    for ds_name in ("gap_n12", "gap_n10", "rand_n10"):
        f_gbm, _ = get("forward", ds_name, "full", "gbm")
        const_r, _ = get("forward", ds_name, "baseline", "const")
        kre_r, _ = get("forward", ds_name, "baseline", "kre")
        cdp1_r, _ = get("forward", ds_name, "baseline", "cdp1")
        if any(x is None for x in (f_gbm, const_r, kre_r, cdp1_r)):
            continue
        bc = f_gbm < const_r
        bk = f_gbm < kre_r
        bd = f_gbm < cdp1_r
        line = (
            f"[{ds_name}] full_gbm={f_gbm:.4f} "
            f"const={const_r:.4f} kre={kre_r:.4f} cdp1={cdp1_r:.4f} "
            f"beats_const={bc} beats_kre={bk} beats_cdp1={bd}"
        )
        print(f"  {line}")
        lines.append(f"(d) {line}")

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if os.path.exists(OUTPUT_PATH):
        sys.exit(
            f"ERROR: {OUTPUT_PATH} already exists. "
            "Delete it manually before re-running to avoid overwriting results."
        )

    boot_rng = np.random.default_rng(0)
    all_rows: list = []

    for ds in DATASETS:
        rows = _run_dataset(ds, boot_rng)
        all_rows.extend(rows)

    df_res = pd.DataFrame(all_rows)

    # ---- Consistency gate ----
    print("\n=== CONSISTENCY CHECK (forward / full-tier / GBM) ===")
    check_ok = True
    for ds_name, expected in _EXPECTED_FWD_FULL_GBM.items():
        sub = df_res[
            (df_res["direction"] == "forward")
            & (df_res["dataset"] == ds_name)
            & (df_res["tier"] == "full")
            & (df_res["method"] == "gbm")
        ]
        if sub.empty:
            print(f"  {ds_name}: MISSING — cannot verify")
            check_ok = False
            continue
        actual = float(sub["rmse"].iloc[0])
        ok = abs(actual - expected) < 5e-5  # match to 4 dp
        status = "OK" if ok else f"MISMATCH (expected {expected:.4f}, got {actual:.4f})"
        print(f"  {ds_name}: {status}")
        if not ok:
            check_ok = False

    if not check_ok:
        sys.exit(
            "\nERROR: Consistency check FAILED — protocol divergence from "
            "scripts/extrapolation_test.py detected. No output written."
        )

    print("Consistency check PASSED.\n")

    # ---- Write CSV ----
    df_res.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}  ({len(df_res)} rows)")

    # ---- Print tables ----
    print("\n\n=== RESULTS TABLES ===")
    for direction in ("forward", "reverse"):
        for ds in DATASETS:
            _print_table(df_res, direction, ds["name"])

    # ---- Verdicts ----
    verdict_lines = _compute_verdicts(df_res)

    print("\n=== SUMMARY ===")
    for vl in verdict_lines:
        print(f"  {vl}")


if __name__ == "__main__":
    main()
