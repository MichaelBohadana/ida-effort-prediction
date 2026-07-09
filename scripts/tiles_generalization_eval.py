"""Generalization study: does the learned effort predictor handle the 15-puzzle's
non-uniform branching?

Blank-position determines branching: corner/edge/interior have 2/3/4 raw moves
(the blank_degree feature records exactly this 2/3/4 value); after parent pruning,
non-root nodes have 1/2/3 children (the root has no parent, so 2/3/4).

Tiers evaluated:
  minimal         : [h_manhattan, bound]
  minimal+blank   : [h_manhattan, bound, blank_row, blank_col, blank_degree]
  full            : all 12 FEATURE_NAMES
  full-minus-blank: 12 features minus blank_row/blank_col/blank_degree

Models: GBM (HistGradientBoosting) + MLP, both seed 0.
Analytic: KRE (kre_predict, cached by (blank_cell, bound)).

HONESTY RULE: no required direction — deltas reported exactly.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# Guard: must run from the project root so relative results/ paths work.
# (Explicit early check for protected files.)
PROTECTED = [
    "results/labels.csv",
    "results/pancake_labels_gap.csv",
    "results/pancake_labels_rand.csv",
    "results/eval_table.csv",
]


def _guard_no_overwrite():
    """Abort if this script would overwrite protected existing files."""
    outputs = [
        "results/tiles_generalization_eval.csv",
        "results/tiles_feature_importance.png",
    ]
    # Check that none of our output paths equal a protected path
    for out_path in outputs:
        for protected_path in PROTECTED:
            if out_path == protected_path:
                raise ValueError(
                    f"Output path {out_path!r} collides with protected file "
                    f"{protected_path!r}"
                )


def _blank_class(blank_cell):
    """Map a 15-puzzle blank cell index to 'corner', 'edge', or 'interior'."""
    corners = {0, 3, 12, 15}
    interior = {5, 6, 9, 10}
    if blank_cell in corners:
        return "corner"
    elif blank_cell in interior:
        return "interior"
    else:
        return "edge"


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Generalization eval on 15-puzzle uniform dataset.")
    ap.add_argument("--data", default="results/tiles_uniform_labels.csv")
    ap.add_argument("--dist", default="results/hdist_overall.npy")
    ap.add_argument("--out", default="results/tiles_generalization_eval.csv")
    ap.add_argument("--fig", default="results/tiles_feature_importance.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Guard: refuse to touch protected files
    _guard_no_overwrite()
    for p in PROTECTED:
        if args.out == p or args.fig == p:
            sys.exit(f"ERROR: output path {args.out!r} collides with protected file")

    if not os.path.exists(args.data):
        sys.exit(f"ERROR: data file not found: {args.data}\n"
                 "Run generate_tiles_uniform.py first.")
    if not os.path.exists(args.dist):
        sys.exit(f"ERROR: distribution file not found: {args.dist}")

    # ── Load & clean ──────────────────────────────────────────────────────────
    from effortpred.distribution import HDistribution
    from effortpred.features import FEATURE_NAMES
    from effortpred.kre import kre_predict
    from effortpred.metrics import eval_log10_predictions
    from effortpred.models import fit_gbm, fit_mlp, split_by_state

    df = pd.read_csv(args.data).sort_values(["state", "bound"]).reset_index(drop=True)
    n_total = len(df)
    n_censored = int(df["censored"].sum())
    df = df[df["censored"] == 0].copy()
    print(f"Loaded {n_total} rows; dropped {n_censored} censored → {len(df)} clean rows")

    df["y"] = np.log10(df["nodes"].astype(float))

    # Blank-position class for start state
    df["blank_cell"] = df["blank_row"] * 4 + df["blank_col"]
    df["blank_class"] = df["blank_cell"].apply(_blank_class)

    n_unique = df["state"].nunique()
    print(f"Unique states: {n_unique}")
    print("Blank-class distribution (states ≈ rows/n_offsets):")
    print(df.groupby("blank_class")["state"].nunique().to_string())

    # ── Split ─────────────────────────────────────────────────────────────────
    train, val, test = split_by_state(df, seed=args.seed)
    print(f"\nSplit: train={len(train)} val={len(val)} test={len(test)} rows")
    print("Test blank-class distribution:")
    print(test.groupby("blank_class")["state"].nunique().to_string())

    y_tr = train["y"].values
    y_va = val["y"].values
    y_te = test["y"].values

    # ── Tier definitions ──────────────────────────────────────────────────────
    BLANK_FEATS = ["blank_row", "blank_col", "blank_degree"]
    FULL_MINUS_BLANK = [f for f in FEATURE_NAMES if f not in BLANK_FEATS]

    tiers = {
        "minimal":          ["h_manhattan", "bound"],
        "minimal+blank":    ["h_manhattan", "bound"] + BLANK_FEATS,
        "full":             list(FEATURE_NAMES),
        "full-minus-blank": FULL_MINUS_BLANK,
    }

    # ── Mean log10 nodes per blank class (shows classes genuinely differ) ─────
    print("\n=== MEAN log10(nodes) BY BLANK CLASS (test set) ===")
    for bc in ["corner", "edge", "interior"]:
        mask = test["blank_class"] == bc
        if mask.sum() > 0:
            print(f"  {bc:10s}: mean={y_te[mask].mean():.3f}  "
                  f"std={y_te[mask].std():.3f}  n_rows={mask.sum()}")

    # ── Train & predict per tier ───────────────────────────────────────────────
    all_preds = {}  # (tier, method) -> predictions on test set
    models_for_importance = {}  # tier -> gbm model (for feature importance plot)

    for tier_name, feat_names in tiers.items():
        X_tr = train[feat_names].values.astype(float)
        X_va = val[feat_names].values.astype(float)
        X_te = test[feat_names].values.astype(float)

        gbm = fit_gbm(X_tr, y_tr, seed=args.seed)
        all_preds[(tier_name, "gbm")] = gbm.predict(X_te)
        if tier_name == "full":
            models_for_importance["full"] = (gbm, feat_names, X_te)

        mlp_fn = fit_mlp(X_tr, y_tr, X_va, y_va, seed=args.seed)
        all_preds[(tier_name, "mlp")] = mlp_fn(X_te)

        print(f"  trained tier={tier_name} ({len(feat_names)} features)")

    # ── KRE predictions ───────────────────────────────────────────────────────
    print("\nComputing KRE predictions (cached by blank_cell × bound)...")
    dist = HDistribution(np.load(args.dist))
    kre_cache = {}  # (blank_cell, bound) -> float

    kre_log = []
    for _, row in test.iterrows():
        state = tuple(int(x) for x in str(row["state"]).split())
        bound = int(row["bound"])
        blank_cell = state.index(0)
        key = (blank_cell, bound)
        if key not in kre_cache:
            kre_cache[key] = kre_predict(state, 4, bound, dist)
        kre_log.append(np.log10(max(kre_cache[key], 1.0)))

    all_preds[("kre", "kre")] = np.array(kre_log)
    print(f"  KRE cache size: {len(kre_cache)} (blank_cell, bound) pairs")

    # ── Metrics: overall + per blank class ────────────────────────────────────
    result_rows = []
    classes = ["all", "corner", "edge", "interior"]

    print("\n=== RESULTS TABLE (rmse_log10) ===")
    header = f"{'tier':<20} {'method':<6} " + " ".join(f"{'  '+c:>12}" for c in classes)
    print(header)
    print("-" * len(header))

    for (tier_name, method_name), preds in all_preds.items():
        row_data = {"tier": tier_name, "method": method_name}
        rmses = {}

        # Overall
        m = eval_log10_predictions(y_te, preds)
        rmses["all"] = m["rmse_log10"]
        row_data["rmse_all"] = m["rmse_log10"]

        for bc in ["corner", "edge", "interior"]:
            mask = (test["blank_class"] == bc).values
            if mask.sum() < 5:
                rmses[bc] = float("nan")
                row_data[f"rmse_{bc}"] = float("nan")
            else:
                m_bc = eval_log10_predictions(y_te[mask], np.asarray(preds)[mask])
                rmses[bc] = m_bc["rmse_log10"]
                row_data[f"rmse_{bc}"] = m_bc["rmse_log10"]

        result_rows.append(row_data)

        line = (f"{tier_name:<20} {method_name:<6} " +
                " ".join(f"{rmses[c]:>12.4f}" for c in classes))
        print(line)

    print()

    # ── Save results CSV ───────────────────────────────────────────────────────
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(args.out, index=False)
    print(f"Saved per-tier results to {args.out}")

    # ── Verdict block ──────────────────────────────────────────────────────────
    def _get_rmse(tier, method, cls="all"):
        col = f"rmse_{cls}"
        row = result_df[(result_df["tier"] == tier) &
                        (result_df["method"] == method)]
        if row.empty:
            return float("nan")
        return float(row.iloc[0][col])

    print("\n" + "=" * 60)
    print("VERDICT BLOCK")
    print("=" * 60)

    for method in ["gbm", "mlp"]:
        print(f"\n--- {method.upper()} ---")

        # (a) Does adding blank features improve over minimal?
        rmse_minimal = _get_rmse("minimal", method)
        rmse_minimal_blank = _get_rmse("minimal+blank", method)
        delta_a = rmse_minimal_blank - rmse_minimal
        direction_a = "IMPROVES" if delta_a < -0.005 else ("HURTS" if delta_a > 0.005 else "NEGLIGIBLE")
        print(f"(a) blank features over minimal:")
        print(f"    minimal={rmse_minimal:.4f}  minimal+blank={rmse_minimal_blank:.4f}  "
              f"delta={delta_a:+.4f}  [{direction_a}]")

        # (b) Does removing blank features from full hurt?
        rmse_full = _get_rmse("full", method)
        rmse_full_minus = _get_rmse("full-minus-blank", method)
        delta_b = rmse_full_minus - rmse_full
        direction_b = "HURTS REMOVAL" if delta_b > 0.005 else ("HURTS FULL" if delta_b < -0.005 else "NEGLIGIBLE")
        print(f"(b) removing blank from full:")
        print(f"    full={rmse_full:.4f}  full-minus-blank={rmse_full_minus:.4f}  "
              f"delta={delta_b:+.4f}  [{direction_b}]")

    # (c) Per-class error patterns: best learned vs KRE
    best_tier = "full"
    print(f"\n(c) Per-class error: {best_tier}/GBM vs KRE")
    print(f"    {'class':10s}  {'full/GBM':>10}  {'KRE':>10}  {'KRE-GBM':>10}")
    for bc in ["corner", "edge", "interior"]:
        rmse_g = _get_rmse(best_tier, "gbm", bc)
        rmse_k = _get_rmse("kre", "kre", bc)
        diff = rmse_k - rmse_g
        print(f"    {bc:10s}  {rmse_g:>10.4f}  {rmse_k:>10.4f}  {diff:>+10.4f}")

    print("\n" + "=" * 60)

    # ── Feature importance plot (GBM on full tier) ────────────────────────────
    if "full" in models_for_importance:
        gbm_full, feat_names_full, X_te_full = models_for_importance["full"]
        print(f"\nComputing permutation importance for 'full' GBM...")
        imp = permutation_importance(
            gbm_full, X_te_full, y_te,
            n_repeats=20, random_state=args.seed)
        order = np.argsort(imp.importances_mean)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(
            [feat_names_full[i] for i in order],
            imp.importances_mean[order],
            xerr=imp.importances_std[order],
            color="steelblue", alpha=0.8,
        )
        ax.set_xlabel("Permutation importance (decrease in R²)")
        ax.set_title("Feature importance — 15-puzzle (GBM, full tier, test set)")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        fig.tight_layout()
        fig.savefig(args.fig, dpi=150)
        plt.close(fig)
        print(f"Saved feature importance plot to {args.fig}")

        print("\nTop features (permutation importance, full tier/GBM):")
        for i in reversed(order):
            print(f"  {feat_names_full[i]:25s}: {imp.importances_mean[i]:.4f} "
                  f"± {imp.importances_std[i]:.4f}")

    print(f"\nDone. Output files:")
    print(f"  {args.out}")
    print(f"  {args.fig}")


if __name__ == "__main__":
    main()
