"""Learning-curve (sample-efficiency) experiment: how many training states
are needed before the learned models reach near-full accuracy?

Uses a FIXED canonical split (seed=0) so val/test sets are identical to the
headline eval, making results directly comparable. For each training-state
count S and repetition r, S states are randomly sampled from the train-split
states, all their rows are used for training, and the model is evaluated on
the fixed test set.

Reference lines (canonical seed-0 full-dataset values):
    gap:  CDP_1=0.308  KRE=1.172
    rand: CDP_1=0.834  KRE=1.779

Usage:
    python scripts/learning_curves.py [--heuristics gap rand]
                                      [--sizes 50 100 200 400 800 1050]
                                      [--reps 3]
                                      [--feature-sets minimal full]
                                      [--seed 0]
                                      [--out-prefix results/learning_curve]
"""

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from effortpred.metrics import eval_log10_predictions
from effortpred.models import fit_gbm, fit_mlp, split_by_state
from effortpred.pancake_features import PANCAKE_FEATURE_NAMES

# Canonical analytic reference values from the headline eval (seed=0, full data).
CANONICAL_REFS = {
    "gap":  {"cdp1": 0.308, "kre": 1.172},
    "rand": {"cdp1": 0.834, "kre": 1.779},
}


def run_heuristic_featureset(heuristic_name, feature_set, args, df,
                              train, val, test):
    """Run all (size, rep) experiments for one (heuristic, feature_set) combo."""
    h_col = f"h_{heuristic_name}"
    feat_names = (PANCAKE_FEATURE_NAMES if feature_set == "full"
                  else [h_col, "bound"])

    # Fixed val/test tensors (same for every size/rep in this combo).
    X_va = val[feat_names].values.astype(float)
    X_te = test[feat_names].values.astype(float)
    y_va = val["y"].values
    y_te = test["y"].values

    # All unique states in the train split.
    train_states = np.array(sorted(train["state"].unique()))
    n_train_states = len(train_states)

    results = []
    for size in args.sizes:
        # Cap at the actual number of available training states.
        effective_size = min(size, n_train_states)
        for rep in range(args.reps):
            rng = random.Random(1000 + rep)
            sampled = list(train_states)
            rng.shuffle(sampled)
            chosen_states = set(sampled[:effective_size])
            sub_train = train[train["state"].isin(chosen_states)]

            X_tr = sub_train[feat_names].values.astype(float)
            y_tr = sub_train["y"].values

            # GBM
            gbm = fit_gbm(X_tr, y_tr, seed=rep)
            pred_gbm = gbm.predict(X_te)
            rmse_gbm = eval_log10_predictions(y_te, pred_gbm)["rmse_log10"]

            # MLP — val set is always the full canonical val split
            mlp_fn = fit_mlp(X_tr, y_tr, X_va, y_va, seed=rep)
            pred_mlp = mlp_fn(X_te)
            rmse_mlp = eval_log10_predictions(y_te, pred_mlp)["rmse_log10"]

            results.append({
                "heuristic":   heuristic_name,
                "feature_set": feature_set,
                "size":        effective_size,   # actual states used
                "rep":         rep,
                "method":      "gbm",
                "rmse_log10":  rmse_gbm,
            })
            results.append({
                "heuristic":   heuristic_name,
                "feature_set": feature_set,
                "size":        effective_size,
                "rep":         rep,
                "method":      "mlp",
                "rmse_log10":  rmse_mlp,
            })
            print(f"  [{heuristic_name}/{feature_set}] size={effective_size:4d} "
                  f"rep={rep}  gbm={rmse_gbm:.4f}  mlp={rmse_mlp:.4f}")

    return results


def make_plot(df_sub, heuristic_name, feature_set, out_prefix):
    """Mean ± std over reps for GBM and MLP; dashed CDP_1/KRE reference lines."""
    refs = CANONICAL_REFS[heuristic_name]
    pivot = df_sub.groupby(["size", "method"])["rmse_log10"].agg(
        mean="mean", std="std").reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"gbm": "#1f77b4", "mlp": "#ff7f0e"}
    for method in ["gbm", "mlp"]:
        sub = pivot[pivot["method"] == method].sort_values("size")
        ax.semilogx(sub["size"], sub["mean"], "o-", color=colors[method],
                    label=method.upper())
        ax.fill_between(sub["size"],
                        sub["mean"] - sub["std"],
                        sub["mean"] + sub["std"],
                        alpha=0.2, color=colors[method])

    # Dashed reference lines for analytic methods.
    ax.axhline(refs["cdp1"], color="green",  linestyle="--", linewidth=1.2,
               label=f"CDP_1 ({refs['cdp1']:.3f})")
    ax.axhline(refs["kre"],  color="red",    linestyle="--", linewidth=1.2,
               label=f"KRE ({refs['kre']:.3f})")

    ax.set_xlabel("Training states (log scale)")
    ax.set_ylabel("Test RMSE log10")
    ax.set_title(f"Learning curve — {heuristic_name} / {feature_set} features")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()

    png_path = f"{out_prefix}_{heuristic_name}_{feature_set}.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Saved {png_path}")
    return png_path


def threshold_sentence(df_sub, heuristic_name, feature_set, full_size):
    """Find the smallest size within 0.05 rmse of the full-size result per model."""
    refs_full = {}
    for method in ["gbm", "mlp"]:
        sub = df_sub[(df_sub["method"] == method) & (df_sub["size"] == full_size)]
        if len(sub) == 0:
            # Use the largest available size
            max_size = df_sub[df_sub["method"] == method]["size"].max()
            sub = df_sub[(df_sub["method"] == method) & (df_sub["size"] == max_size)]
        refs_full[method] = sub["rmse_log10"].mean()

    sentences = []
    for method in ["gbm", "mlp"]:
        target = refs_full[method] + 0.05
        sub = df_sub[df_sub["method"] == method].groupby("size")["rmse_log10"].mean()
        sub = sub.sort_index()
        within = sub[sub <= target]
        if len(within) == 0:
            sentences.append(f"{method.upper()}: no size reached within 0.05 of full-data result ({refs_full[method]:.4f})")
        else:
            smallest = within.index[0]
            sentences.append(
                f"{method.upper()}: smallest size within 0.05 rmse of full ({refs_full[method]:.4f}) "
                f"= {smallest} training states (mean rmse={within[smallest]:.4f})"
            )
    return sentences


def main():
    ap = argparse.ArgumentParser(
        description="Learning curves for pancake effort prediction")
    ap.add_argument("--heuristics", nargs="+", default=["gap", "rand"],
                    choices=["gap", "rand"])
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[50, 100, 200, 400, 800, 1050])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--feature-sets", nargs="+", default=["minimal", "full"],
                    choices=["minimal", "full"])
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the canonical split (keep at 0 for comparability)")
    ap.add_argument("--out-prefix", default="results/learning_curve")
    args = ap.parse_args()

    # Safety: ensure we never write to protected files.
    protected_prefixes = [
        "results/pancake_eval_",
        "results/pancake_labels_",
    ]
    for p in protected_prefixes:
        if args.out_prefix.startswith(p):
            raise ValueError(f"--out-prefix {args.out_prefix!r} would shadow protected files.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix + "_x.csv")),
                exist_ok=True)

    all_results = []

    for heuristic_name in args.heuristics:
        print(f"\n{'='*60}")
        print(f"  Heuristic: {heuristic_name.upper()}")
        print(f"{'='*60}")

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
        df = df.sort_values(["state", "bound"]).reset_index(drop=True)

        # FIXED canonical split — identical to headline eval.
        train, val, test = split_by_state(df, seed=args.seed)
        n_train_states = train["state"].nunique()
        full_size = n_train_states
        print(f"  Train states: {n_train_states}  Val states: {val['state'].nunique()}"
              f"  Test states: {test['state'].nunique()}")

        for feature_set in args.feature_sets:
            print(f"\n  Feature set: {feature_set}")
            rows = run_heuristic_featureset(
                heuristic_name, feature_set, args, df, train, val, test)
            all_results.extend(rows)

    df_all = pd.DataFrame(all_results)

    # Write CSV.
    csv_out = args.out_prefix + ".csv"
    df_all.to_csv(csv_out, index=False)
    print(f"\n  Wrote {csv_out} ({len(df_all)} rows)")

    # Plots + pivots + threshold sentences.
    for heuristic_name in args.heuristics:
        df_h = df_all[df_all["heuristic"] == heuristic_name]
        # Determine full_size for this heuristic.
        csv_path = os.path.join(
            os.path.dirname(__file__), "..", "results",
            f"pancake_labels_{heuristic_name}.csv"
        )
        tmp_df = pd.read_csv(csv_path)
        tmp_df = tmp_df[tmp_df["censored"] == 0].copy()
        tmp_df["y"] = np.log10(tmp_df["nodes"].astype(float))
        train_tmp, _, _ = split_by_state(tmp_df, seed=args.seed)
        full_size = train_tmp["state"].nunique()

        for feature_set in args.feature_sets:
            df_sub = df_h[df_h["feature_set"] == feature_set]

            # Pivot table: mean rmse per (size, method).
            pivot = df_sub.groupby(["size", "method"])["rmse_log10"].mean().unstack()
            print(f"\n  Pivot [{heuristic_name}/{feature_set}]:")
            print(pivot.round(4).to_string())

            # Threshold sentences.
            sents = threshold_sentence(df_sub, heuristic_name, feature_set, full_size)
            for s in sents:
                print(f"  >> {s}")

            # Plot.
            make_plot(df_sub, heuristic_name, feature_set, args.out_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
