"""Generate the five Experimental-Results figures (F1--F5) for the report.

AAAI-style: serif typography, Okabe--Ito colorblind-safe palette, embedded
fonts (Type 42), vector PDF sized to \\columnwidth (3.3in) or \\textwidth (7in).
Every plotted value is read from a frozen results/*.csv; F1 reproduces the
frozen test-set predictions and gates on matching the frozen aggregate RMSE.

Run from the method repo root:  python scripts/make_report_figures.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COL, FULL = 3.3, 7.0
FIGDIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                      "..", "..", "method3_report", "figures"))
# Okabe--Ito colorblind-safe assignments, consistent across every figure.
C = {"gbm": "#0072B2", "mlp": "#009E73", "cdp1": "#E69F00", "kre": "#D55E00",
     "mean_baseline": "#999999", "slack_baseline": "#666666",
     "full": "#0072B2", "minimal": "#D55E00"}
LB = {"gbm": "GBM", "mlp": "MLP", "cdp1": r"CDP$_1$", "kre": "KRE",
      "mean_baseline": "mean", "slack_baseline": "slack"}
MK = {"gbm": "o", "mlp": "s", "cdp1": "^", "kre": "D"}


def _save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}")


# ---------------------------------------------------------------- F1: scatter
def fig1_headline_scatter():
    """Reproduce the frozen 12-pancake GAP (full-tier) test predictions and
    plot predicted vs actual for GBM, CDP_1, KRE. Gate on the frozen RMSE."""
    from effortpred.models import split_by_state, fit_gbm
    from effortpred.pancake_features import PANCAKE_FEATURE_NAMES
    from effortpred.pancake import estimate_pancake_distribution, gap_h
    from effortpred.pancake_tree import kre_predict_pancake
    from effortpred.cdp import cdp1_predict
    from effortpred.conditional import H_MAX_PANCAKE, sample_conditional_matrix
    from effortpred.metrics import eval_log10_predictions

    n, seed = 12, 0
    df = pd.read_csv("results/pancake_labels_gap.csv")
    df = df.sort_values(["state", "bound"]).reset_index(drop=True)
    df = df[df["censored"] == 0].copy()
    df["y"] = np.log10(df["nodes"].astype(float))
    train, val, test = split_by_state(df, seed=seed)
    feat = PANCAKE_FEATURE_NAMES
    X_tr, X_te = train[feat].values.astype(float), test[feat].values.astype(float)
    y_tr, y_te = train["y"].values, test["y"].values

    preds = {"gbm": fit_gbm(X_tr, y_tr, seed=seed).predict(X_te)}
    dist = estimate_pancake_distribution(n, gap_h, 200_000, random.Random(seed + 100))
    cond = sample_conditional_matrix(n, gap_h, 200_000, random.Random(seed + 200),
                                     h_max=H_MAX_PANCAKE(n))
    kre_cache, kre_log, cdp_log = {}, [], []
    for h0, bound in zip(test["h_gap"].astype(int), test["bound"].astype(int)):
        if bound not in kre_cache:
            kre_cache[bound] = kre_predict_pancake(n, bound, dist)
        kre_log.append(np.log10(max(kre_cache[bound], 1.0)))
        cdp_log.append(np.log10(max(cdp1_predict(h0, bound, cond, n - 1, n - 2), 1.0)))
    preds["kre"], preds["cdp1"] = np.array(kre_log), np.array(cdp_log)

    frozen = {"gbm": 0.20685616730452358, "cdp1": 0.3077005027122436,
              "kre": 1.1718699124243763}
    rmse = {k: eval_log10_predictions(y_te, p)["rmse_log10"] for k, p in preds.items()}
    for k, v in frozen.items():
        assert abs(rmse[k] - v) < 1e-3, f"GATE FAIL {k}: {rmse[k]:.5f} vs frozen {v:.5f}"
    print(f"  F1 gate OK: gbm {rmse['gbm']:.3f}, cdp1 {rmse['cdp1']:.3f}, kre {rmse['kre']:.3f}")

    lo = min(y_te.min(), *(p.min() for p in preds.values())) - 0.2
    hi = max(y_te.max(), *(p.max() for p in preds.values())) + 0.2
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.35), sharex=True, sharey=True)
    for ax, name in zip(axes, ["gbm", "cdp1", "kre"]):
        ax.plot([lo, hi], [lo, hi], color="0.35", lw=0.8, ls="--", zorder=1)
        ax.scatter(y_te, preds[name], s=3.2, alpha=0.32, color=C[name],
                   edgecolors="none", rasterized=True, zorder=2)
        ax.set_title(f"{LB[name]} (RMSE {rmse[name]:.2f})")
        ax.set_xlabel(r"actual $\log_{10}$ nodes")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
    axes[0].set_ylabel(r"predicted $\log_{10}$ nodes")
    _save(fig, "fig_headline_scatter.pdf")


# ------------------------------------------------------------ F2: depth lines
def fig2_depth():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=True)
    for ax, (h, title) in zip(axes, [("gap", "GAP (consistent)"),
                                      ("rand", "RAND (inconsistent)")]):
        d = pd.read_csv(f"results/pancake_eval_{h}_n10_by_offset.csv")
        for m in ["gbm", "mlp", "cdp1", "kre"]:
            s = d[d["method"] == m].sort_values("offset")
            ax.plot(s["offset"], s["rmse_log10"], marker=MK[m], ms=4,
                    color=C[m], label=LB[m])
        ax.set_title(title)
        ax.set_xlabel(r"offset $\delta$ (bound $-\,h$)")
        ax.set_xticks(sorted(d["offset"].unique()))
        ax.grid(True, lw=0.3, alpha=0.5)
    axes[0].set_ylabel(r"RMSE of $\log_{10}$ nodes")
    # Legend in the GAP panel's empty upper-right (KRE has descended there),
    # so it never occludes the high RAND curves in the right panel.
    axes[0].legend(frameon=False, loc="upper right", handlelength=1.4)
    _save(fig, "fig_depth.pdf")


# ------------------------------------------------------------- F3: tail bars
def fig3_tails():
    d = pd.read_csv("results/tail_errors.csv")
    methods = ["gbm", "mlp", "cdp1", "kre"]
    dsets = [("gap_n12", "GAP"), ("rand_n12", "RAND")]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    x = np.arange(len(dsets)); w = 0.2
    for i, m in enumerate(methods):
        vals = [float(d[(d["dataset"] == ds) & (d["method"] == m)]["factor_at_p95"].iloc[0])
                for ds, _ in dsets]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=C[m], label=LB[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.05,
                    f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center",
                    va="bottom", fontsize=5.5)
    ax.set_yscale("log")
    ax.set_ylim(1, 1500)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in dsets])
    ax.set_ylabel(r"factor error at the 95th percentile")
    ax.axhline(10, color="0.4", lw=0.7, ls=":")
    ax.text(1.45, 11, r"$10\times$", fontsize=6, color="0.4", va="bottom", ha="right")
    ax.legend(frameon=False, ncol=2, handlelength=1.2, columnspacing=1.0,
              loc="upper left")
    _save(fig, "fig_tails.pdf")


# --------------------------------------------------------- F4: learning curve
def fig4_learning_curve():
    from matplotlib.ticker import FixedLocator, FixedFormatter
    d = pd.read_csv("results/learning_curve10.csv")
    d = d[d["heuristic"] == "gap"]
    cdp1 = 0.3077005027122436  # frozen gap full-tier CDP_1 reference
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    # method = colour (consistent across figures); tier = line style + marker
    # fill, so full (solid, filled) is distinct from minimal (dashed, open).
    styles = [("gbm", "full", "-", True), ("mlp", "full", "-", True),
              ("gbm", "minimal", "--", False), ("mlp", "minimal", "--", False)]
    for m, fs, ls, filled in styles:
        g = (d[(d["method"] == m) & (d["feature_set"] == fs)]
             .groupby("size")["rmse_log10"].agg(["mean", "std"]).reset_index())
        ax.plot(g["size"], g["mean"], ls=ls, marker=MK[m], ms=4.5, color=C[m],
                markerfacecolor=(C[m] if filled else "white"),
                markeredgecolor=C[m], markeredgewidth=1.0, label=f"{LB[m]}, {fs}")
        ax.fill_between(g["size"], g["mean"] - g["std"], g["mean"] + g["std"],
                        color=C[m], alpha=0.10, lw=0)
    ax.axhline(cdp1, color=C["cdp1"], lw=1.0, ls=":", label=r"CDP$_1$ (full data)")
    ax.set_xscale("log")
    sizes = [50, 100, 200, 400, 800]
    ax.xaxis.set_major_locator(FixedLocator(sizes))
    ax.xaxis.set_major_formatter(FixedFormatter([str(s) for s in sizes]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("training states")
    ax.set_ylabel(r"RMSE of $\log_{10}$ nodes")
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.legend(frameon=False, handlelength=2.2, fontsize=6)
    _save(fig, "fig_learning_curve.pdf")


# ----------------------------------------------------- F5: feature importance
def fig5_importance():
    d = pd.read_csv("results/mlp_importance_gap.csv").sort_values("importance_mean")
    pretty = {"slack_gap": "slack (GAP)", "slack_gap2": "slack (GAP-2)",
              "slack_rand": "slack (RAND)", "h_gap": "GAP value",
              "h_gap2": "GAP-2 value", "h_rand": "RAND value", "bound": "bound",
              "plate_gap": "plate-gap", "num_fixed_suffix": "solved-suffix",
              "inversions": "inversions", "max_displacement": "max displacement",
              "mean_displacement": "mean displacement", "first_gap_pos": "first-gap pos"}
    labels = [pretty.get(f, f) for f in d["feature"]]
    y = np.arange(len(d))
    colors = ["#0072B2" if "slack" in f else "#999999" for f in d["feature"]]
    fig, ax = plt.subplots(figsize=(COL, 2.9))
    ax.barh(y, d["importance_mean"], xerr=d["importance_std"], color=colors,
            error_kw=dict(lw=0.6, ecolor="0.3"))
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("permutation importance (increase in RMSE)")
    ax.grid(True, axis="x", lw=0.3, alpha=0.5)
    _save(fig, "fig_importance.pdf")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    print(f"figures -> {FIGDIR}")
    fig1_headline_scatter()
    fig2_depth()
    fig3_tails()
    fig4_learning_curve()
    fig5_importance()
    print("done.")
