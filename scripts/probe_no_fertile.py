"""Ablation: does the probe tier's gain survive WITHOUT n_fertile (the
label-summand feature)? Answer (2026-07-03): yes — MLP essentially unchanged
(gap 0.154->0.156, rand 0.209->0.214); full_probe without n_fertile matches or
beats with (rand 0.155/0.152). DECISION: the report's probe tier uses the 5
transition features only; n_fertile is a footnote. Also demonstrates
importance-vs-ablation: permutation importance ranked n_fertile near the top,
yet removing it costs ~nothing (correlated features shadow each other).
Output: results/probe_no_fertile.csv
"""

import numpy as np
import pandas as pd

from effortpred.models import fit_gbm, fit_mlp, split_by_state
from effortpred.pancake_features import PANCAKE_FEATURE_NAMES
from effortpred.probe_features import PROBE_FEATURE_NAMES, extract_probe_features


def rmse(p, t):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(t)) ** 2)))


def main():
    rows_out = []
    for h in ("gap", "rand"):
        df = pd.read_csv(f"results/pancake_labels_{h}.csv")
        df = df[df.censored == 0].copy()
        df = df.sort_values(["state", "bound"]).reset_index(drop=True)
        df["y"] = np.log10(df.nodes.astype(float))
        probe = pd.DataFrame(
            [extract_probe_features(tuple(int(x) for x in s.split()), int(b))
             for s, b in zip(df.state, df.bound)], index=df.index)
        df = pd.concat([df, probe], axis=1)
        train, val, test = split_by_state(df, seed=0)
        hcol = f"h_{h}"
        active6 = [f"probe_{h}_{s}" for s in
                   ("min_child", "mean_child", "max_child",
                    "n_improving", "n_worsening", "n_fertile")]
        active5 = [f for f in active6 if not f.endswith("n_fertile")]
        tiers = {
            "minimal_probe_all6": [hcol, "bound"] + active6,
            "minimal_probe_no_fertile": [hcol, "bound"] + active5,
            "minimal_plus_fertile_only": [hcol, "bound", f"probe_{h}_n_fertile"],
            "full_probe_no_fertile": PANCAKE_FEATURE_NAMES + [
                f for f in PROBE_FEATURE_NAMES if not f.endswith("n_fertile")],
        }
        for name, feats in tiers.items():
            Xtr, Xva, Xte = (d[feats].values.astype(float)
                             for d in (train, val, test))
            g = rmse(fit_gbm(Xtr, train.y.values, seed=0).predict(Xte),
                     test.y.values)
            m = rmse(fit_mlp(Xtr, train.y.values, Xva, val.y.values,
                             seed=0)(Xte), test.y.values)
            print(f"{h} {name:28s} GBM {g:.4f}  MLP {m:.4f}")
            rows_out.append({"heuristic": h, "tier": name, "gbm": g, "mlp": m})
    pd.DataFrame(rows_out).to_csv("results/probe_no_fertile.csv", index=False)


if __name__ == "__main__":
    main()
