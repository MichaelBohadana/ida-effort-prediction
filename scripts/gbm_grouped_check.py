"""Quantification of the cross-model audit's GBM finding (2026-07-03):
fit_gbm uses sklearn's INTERNAL row-level 15% early-stopping split rather than
the grouped validation set. This script reruns the GBM under the strictly
correct protocol (no internal early stopping; max_iter chosen on the
group-disjoint val set; single test evaluation). RESULT: deltas vs published
numbers are -0.013..+0.002 — i.e., the published GBM numbers are the SAME or
slightly PESSIMISTIC. No conclusion or number requires revision; the flaw's
direction is against the learned models, not for them. Test set was never
touched during fitting under either protocol (audit-confirmed)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from effortpred.models import split_by_state
from effortpred.pancake_features import PANCAKE_FEATURE_NAMES


def rmse(p, t):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(t)) ** 2)))


def main():
    published = {("gap", "full"): 0.207, ("gap", "minimal"): 0.237,
                 ("rand", "full"): 0.266, ("rand", "minimal"): 0.582}
    for h in ("gap", "rand"):
        df = pd.read_csv(f"results/pancake_labels_{h}.csv")
        df = df[df.censored == 0].sort_values(["state", "bound"]).reset_index(drop=True)
        df["y"] = np.log10(df.nodes.astype(float))
        train, val, test = split_by_state(df, seed=0)
        for tier, feats in (("full", PANCAKE_FEATURE_NAMES),
                            ("minimal", [f"h_{h}", "bound"])):
            Xtr, Xva, Xte = (d[feats].values.astype(float)
                             for d in (train, val, test))
            best = None
            for it in range(25, 526, 25):
                m = HistGradientBoostingRegressor(
                    random_state=0, max_iter=it, early_stopping=False)
                m.fit(Xtr, train.y.values)
                v = rmse(m.predict(Xva), val.y.values)
                if best is None or v < best[0]:
                    best = (v, it, m)
            _, it, m = best
            t = rmse(m.predict(Xte), test.y.values)
            pub = published[(h, tier)]
            print(f"{h}/{tier}: grouped GBM test rmse {t:.4f} (max_iter={it}) "
                  f"vs published {pub:.3f}  delta {t - pub:+.4f}")


if __name__ == "__main__":
    main()
