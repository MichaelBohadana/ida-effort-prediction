import numpy as np
import pandas as pd

from effortpred.models import (
    GapBaseline, MeanBaseline, fit_gbm, fit_mlp, split_by_state,
)


def _synthetic_df(n_states=60, rows_per_state=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_states):
        x1, x2 = rng.normal(size=2)
        for j in range(rows_per_state):
            gap = 2 * j
            rows.append({
                "state": f"s{i}", "f1": x1, "f2": x2, "gap": gap,
                "y": 1.5 * x1 - 0.7 * x2 + 0.4 * gap + rng.normal(0, 0.05),
            })
    return pd.DataFrame(rows)


def test_split_by_state_no_leakage():
    df = _synthetic_df()
    train, val, test = split_by_state(df, seed=0)
    st = set(train["state"]); sv = set(val["state"]); ste = set(test["state"])
    assert st.isdisjoint(sv) and st.isdisjoint(ste) and sv.isdisjoint(ste)
    assert len(train) + len(val) + len(test) == len(df)
    assert len(train) > len(val) and len(train) > len(test)


def test_gap_baseline_table():
    b = GapBaseline().fit(np.array([0, 0, 2, 2]), np.array([1.0, 3.0, 5.0, 7.0]))
    assert np.allclose(b.predict(np.array([0, 2])), [2.0, 6.0])
    # unseen gap falls back to the global mean
    assert np.allclose(b.predict(np.array([4])), [4.0])


def test_gbm_and_mlp_beat_mean_baseline_on_synthetic_data():
    df = _synthetic_df()
    train, val, test = split_by_state(df, seed=0)
    feats = ["f1", "f2", "gap"]
    X_tr, y_tr = train[feats].values, train["y"].values
    X_va, y_va = val[feats].values, val["y"].values
    X_te, y_te = test[feats].values, test["y"].values

    def rmse(p):
        return float(np.sqrt(np.mean((p - y_te) ** 2)))

    base = rmse(MeanBaseline().fit(X_tr, y_tr).predict(X_te))
    gbm = rmse(fit_gbm(X_tr, y_tr, seed=0).predict(X_te))
    mlp = rmse(fit_mlp(X_tr, y_tr, X_va, y_va, seed=0)(X_te))
    assert gbm < base / 2
    assert mlp < base / 2
