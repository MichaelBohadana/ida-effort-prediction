"""Predictors: leakage-safe split, trivial baselines, GBM, small MLP.
Target everywhere: log10(nodes)."""

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from torch import nn


def split_by_state(df, seed=0):
    """70/15/15 train/val/test, grouped by the `state` column so no state
    appears on both sides of any boundary."""
    groups = df["state"].values
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    (train_idx, rest_idx), = outer.split(df, groups=groups)
    rest = df.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    (val_idx, test_idx), = inner.split(rest, groups=rest["state"].values)
    return (df.iloc[train_idx].copy(),
            rest.iloc[val_idx].copy(),
            rest.iloc[test_idx].copy())


class MeanBaseline:
    """Predict the training mean. The floor every model must beat."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class GapBaseline:
    """Predict the training mean log-nodes for the row's gap = bound - h.
    The strong 'trivial' baseline: a learned model must beat this to claim
    its features carry signal beyond (h, bound)."""

    def fit(self, gaps, y):
        df = pd.DataFrame({"gap": np.asarray(gaps), "y": np.asarray(y, dtype=float)})
        self.table_ = df.groupby("gap")["y"].mean().to_dict()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, gaps):
        return np.array([self.table_.get(g, self.global_) for g in np.asarray(gaps)])


def fit_gbm(X_train, y_train, seed=0):
    model = HistGradientBoostingRegressor(
        random_state=seed, max_iter=500,
        early_stopping=True, validation_fraction=0.15,
    )
    model.fit(X_train, y_train)
    return model


def fit_mlp(X_train, y_train, X_val, y_val, seed=0,
            max_epochs=500, patience=25, lr=1e-3, batch_size=256):
    """Small MLP on standardized features, early-stopped on validation MSE.
    Returns predict(X) -> np.ndarray."""
    torch.manual_seed(seed)
    X_train = np.asarray(X_train, dtype=np.float64)
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0) + 1e-9

    def prep(X):
        return torch.tensor((np.asarray(X, dtype=np.float64) - mu) / sigma,
                            dtype=torch.float32)

    Xt = prep(X_train)
    yt = torch.tensor(np.asarray(y_train), dtype=torch.float32).unsqueeze(1)
    Xv = prep(X_val)
    yv = torch.tensor(np.asarray(y_val), dtype=torch.float32).unsqueeze(1)

    model = nn.Sequential(
        nn.Linear(Xt.shape[1], 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    for _epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), yv).item()
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return model(prep(X)).squeeze(1).numpy()

    return predict
