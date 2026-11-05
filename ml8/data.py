"""Sample datasets shared by the book and the dashboard.

Every generator is seeded, so the notebooks and the exported dashboard data
always contain exactly the same points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split

SEED = 7

# Grid used for every 2-D decision surface (moons data).
GRID_N = 32
X1_RANGE = (-1.7, 2.8)
X2_RANGE = (-1.3, 1.8)


def regression_data(n: int = 60, seed: int = SEED) -> pd.DataFrame:
    """1-D regression: y = 1.8 x + 3 + noise, x in [0, 10]."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    y = 1.8 * x + 3 + rng.normal(0, 2.5, n)
    return pd.DataFrame({"id": np.arange(n), "x": x.round(4), "y": y.round(4)})


def logistic_data(n: int = 60, seed: int = SEED) -> pd.DataFrame:
    """1-D binary data: P(y=1 | x) = sigmoid(1.6 (x - 5))."""
    rng = np.random.default_rng(seed + 1)
    x = np.sort(rng.uniform(0, 10, n))
    p = 1 / (1 + np.exp(-1.6 * (x - 5)))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"id": np.arange(n), "x": x.round(4), "y": y})


def moons_data(n: int = 400, noise: float = 0.3, seed: int = SEED) -> pd.DataFrame:
    """Two interleaving half-moons with a 70/30 train/test split."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    idx_train, _ = train_test_split(np.arange(n), test_size=0.3, random_state=seed, stratify=y)
    split = np.where(np.isin(np.arange(n), idx_train), "train", "test")
    return pd.DataFrame(
        {"id": np.arange(n), "x1": X[:, 0].round(4), "x2": X[:, 1].round(4), "label": y, "split": split}
    )


def moons_arrays(df: pd.DataFrame | None = None):
    """Return X_train, y_train, X_test, y_test as numpy arrays."""
    df = moons_data() if df is None else df
    tr, te = df[df.split == "train"], df[df.split == "test"]
    return (tr[["x1", "x2"]].to_numpy(), tr.label.to_numpy(), te[["x1", "x2"]].to_numpy(), te.label.to_numpy())


def blobs_data(n: int = 150, seed: int = 3) -> pd.DataFrame:
    """Three Gaussian blobs for k-means (true labels kept only for scoring)."""
    X, y = make_blobs(n_samples=n, centers=3, cluster_std=0.9, random_state=seed)
    return pd.DataFrame({"id": np.arange(n), "x1": X[:, 0].round(4), "x2": X[:, 1].round(4), "true_cluster": y})


def grid():
    """Cell centres of the GRID_N x GRID_N surface grid: (xx, yy, points)."""
    gx = np.linspace(*X1_RANGE, GRID_N)
    gy = np.linspace(*X2_RANGE, GRID_N)
    xx, yy = np.meshgrid(gx, gy)  # xx[j, i] -> column i, row j
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]
