"""Canonical model settings shared by the notebooks and the dashboard export."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from . import data

TREE_DEPTHS = [1, 2, 3, 4, 5, 6]
FOREST_SIZES = [1, 3, 5, 10, 25, 50, 100]
BOOST_STAGES = [1, 2, 3, 5, 8, 12, 20, 30, 45, 60]
KNN_KS = [1, 3, 5, 15, 45]
NN_EPOCHS = [1, 10, 25, 50, 100, 200, 400]
NN_HIDDEN = (8, 8)

LINEAR_RECORD = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400, 800, 1500, 3000]
LOGISTIC_RECORD = [0, 5, 10, 20, 40, 80, 150, 300, 600, 1000, 1500, 2500, 4000, 6000, 8000]


def tree(depth: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(criterion="gini", max_depth=depth, random_state=data.SEED)


def forest(n_trees: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_trees, max_depth=6, random_state=data.SEED)


def boosting(n_stages: int = max(BOOST_STAGES)) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=n_stages, learning_rate=0.3, max_depth=3, random_state=data.SEED)


def knn(k: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k)


def logistic_2d() -> LogisticRegression:
    return LogisticRegression()


def train_mlp(X, y, epochs=NN_EPOCHS, X_test=None, y_test=None):
    """Train an MLP one epoch at a time with partial_fit.

    Returns the model, a list of per-epoch dicts and a dict of grid
    probabilities at each epoch in `epochs`.
    """
    mlp = MLPClassifier(hidden_layer_sizes=NN_HIDDEN, activation="tanh", solver="adam",
                        learning_rate_init=0.02, batch_size=32, random_state=data.SEED)
    _, _, pts = data.grid()
    rng = np.random.default_rng(data.SEED)
    history, surfaces = [], {}
    for ep in range(1, max(epochs) + 1):
        order = rng.permutation(len(X))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp.partial_fit(X[order], y[order], classes=[0, 1])
        row = {"epoch": ep, "loss": float(mlp.loss_), "train_acc": float(mlp.score(X, y))}
        if X_test is not None:
            row["test_acc"] = float(mlp.score(X_test, y_test))
        history.append(row)
        if ep in epochs:
            surfaces[ep] = mlp.predict_proba(pts)[:, 1]
    return mlp, history, surfaces
