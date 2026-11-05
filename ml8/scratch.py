"""Small from-scratch implementations of the formulas shown in the video.

They are deliberately short and readable; scikit-learn is used alongside them
in the notebooks to confirm they give the same answers.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --- 1. Linear regression:  y_hat = w x + b ---------------------------------
def linear_gd(x, y, w0=-0.2, b0=5.0, lr=0.02, iters=3000, record=None):
    """Gradient descent on the mean squared error.

    Returns final (w, b) and a list of (iter, w, b, mse) for the iterations in
    `record` (all iterations if None).
    """
    w, b, n = w0, b0, len(x)
    hist = []
    for t in range(iters + 1):
        y_hat = w * x + b
        mse = np.mean((y_hat - y) ** 2)
        if record is None or t in record:
            hist.append((t, w, b, mse))
        grad_w = 2 / n * np.sum((y_hat - y) * x)
        grad_b = 2 / n * np.sum(y_hat - y)
        w, b = w - lr * grad_w, b - lr * grad_b
    return (w, b), hist


# --- 2. Logistic regression:  p = sigma(w x + b) -----------------------------
def logistic_gd(x, y, w0=0.0, b0=0.0, lr=0.12, iters=8000, record=None):
    """Gradient descent on the mean log-loss (binary cross-entropy)."""
    w, b, n = w0, b0, len(x)
    hist = []
    for t in range(iters + 1):
        p = sigmoid(w * x + b)
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        if record is None or t in record:
            acc = np.mean((p >= 0.5) == y)
            hist.append((t, w, b, loss, acc))
        w, b = w - lr * np.mean((p - y) * x), b - lr * np.mean(p - y)
    return (w, b), hist


# --- 3. Decision tree:  G = 1 - sum_k p_k^2 ----------------------------------
def gini(labels) -> float:
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(1 - np.sum(p ** 2))


def best_split(X, y):
    """Exhaustive search for the axis-aligned split with lowest weighted Gini."""
    best = (None, None, gini(y))  # feature, threshold, weighted gini
    n = len(y)
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for thr in (values[:-1] + values[1:]) / 2:
            left = X[:, f] <= thr
            g = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / n
            if g < best[2]:
                best = (f, float(thr), float(g))
    return best


# --- 6. KNN:  y_hat = mode{ y_i : i in N_k(x) } ------------------------------
def knn_predict(X_train, y_train, X_query, k=5):
    d = np.linalg.norm(X_query[:, None, :] - X_train[None, :, :], axis=2)
    nn = np.argsort(d, axis=1)[:, :k]
    votes = y_train[nn]
    return (votes.mean(axis=1) > 0.5).astype(int), nn


# --- 7. k-means:  mu_j = (1/|C_j|) sum_{i in C_j} x_i ------------------------
def kmeans_lloyd(X, k, seed=0, max_iter=50):
    """Lloyd's algorithm. Returns centroids per iteration and assignments per iteration.

    Iteration 0 = random initial centroids (picked from the data) and the
    assignment they induce; each later iteration recomputes the means.
    """
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), size=k, replace=False)].astype(float)
    cents, assigns = [], []
    for _ in range(max_iter):
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        a = d.argmin(axis=1)
        cents.append(centroids.copy())
        assigns.append(a.copy())
        new = np.array([X[a == j].mean(axis=0) if np.any(a == j) else centroids[j] for j in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    return cents, assigns


def inertia(X, centroids, assign) -> float:
    return float(np.sum((X - centroids[assign]) ** 2))
