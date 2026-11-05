"""Export every dataset the dashboard reads, as CSV files.

Usage:  python scripts/export_dashboard_data.py [out_dir]   (default: dashboard/data)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml8 import data, models, scratch  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "dashboard" / "data")
OUT.mkdir(parents=True, exist_ok=True)


def save(df: pd.DataFrame, name: str):
    df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f")
    print(f"{name:16s} {len(df):6d} rows  {(OUT / f'{name}.csv').stat().st_size/1024:7.1f} KB")


# ---------------------------------------------------------------- 1. linear
reg = data.regression_data()
save(reg, "reg_points")
(w, b), hist = scratch.linear_gd(reg.x.to_numpy(), reg.y.to_numpy(), record=set(models.LINEAR_RECORD))
save(pd.DataFrame(hist, columns=["iter", "w", "b", "mse"]), "reg_gd")
ols = LinearRegression().fit(reg[["x"]], reg.y)
r2 = ols.score(reg[["x"]], reg.y)

# ---------------------------------------------------------------- 2. logistic
lg = data.logistic_data()
save(lg, "logit_points")
_, hist = scratch.logistic_gd(lg.x.to_numpy(), lg.y.to_numpy(), record=set(models.LOGISTIC_RECORD))
lg_gd = pd.DataFrame(hist, columns=["iter", "w", "b", "logloss", "acc"])
save(lg_gd, "logit_gd")

# ---------------------------------------------------------------- moons
moons = data.moons_data()
save(moons, "moons")
Xtr, ytr, Xte, yte = data.moons_arrays(moons)
train_ids = moons[moons.split == "train"].id.to_numpy()
_, _, pts = data.grid()
N = data.GRID_N
save(pd.DataFrame([{"n": N, "x1_min": data.X1_RANGE[0], "x1_max": data.X1_RANGE[1],
                    "x2_min": data.X2_RANGE[0], "x2_max": data.X2_RANGE[1]}]), "grid")

surf_rows, metric_rows = [], []


def add_surface(model_name, param, proba):
    for idx, p in enumerate(proba):
        surf_rows.append((model_name, param, idx % N, idx // N, round(float(p), 3)))


def add_metric(model_name, param, clf=None, pred_tr=None, pred_te=None):
    if clf is not None:
        pred_tr, pred_te = clf.predict(Xtr), clf.predict(Xte)
    metric_rows.append({"model": model_name, "param": param,
                        "train_acc": float(np.mean(pred_tr == ytr)), "test_acc": float(np.mean(pred_te == yte)),
                        "train_mistakes": int(np.sum(pred_tr != ytr))})


# 3. tree
split_rows = []
for d in models.TREE_DEPTHS:
    clf = models.tree(d).fit(Xtr, ytr)
    add_surface("tree", d, clf.predict_proba(pts)[:, 1])
    add_metric("tree", d, clf)
    t = clf.tree_

    def walk(node, box, depth):
        if t.children_left[node] == -1:
            return
        f, thr = int(t.feature[node]), float(t.threshold[node])
        x0, x1, y0, y1 = box
        seg = (thr, thr, y0, y1) if f == 0 else (x0, x1, thr, thr)
        split_rows.append({"max_depth": d, "node": node, "node_depth": depth, "feature": f"x{f+1}",
                           "threshold": thr, "sx0": seg[0], "sx1": seg[1], "sy0": seg[2], "sy1": seg[3],
                           "gini": float(t.impurity[node]), "samples": int(t.n_node_samples[node])})
        lbox = (x0, thr, y0, y1) if f == 0 else (x0, x1, y0, thr)
        rbox = (thr, x1, y0, y1) if f == 0 else (x0, x1, thr, y1)
        walk(t.children_left[node], lbox, depth + 1)
        walk(t.children_right[node], rbox, depth + 1)

    walk(0, (*data.X1_RANGE, *data.X2_RANGE), 0)
save(pd.DataFrame(split_rows), "tree_splits")

# 4. forest (+ 9 individual member trees, like the video's 3x3 grid)
member_rows = []
for T in models.FOREST_SIZES:
    clf = models.forest(T).fit(Xtr, ytr)
    add_surface("forest", T, clf.predict_proba(pts)[:, 1])
    add_metric("forest", T, clf)
    if T == max(models.FOREST_SIZES):
        for m, est in enumerate(clf.estimators_[:9]):
            for idx, p in enumerate(est.predict_proba(pts)[:, 1]):
                member_rows.append((m, idx % N, idx // N, round(float(p), 3)))
save(pd.DataFrame(member_rows, columns=["tree", "i", "j", "p1"]), "forest_members")

# 5. gradient boosting
gb = models.boosting().fit(Xtr, ytr)
stages = set(models.BOOST_STAGES)
mistake_rows = []
for s, (ptr, pte, pg) in enumerate(zip(gb.staged_predict(Xtr), gb.staged_predict(Xte),
                                       gb.staged_predict_proba(pts)), start=1):
    if s in stages:
        add_surface("boosting", s, pg[:, 1])
        add_metric("boosting", s, pred_tr=ptr, pred_te=pte)
        for i in np.flatnonzero(ptr != ytr):
            mistake_rows.append({"stage": s, "id": int(train_ids[i])})
save(pd.DataFrame(mistake_rows, columns=["stage", "id"]), "boost_mistakes")

# 6. KNN
for k in models.KNN_KS:
    clf = models.knn(k).fit(Xtr, ytr)
    add_surface("knn", k, clf.predict_proba(pts)[:, 1])
    add_metric("knn", k, clf)

# 8. neural net
mlp, nn_hist, nn_surf = models.train_mlp(Xtr, ytr, X_test=Xte, y_test=yte)
for ep, p in nn_surf.items():
    add_surface("nn", ep, p)
for ep in models.NN_EPOCHS:
    h = nn_hist[ep - 1]
    metric_rows.append({"model": "nn", "param": ep, "train_acc": h["train_acc"], "test_acc": h["test_acc"],
                        "train_mistakes": int(round((1 - h["train_acc"]) * len(ytr)))})
save(pd.DataFrame(nn_hist), "nn_training")
w_rows = []
for layer, W in enumerate(mlp.coefs_):
    for src in range(W.shape[0]):
        for dst in range(W.shape[1]):
            w_rows.append({"layer": layer, "src": src, "dst": dst, "weight": float(W[src, dst])})
save(pd.DataFrame(w_rows), "nn_weights")

# logistic regression on moons (baseline for the comparison)
lr2 = models.logistic_2d().fit(Xtr, ytr)
add_surface("logistic", 0, lr2.predict_proba(pts)[:, 1])
add_metric("logistic", 0, lr2)

save(pd.DataFrame(surf_rows, columns=["model", "param", "i", "j", "p1"]), "surfaces")
save(pd.DataFrame(metric_rows), "model_metrics")

# ---------------------------------------------------------------- 7. k-means
blobs = data.blobs_data()
save(blobs, "blobs")
Xb = blobs[["x1", "x2"]].to_numpy()
step_rows, assign_rows, elbow_rows = [], [], []
for k in [2, 3, 4, 5]:
    cents, assigns = scratch.kmeans_lloyd(Xb, k, seed=1)
    for it, (c, a) in enumerate(zip(cents, assigns)):
        for j in range(k):
            step_rows.append({"k": k, "iter": it, "cluster": j, "cx": c[j, 0], "cy": c[j, 1]})
        for i, aj in enumerate(a):
            assign_rows.append((k, it, i, int(aj)))
    elbow_rows.append({"k": k, "inertia": scratch.inertia(Xb, cents[-1], assigns[-1]),
                       "iterations": len(cents) - 1,
                       "ari": adjusted_rand_score(blobs.true_cluster, assigns[-1])})
save(pd.DataFrame(step_rows), "kmeans_steps")
save(pd.DataFrame(assign_rows, columns=["k", "iter", "id", "cluster"]), "kmeans_assign")
elbow = pd.DataFrame(elbow_rows)
save(elbow, "kmeans_elbow")

# ---------------------------------------------------------------- scorecard
m = pd.DataFrame(metric_rows)


def acc(model, param):
    return float(m[(m.model == model) & (m.param == param)].test_acc.iloc[0])


score = pd.DataFrame([
    ("linear", "Linear regression", "Linear models", "Regression", "R² (sample line)", r2),
    ("logistic", "Logistic regression", "Linear models", "Classification", "Test accuracy (moons)", acc("logistic", 0)),
    ("tree", "Decision tree", "Tree models", "Classification", "Test accuracy (moons, depth 4)", acc("tree", 4)),
    ("forest", "Random forest", "Tree models", "Classification", "Test accuracy (moons, 100 trees)", acc("forest", 100)),
    ("boosting", "Gradient boosting", "Tree models", "Classification", "Test accuracy (moons, 60 stages)", acc("boosting", 60)),
    ("knn", "K-nearest neighbours", "Distance-based", "Classification", "Test accuracy (moons, k = 15)", acc("knn", 15)),
    ("kmeans", "K-means", "Distance-based", "Clustering", "Adjusted Rand index (blobs, k = 3)",
     float(elbow[elbow.k == 3].ari.iloc[0])),
    ("nn", "Neural network", "Neural networks", "Classification", "Test accuracy (moons, 400 epochs)", acc("nn", 400)),
], columns=["key", "algorithm", "family", "task", "metric", "value"])
save(score, "scorecard")
print(score.to_string())
print("OLS:", ols.coef_[0], ols.intercept_, "GD:", w, b)
