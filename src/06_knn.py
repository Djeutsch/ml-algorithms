# %% [markdown]
# # 6 · K-nearest neighbours
#
# ```{math}
# \hat y = \operatorname{mode}\{\, y_i : i \in N_k(x) \,\}
# ```
#
# **Family:** distance-based · **Task:** classification
#
# The video puts a yellow circle around a query point and draws spokes to its five
# nearest neighbours, $N_5(x)$. There is no training step: the model *is* the training
# set, and prediction is a majority vote among the $k$ closest points (Euclidean
# distance here).

# %% [markdown]
# ## Imports
#
# `ml8` is the book's own package; [The sample data](00_sample_data.ipynb) explains how it is
# imported. This chapter uses:
#
# | File | What this chapter uses |
# |---|---|
# | `ml8/data.py` | `data.moons_arrays`: two-moons train and test arrays (`X_train, y_train, X_test, y_test`) |
# | `ml8/scratch.py` | `scratch.knn_predict`: KNN prediction in NumPy |
# | `ml8/models.py` | `models.knn`: a scikit-learn `KNeighborsClassifier` for a given $k$<br>`models.KNN_KS`: the values of $k$ to compare |
# | `ml8/plotting.py` | `P.plot_surface`: draws a decision surface and the points<br>the palette (`P.YELLOW`) |

# %%
# ml8 sits one folder above notebooks/, so put that folder on Python's search path
import sys
sys.path.insert(0, "..")

# Installed packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# The book's own modules
from ml8 import data              # ml8/data.py: the seeded sample datasets
from ml8 import models            # ml8/models.py: shared model settings
from ml8 import plotting as P     # ml8/plotting.py: colours and plot helpers
from ml8 import scratch           # ml8/scratch.py: the from-scratch formulas

P.use_style()  # the dark style used by every figure
pd.set_option("display.precision", 4)

# %%
Xtr, ytr, Xte, yte = data.moons_arrays()
q = np.array([[0.45, 0.3]])
pred, nn = scratch.knn_predict(Xtr, ytr, q, k=5)
d = np.linalg.norm(Xtr[nn[0]] - q, axis=1)
pd.DataFrame({"x1": Xtr[nn[0], 0], "x2": Xtr[nn[0], 1], "label": ytr[nn[0]], "distance": d})

# %%
fig, ax = plt.subplots(figsize=(5.5, 3.8))
P.plot_surface(ax, models.knn(5).fit(Xtr, ytr), Xtr, ytr, title=f"k = 5 · query predicted as class {pred[0]}")
ax.add_patch(plt.Circle(q[0], d.max(), fill=False, ec=P.YELLOW, lw=1.5))
for p in Xtr[nn[0]]:
    ax.plot([q[0, 0], p[0]], [q[0, 1], p[1]], c=P.YELLOW, lw=1.2)
ax.scatter(*q[0], s=60, c=P.YELLOW, zorder=5);

# %% [markdown]
# ## From scratch versus scikit-learn
#
# The two implementations should give identical predictions on the test set (ties are
# impossible with odd $k$ and two classes).

# %%
for k in models.KNN_KS:
    mine, _ = scratch.knn_predict(Xtr, ytr, Xte, k=k)
    theirs = models.knn(k).fit(Xtr, ytr).predict(Xte)
    print(f"k={k:>2}: identical = {np.array_equal(mine, theirs)}, test accuracy = {np.mean(mine == yte):.3f}")

# %% [markdown]
# ## Choosing k: the bias–variance dial

# %%
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, k in zip(axes, [1, 5, 15, 45]):
    m = models.knn(k).fit(Xtr, ytr)
    P.plot_surface(ax, m, Xtr, ytr, title=f"k = {k} · train {m.score(Xtr, ytr):.0%} · test {m.score(Xte, yte):.1%}")
plt.tight_layout()

# %% [markdown]
# With $k=1$ every training point is its own neighbour, so training accuracy is 100% and
# the boundary is ragged (high variance). Large $k$ smooths the boundary until it starts
# to ignore the curve of the moons (high bias).
#
# ```{important}
# Distances mix features directly, so features on different scales must be standardised
# first (`StandardScaler`). Both moons features happen to share a scale.
# ```
