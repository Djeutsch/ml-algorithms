# %% [markdown]
# # The sample data
#
# Every chapter uses one of four small, seeded datasets. They live in `ml8/data.py`,
# so the notebooks and the companion dashboard always show exactly the same points.
#
# | Dataset | Used by | Shape |
# |---|---|---|
# | Regression line | Linear regression | 60 points, $y = 1.8x + 3 + \varepsilon$, $\varepsilon \sim \mathcal N(0, 2.5^2)$ |
# | Logistic 1-D | Logistic regression | 60 points, $P(y=1\mid x) = \sigma(1.6(x-5))$ |
# | Two moons | Tree, forest, boosting, KNN, neural net | 400 points, noise 0.3, 70/30 stratified train/test split |
# | Blobs | k-means | 150 points in three Gaussian clusters |

# %% [markdown]
# ## How the notebooks import code
#
# Every chapter opens with an **Imports** cell. It loads two kinds of code:
#
# - **Installed packages** (NumPy, pandas, Matplotlib, scikit-learn), which come from
#   `pip install -r requirements.txt`.
# - **The book's own package, `ml8`**, which is a folder in this repository rather than
#   something you install:
#
# ```text
# eight-ml-algorithms/
# ├── ml8/
# │   ├── data.py        the four datasets on this page
# │   ├── scratch.py     the formulas, written from scratch in NumPy
# │   ├── models.py      scikit-learn models with the book's settings
# │   └── plotting.py    the colours and plot helpers
# └── notebooks/         you are here
# ```
#
# | Import | File | Used as |
# |---|---|---|
# | `from ml8 import data` | `ml8/data.py` | `data.moons_arrays()` |
# | `from ml8 import scratch` | `ml8/scratch.py` | `scratch.gini(labels)` |
# | `from ml8 import models` | `ml8/models.py` | `models.tree(depth=3)` |
# | `from ml8 import plotting as P` | `ml8/plotting.py` | `P.plot_surface(...)`, `P.BLUE` |
#
# Python only finds `ml8` if its parent folder is on the search path, and the notebooks run
# from `notebooks/`, one level below it. That is what `sys.path.insert(0, "..")` does, so
# it needs to run before any `ml8` import.
#
# Here is the full cell for this page. Each chapter's cell has the same shape and imports
# only what that chapter uses.

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
from ml8 import plotting as P     # ml8/plotting.py: colours and plot helpers

P.use_style()  # the dark style used by every figure
pd.set_option("display.precision", 4)

# %%
reg, lg, moons, blobs = data.regression_data(), data.logistic_data(), data.moons_data(), data.blobs_data()
pd.DataFrame({
    "rows": [len(reg), len(lg), len(moons), len(blobs)],
    "columns": [", ".join(d.columns) for d in (reg, lg, moons, blobs)],
}, index=["regression", "logistic", "moons", "blobs"])

# %%
fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
ax[0].scatter(reg.x, reg.y, s=14, c=P.BLUE); ax[0].set_title("Regression line")
ax[1].scatter(lg.x, lg.y, s=14, c=P.point_colors(lg.y)); ax[1].set_title("Logistic 1-D"); ax[1].set_yticks([0, 1])
tr = moons[moons.split == "train"]; te = moons[moons.split == "test"]
ax[2].scatter(tr.x1, tr.x2, s=12, c=P.point_colors(tr.label))
ax[2].scatter(te.x1, te.x2, s=22, facecolors="none", edgecolors=P.point_colors(te.label), linewidths=0.9)
ax[2].set_title("Two moons (rings = test)")
ax[3].scatter(blobs.x1, blobs.x2, s=12, c=np.array([P.BLUE, P.GREEN, P.PURPLE])[blobs.true_cluster]); ax[3].set_title("Blobs (true groups)")
plt.tight_layout()

# %% [markdown]
# The two-moons set is the workhorse: its classes curl into each other, so a straight
# boundary cannot separate them. That makes the differences between the algorithms
# easy to see.

# %%
moons.groupby(["split", "label"]).size().unstack()
