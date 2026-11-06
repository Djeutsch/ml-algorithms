# %% [markdown]
# # 4 · Random forest
#
# ```{math}
# \hat y = \operatorname{mode}\big(h_1(x), \dots, h_T(x)\big)
# ```
#
# **Family:** tree models · **Task:** classification
#
# The video shows a 3×3 grid of jagged single-tree regions and then one smoother map:
# their vote. Each tree $h_t$ is grown on a bootstrap resample of the training set and
# considers only a random subset of features at each split, so the trees make
# different mistakes and those mistakes partly cancel.
#
# ```{note}
# The video's formula is a hard majority vote. scikit-learn's `RandomForestClassifier`
# actually averages the trees' predicted probabilities (soft voting). Below we compute
# both and measure how often they disagree.
# ```

# %% [markdown]
# ## Imports
#
# `ml8` is the book's own package; [The sample data](00_sample_data.ipynb) explains how it is
# imported. This chapter uses:
#
# | File | What this chapter uses |
# |---|---|
# | `ml8/data.py` | `data.grid`: the grid of points behind every decision surface<br>`data.moons_arrays`: two-moons train and test arrays (`X_train, y_train, X_test, y_test`) |
# | `ml8/models.py` | `models.forest`: a scikit-learn `RandomForestClassifier` of a given size<br>`models.FOREST_SIZES`: the forest sizes to compare |
# | `ml8/plotting.py` | `P.plot_surface`: draws a decision surface and the points<br>the palette (`P.FG`, `P.GREEN`) |

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

P.use_style()  # the dark style used by every figure
pd.set_option("display.precision", 4)

# %%
Xtr, ytr, Xte, yte = data.moons_arrays()
rf = models.forest(100).fit(Xtr, ytr)

fig, axes = plt.subplots(3, 3, figsize=(8, 5.6))
for ax, est in zip(axes.ravel(), rf.estimators_[:9]):
    P.plot_surface(ax, est.predict_proba(data.grid()[2])[:, 1])
fig.suptitle("Nine of the hundred trees", color=P.FG)
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize=(5.5, 3.8))
P.plot_surface(ax, rf, Xtr, ytr, title=f"Vote of 100 trees · test accuracy {rf.score(Xte, yte):.1%}");

# %% [markdown]
# ## Hard vote (the video's mode) versus soft vote (scikit-learn)

# %%
votes = np.stack([est.predict(Xte) for est in rf.estimators_])      # T x n_test
hard = (votes.mean(axis=0) > 0.5).astype(int)                         # mode for two classes
soft = rf.predict(Xte)
print(f"hard-vote test accuracy {np.mean(hard == yte):.3f}")
print(f"soft-vote test accuracy {np.mean(soft == yte):.3f}")
print(f"points where they disagree: {np.sum(hard != soft)} of {len(yte)}")

# %% [markdown]
# ## More trees, steadier answers

# %%
rows = []
for T in models.FOREST_SIZES:
    m = models.forest(T).fit(Xtr, ytr)
    rows.append({"trees": T, "train": m.score(Xtr, ytr), "test": m.score(Xte, yte)})
curve = pd.DataFrame(rows).set_index("trees")
ax = curve.plot(marker="o", logx=True, figsize=(6, 3), color=[P.GREEN, P.FG])
ax.set_ylabel("accuracy"); ax.set_title("Accuracy by number of trees");
curve.T

# %% [markdown]
# A single tree in the forest is worse than the tuned decision tree of chapter 3, since it
# sees a resample and random features. Averaging ten or more recovers that and passes the single tuned tree. Past
# that, extra trees cost time but change little: forests do not overfit by adding trees.
#
# **Try it:** set `max_features=None` in `models.forest` so every tree sees both features.
# The trees become more alike and the benefit of voting shrinks.
