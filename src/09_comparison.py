# %% [markdown]
# # Side by side
#
# All six classifiers on the same two-moons training points, scored on the same held-out
# points. These are the numbers behind the dashboard's scorecard.

# %% [markdown]
# ## Imports
#
# `ml8` is the book's own package; [The sample data](00_sample_data.ipynb) explains how it is
# imported. This chapter uses:
#
# | File | What this chapter uses |
# |---|---|
# | `ml8/data.py` | `data.moons_arrays`: two-moons train and test arrays (`X_train, y_train, X_test, y_test`) |
# | `ml8/models.py` | `models.boosting`: a scikit-learn `GradientBoostingClassifier` with the book's settings<br>`models.forest`: a scikit-learn `RandomForestClassifier` of a given size<br>`models.knn`: a scikit-learn `KNeighborsClassifier` for a given $k$<br>`models.logistic_2d`: a scikit-learn `LogisticRegression` for the two-moons data<br>`models.train_mlp`: trains a scikit-learn `MLPClassifier` one epoch at a time<br>`models.tree`: a scikit-learn `DecisionTreeClassifier` of a given depth |
# | `ml8/plotting.py` | `P.plot_surface`: draws a decision surface and the points |

# %%
# ml8 sits one folder above notebooks/, so put that folder on Python's search path
import sys
sys.path.insert(0, "..")

# Installed packages
import matplotlib.pyplot as plt
import pandas as pd

# The book's own modules
from ml8 import data              # ml8/data.py: the seeded sample datasets
from ml8 import models            # ml8/models.py: shared model settings
from ml8 import plotting as P     # ml8/plotting.py: colours and plot helpers

P.use_style()  # the dark style used by every figure
pd.set_option("display.precision", 4)

# %%
Xtr, ytr, Xte, yte = data.moons_arrays()
mlp, hist, surf = models.train_mlp(Xtr, ytr, X_test=Xte, y_test=yte)
fitted = {
    "Logistic regression": models.logistic_2d().fit(Xtr, ytr),
    "Decision tree (depth 4)": models.tree(4).fit(Xtr, ytr),
    "Random forest (100 trees)": models.forest(100).fit(Xtr, ytr),
    "Gradient boosting (60 trees)": models.boosting(60).fit(Xtr, ytr),
    "KNN (k = 15)": models.knn(15).fit(Xtr, ytr),
    "Neural network (400 epochs)": mlp,
}
fig, axes = plt.subplots(2, 3, figsize=(14, 6.4))
for ax, (name, m) in zip(axes.ravel(), fitted.items()):
    P.plot_surface(ax, m, Xtr, ytr, title=f"{name}\ntest {m.score(Xte, yte):.1%}")
plt.tight_layout()

# %%
table = pd.DataFrame({name: {"train accuracy": m.score(Xtr, ytr), "test accuracy": m.score(Xte, yte),
                             "train − test gap": m.score(Xtr, ytr) - m.score(Xte, yte)} for name, m in fitted.items()}).T
table.sort_values("test accuracy", ascending=False)

# %% [markdown]
# With 120 test points, one point is 0.83 percentage points, so gaps of one or two points
# between models are within noise. The clear signals are the linear model's ceiling and
# boosting's large train–test gap at 60 trees.
#
# ## When to reach for which
#
# | Algorithm | Family | Strength | Watch out for |
# |---|---|---|---|
# | Linear regression | Linear | Fast, interpretable coefficients | Only straight-line relationships |
# | Logistic regression | Linear | Calibrated probabilities, strong baseline | Linear boundary unless you engineer features |
# | Decision tree | Tree | Readable rules, no scaling needed | Overfits as it deepens |
# | Random forest | Tree | Robust default for tabular data | Larger, slower, less interpretable |
# | Gradient boosting | Tree | Often the most accurate on tabular data | Needs tuning of trees and learning rate |
# | KNN | Distance | No training, simple | Slow at prediction time; needs scaled features |
# | K-means | Distance | Quick clustering | You choose *k*; assumes round clusters |
# | Neural network | Neural | Learns any boundary given data | Needs data, tuning and compute |
