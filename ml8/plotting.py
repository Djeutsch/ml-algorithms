"""Matplotlib helpers in the dark, high-contrast style of the video."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from . import data

BG = "#0d1117"
FG = "#e6edf3"
BLUE = "#4cb8e8"     # class 0 / linear models
RED = "#f06a4f"      # class 1
YELLOW = "#f2e341"   # fitted line / highlights / neural net
GREEN = "#7fb069"    # tree models
PURPLE = "#b48ce6"   # distance-based models

FAMILY_COLORS = {"Linear models": BLUE, "Tree models": GREEN, "Distance-based": PURPLE, "Neural networks": YELLOW}

SURFACE_CMAP = LinearSegmentedColormap.from_list("surface", ["#1f4e66", "#3a2d3a", "#7a2e22"])


def use_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": "#8b949e", "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": "#8b949e", "ytick.color": "#8b949e", "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 110, "font.size": 10, "axes.titleweight": "bold",
    })


def point_colors(labels):
    return np.where(np.asarray(labels) == 1, RED, BLUE)


def plot_surface(ax, model_or_proba, X=None, y=None, title=None, highlight=None):
    """Draw P(class 1) over the moons grid plus the points.

    model_or_proba: a fitted classifier with predict_proba, or an array of
    probabilities already evaluated on `data.grid()`.
    """
    xx, yy, pts = data.grid()
    if hasattr(model_or_proba, "predict_proba"):
        p = model_or_proba.predict_proba(pts)[:, 1]
    else:
        p = np.asarray(model_or_proba)
    ax.contourf(xx, yy, p.reshape(xx.shape), levels=np.linspace(0, 1, 11), cmap=SURFACE_CMAP, alpha=0.9)
    ax.contour(xx, yy, p.reshape(xx.shape), levels=[0.5], colors=[FG], linewidths=0.8, linestyles="--")
    if X is not None:
        ax.scatter(X[:, 0], X[:, 1], c=point_colors(y), s=16, edgecolors="none")
    if highlight is not None and len(highlight):
        ax.scatter(X[highlight, 0], X[highlight, 1], s=90, facecolors="none", edgecolors=YELLOW, linewidths=1.5)
    ax.set_xlim(*data.X1_RANGE)
    ax.set_ylim(*data.X2_RANGE)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax
