"""Generate ten candidate logos for the book and install the chosen one.

Every logo is drawn with matplotlib in the book's own palette (ml8.plotting). The
square version sits on a dark rounded tile. The chosen logo is also built into a
wide lockup (mark on the left, stacked title on the right) with no background, in
one version per book theme, for the top of the book's sidebar.

Usage:
    python scripts/generate_logos.py                 # draw all ten + contact sheet
    python scripts/generate_logos.py --use moons8    # also install one as the site logo
    python scripts/generate_logos.py --list          # show the candidates

Outputs:
    logos/candidates/NN_<name>.svg / .png   every candidate (512 px PNG)
    logos/contact_sheet.png                 all ten side by side, for picking
    images/logo.svg, images/logo.png        the chosen logo, square tile (--use)
    images/logo-wide-light.svg / -dark.svg  wide lockups for the book (--use)
    images/logo-wide-light.png / -dark.png  the same, for the README (--use)
    images/favicon.ico                      16/32/48 px icon  (--use)
    dashboard/favicon.ico                   same icon for the dashboard (--use)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from PIL import Image  # noqa: E402  (installed with matplotlib)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ml8 import data  # noqa: E402
from ml8.plotting import BG, BLUE, FG, GREEN, PURPLE, RED, YELLOW  # noqa: E402

CANDIDATES_DIR = ROOT / "logos" / "candidates"
FAMILIES = [BLUE, GREEN, PURPLE, YELLOW]  # same order as the video's button grid
MUTED = "#8b949e"
plt.rcParams["svg.fonttype"] = "path"  # text becomes outlines: no font needed to view the SVG


@dataclass(frozen=True)
class Theme:
    ink: str             # main strokes and text
    paper: str           # what the logo sits on
    accent: str = YELLOW  # top line of the lockup's title
    tile: bool = False   # draw the dark rounded tile


ON_TILE = Theme(ink=FG, paper=BG, tile=True)
ON_LIGHT = Theme(ink=BG, paper="#ffffff", accent="#a87f00")
ON_DARK = Theme(ink=FG, paper=BG)


# ---------------------------------------------------------------- helpers
def tile(ax, t: Theme):
    """Dark rounded-square background filling the unit square (if the theme has one)."""
    if t.tile:
        ax.add_patch(patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.18",
                                            fc=t.paper, ec="none", zorder=0))


def text(ax, x, y, s, size, color, weight="bold", **kw):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha="center", va="center",
            family="DejaVu Sans", **kw)


def moons_xy(n=160):
    """The book's real two-moons sample, rescaled into [0, 1]^2."""
    df = data.moons_data().iloc[:n]
    X = df[["x1", "x2"]].to_numpy()
    X = (X - X.min(0)) / (X.max(0) - X.min(0))
    return X, df.label.to_numpy()


# ---------------------------------------------------------------- the ten logos
def grid8(ax, t: Theme):
    """The video's 4 x 2 button grid, coloured by family."""
    tile(ax, t)
    w, h, gx, gy = 0.17, 0.25, 0.045, 0.07
    x0 = (1 - (4 * w + 3 * gx)) / 2
    y0 = (1 - (2 * h + gy)) / 2
    for row in range(2):
        for col in range(4):
            ax.add_patch(patches.FancyBboxPatch((x0 + col * (w + gx), y0 + (1 - row) * (h + gy)), w, h,
                                                boxstyle="round,pad=0,rounding_size=0.035",
                                                fc=FAMILIES[col], ec="none", alpha=0.95 if row == 0 else 0.6))


def moons8(ax, t: Theme):
    """A figure-eight traced through the two-moons data: the book in one mark."""
    tile(ax, t)
    u = np.linspace(0, 2 * np.pi, 400)
    x, y = 0.5 + 0.2 * np.sin(2 * u), 0.5 + 0.33 * np.sin(u)
    X, lab = moons_xy(220)
    rng = np.random.default_rng(data.SEED)
    keep = rng.random(len(X)) < 0.75
    # squeeze the moons into the eight's two loops
    px = 0.2 + 0.6 * X[keep, 0]
    py = 0.18 + 0.64 * X[keep, 1]
    ax.scatter(px, py, s=9, c=np.where(lab[keep] == 1, RED, BLUE), alpha=0.55, lw=0, zorder=1)
    ax.plot(x, y, color=YELLOW, lw=9, solid_capstyle="round", zorder=2)


def neural8(ax, t: Theme):
    """A 2-4-2 network whose nodes sit on an '8'."""
    tile(ax, t)
    layers = [[0.35, 0.65], [0.2, 0.4, 0.6, 0.8], [0.35, 0.65]]
    xs = [0.22, 0.5, 0.78]
    for (xa, la), (xb, lb) in zip(zip(xs, layers), zip(xs[1:], layers[1:])):
        for ya in la:
            for yb in lb:
                ax.plot([xa, xb], [ya, yb], color=MUTED, lw=1.6, alpha=0.6, zorder=1)
    colors = [BLUE, YELLOW, RED]
    for x, ys, c in zip(xs, layers, colors):
        for yy in ys:
            ax.add_patch(patches.Circle((x, yy), 0.055, fc=c, ec=t.paper, lw=3, zorder=2))


def tree8(ax, t: Theme):
    """A depth-3 decision tree with eight leaves in the four family colours."""
    tile(ax, t)
    nodes = {(0, 0): (0.5, 0.8)}
    for d in (1, 2, 3):
        for i in range(2 ** d):
            nodes[(d, i)] = ((i + 0.5) / 2 ** d * 0.84 + 0.08, 0.8 - d * 0.2)
    for (d, i), (x, y) in nodes.items():
        if d:
            px, py = nodes[(d - 1, i // 2)]
            ax.plot([px, x], [py, y], color=MUTED, lw=3, zorder=1)
    for (d, i), (x, y) in nodes.items():
        c = FAMILIES[i // 2] if d == 3 else t.ink
        ax.add_patch(patches.Circle((x, y), 0.045 if d == 3 else 0.04, fc=c, ec=t.paper, lw=2, zorder=2))


def stopwatch(ax, t: Theme):
    """'Slowed down': a stopwatch with an 8 on the dial."""
    tile(ax, t)
    ax.add_patch(patches.Circle((0.5, 0.45), 0.3, fc="none", ec=t.ink, lw=7))
    ax.add_patch(patches.FancyBboxPatch((0.44, 0.8), 0.12, 0.06, boxstyle="round,pad=0,rounding_size=0.02", fc=t.ink))
    for k, c in enumerate(FAMILIES):
        ax.add_patch(patches.Wedge((0.5, 0.45), 0.24, 90 - 90 * (k + 1), 90 - 90 * k, width=0.035, fc=c))
    text(ax, 0.5, 0.45, "8", 70, t.ink)


def monogram(ax, t: Theme):
    """'ml8' wordmark with a family-coloured underline."""
    tile(ax, t)
    text(ax, 0.5, 0.56, "ml8", 66, t.ink)
    for k, c in enumerate(FAMILIES):
        ax.add_patch(patches.Rectangle((0.2 + k * 0.15, 0.3), 0.13, 0.05, fc=c))


def boundary(ax, t: Theme):
    """Two-moons scatter split by a curved dashed decision boundary."""
    tile(ax, t)
    X, lab = moons_xy(200)
    px, py = 0.14 + 0.72 * X[:, 0], 0.18 + 0.64 * X[:, 1]
    ax.scatter(px, py, s=22, c=np.where(lab == 1, RED, BLUE), lw=0, zorder=1)
    u = np.linspace(0.12, 0.88, 200)
    ax.plot(u, 0.5 + 0.17 * np.sin((u - 0.12) / 0.76 * 2 * np.pi), color=YELLOW, lw=5, ls=(0, (3, 1.5)), zorder=2)


def hexbadge(ax, t: Theme):
    """Hexagon badge split into the four family colours with an 8."""
    tile(ax, t)
    ang = np.deg2rad(np.arange(6) * 60 + 30)
    hx, hy = 0.5 + 0.38 * np.cos(ang), 0.5 + 0.38 * np.sin(ang)
    for k, c in enumerate(FAMILIES):
        ax.add_patch(patches.Wedge((0.5, 0.5), 0.5, 90 * k, 90 * (k + 1), fc=c, ec="none",
                                   clip_path=None, alpha=0.9))
    hexagon = patches.Polygon(np.c_[hx, hy], closed=True, fc="none", ec=t.paper, lw=0)
    ax.add_patch(hexagon)
    for p in ax.patches[1:5]:
        p.set_clip_path(hexagon)
    ax.add_patch(patches.Circle((0.5, 0.5), 0.2, fc=t.paper, ec="none"))
    text(ax, 0.5, 0.5, "8", 54, t.ink)


def kmeans(ax, t: Theme):
    """Four clusters (one per family) pulled towards their centroids."""
    tile(ax, t)
    rng = np.random.default_rng(data.SEED)
    centres = [(0.3, 0.68), (0.7, 0.68), (0.3, 0.32), (0.7, 0.32)]
    for (cx, cy), c in zip(centres, FAMILIES):
        pts = rng.normal((cx, cy), 0.06, size=(14, 2))
        for x, y in pts:
            ax.plot([cx, x], [cy, y], color=c, lw=1, alpha=0.35, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=26, color=c, lw=0, zorder=2)
        ax.scatter([cx], [cy], s=260, marker="P", color=t.ink, edgecolors=t.paper, lw=2, zorder=3)


def curves(ax, t: Theme):
    """Gradient descent: a loss curve settling into a sigmoid, with 8 steps."""
    tile(ax, t)
    u = np.linspace(-6, 6, 200)
    sig = 1 / (1 + np.exp(-u))
    x = 0.15 + 0.7 * (u + 6) / 12
    ax.plot(x, 0.2 + 0.6 * sig, color=YELLOW, lw=7, solid_capstyle="round", zorder=2)
    steps = np.linspace(-5.2, 5.2, 8)
    for k, s in enumerate(steps):
        ax.add_patch(patches.Circle((0.15 + 0.7 * (s + 6) / 12, 0.2 + 0.6 / (1 + np.exp(-s))), 0.032,
                                    fc=FAMILIES[k // 2], ec=t.paper, lw=2, zorder=3))


LOGOS = {f.__name__: f for f in (moons8, grid8, neural8, tree8, stopwatch, monogram, boundary, hexbadge, kmeans, curves)}


# ---------------------------------------------------------------- output
def new_ax(size_in=4.0):
    fig = plt.figure(figsize=(size_in, size_in))
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def render(name: str, stem: Path, png_px: int = 512):
    fig, ax = new_ax()
    LOGOS[name](ax, ON_TILE)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=png_px / 4, transparent=True)
    plt.close(fig)


def render_wide(name: str, stem: Path, t: Theme, png_height: int = 240):
    """Wide lockup: the mark on the left, the stacked book title on the right."""
    width, height = 14.0, 4.0  # height matches render(), so font sizes and line widths in the mark hold
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_alpha(0)
    mark = fig.add_axes([0, 0, height / width, 1])
    mark.set_xlim(0, 1); mark.set_ylim(0, 1); mark.set_aspect("equal"); mark.axis("off")
    LOGOS[name](mark, t)
    left = (height + 0.24) / width
    for y, s, size, color, weight in ((0.80, "EIGHT ML", 66, t.accent, "bold"),
                                      (0.45, "Algorithms", 90, t.ink, "normal"),
                                      (0.13, "Slowed Down", 42, t.ink, "normal")):
        fig.text(left, y, s, fontsize=size, color=color, weight=weight, ha="left", va="center", family="DejaVu Sans")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), transparent=True)
    fig.savefig(stem.with_suffix(".png"), dpi=png_height / height, transparent=True)
    plt.close(fig)


def render_favicon(dest: list[Path]):
    """A simplified mark that survives 16 px: thick family ring and a large 8, no fine detail."""
    fig, ax = new_ax()
    tile(ax, ON_TILE)
    for k, c in enumerate(FAMILIES):
        ax.add_patch(patches.Wedge((0.5, 0.5), 0.45, 90 - 90 * (k + 1), 90 - 90 * k, width=0.11, fc=c))
    text(ax, 0.5, 0.49, "8", 150, FG)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=64, transparent=True)
    plt.close(fig)
    for path in dest:
        Image.open(buffer).save(path, sizes=[(16, 16), (32, 32), (48, 48)])


def contact_sheet(paths: list[Path], out: Path):
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.patch.set_facecolor("#f4f4f2")
    for ax, (k, path) in zip(axes.flat, enumerate(paths, 1)):
        ax.imshow(plt.imread(path)); ax.axis("off")
        ax.set_title(f"{k:02d} · {path.stem.split('_', 1)[1]}", fontsize=12, color="#1b1d22")
    fig.tight_layout()
    fig.savefig(out, dpi=90, facecolor=fig.get_facecolor())
    plt.close(fig)


def install(name: str):
    images = ROOT / "images"
    render(name, images / "logo")
    render_wide(name, images / "logo-wide-light", ON_LIGHT)
    render_wide(name, images / "logo-wide-dark", ON_DARK)
    render_favicon([images / "favicon.ico", ROOT / "dashboard" / "favicon.ico"])
    shutil.copy(images / "logo.png", ROOT / "dashboard" / "logo.png")
    print(f"installed '{name}' -> images/logo*.svg|png, images/favicon.ico, dashboard/favicon.ico, dashboard/logo.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--use", choices=LOGOS, help="install this candidate as the book and dashboard logo")
    ap.add_argument("--list", action="store_true", help="list the candidates and exit")
    args = ap.parse_args()

    if args.list:
        for k, (name, f) in enumerate(LOGOS.items(), 1):
            print(f"{k:02d}  {name:<10} {f.__doc__}")
        return

    paths = []
    for k, name in enumerate(LOGOS, 1):
        stem = CANDIDATES_DIR / f"{k:02d}_{name}"
        render(name, stem)
        paths.append(stem.with_suffix(".png"))
    contact_sheet(paths, ROOT / "logos" / "contact_sheet.png")
    print(f"wrote {len(paths)} candidates to logos/candidates/ and logos/contact_sheet.png")

    if args.use:
        install(args.use)


if __name__ == "__main__":
    main()
