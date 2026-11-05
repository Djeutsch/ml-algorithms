# Eight ML Algorithms, Slowed Down

An interactive dashboard and a Jupyter Book that unpack an "8 algorithms · under 60 seconds"
video: linear regression, logistic regression, decision tree, random forest, gradient
boosting, k-nearest neighbours, k-means and a neural network. Each is run on small, seeded
sample data, and the dashboard and book always show the same numbers.

| | Live link | Source |
|---|---|---|
| 📘 Jupyter Book | https://__OWNER__.github.io/__REPO__/ | `myst.yml`, `intro.md`, `notebooks/`, `src/`, `ml8/` |
| 📊 Dashboard | https://__OWNER__.github.io/__REPO__/dashboard/ | `dashboard/` |

Both are deployed to GitHub Pages by `.github/workflows/deploy.yml` on every push to `main`.

---

## Use them locally

### Prerequisites

- Python 3.10+ (tested on 3.12 and 3.13)
- Node.js 18+ (Jupyter Book 2 uses it to build the site; it will offer to install it if missing)
- Git

```bash
git clone https://github.com/__OWNER__/__REPO__.git
cd __REPO__
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dashboard

The dashboard is a static page (HTML, JavaScript and CSV files, with d3 vendored in
`dashboard/vendor/`), so it needs no build and works offline. Browsers block `fetch` from
`file://` URLs, so serve it over HTTP rather than double-clicking the file:

```bash
python -m http.server 8000
# open http://localhost:8000/dashboard/
```

What you can do on it:

- **Overview:** test accuracy of every classifier on the shared two-moons sample, and a
  scorecard with each algorithm's headline metric.
- **Pick an algorithm** in the 4 × 2 grid (coloured by family, as in the video). Each panel
  shows the formula, a control and live statistics:
  - linear and logistic regression: a gradient-descent step slider with **Play**
  - decision tree: maximum depth, split lines and a table of every split's Gini
  - random forest: number of trees, plus nine individual member trees
  - gradient boosting: trees added so far, with misclassified points ringed
  - KNN: number of neighbours; **click the chart** to move the query point
  - k-means: number of clusters and iteration, with an elbow chart
  - neural network: training epoch, loss curve and a weight diagram
- Hover any chart for exact values. Each tab has its own link (for example `#knn`).
- It follows your system's light or dark setting.

### Jupyter Book

The notebooks in `notebooks/` are stored **with outputs**, so you can read or build them
straight away.

```bash
jupyter book start                 # live preview at http://localhost:3000
jupyter book build --html          # static site in _build/html
```

To open a chapter as a notebook instead: `jupyter lab notebooks/` (install `jupyterlab`
first).

To serve the built site together with the dashboard, the same way GitHub Pages does:

```bash
jupyter book build --html
cp -r dashboard _build/html/dashboard
python -m http.server 8000 -d _build/html
# book: http://localhost:8000/   dashboard: http://localhost:8000/dashboard/
```

### Regenerate everything from code

All data comes from the `ml8` package, so you can change a setting and rebuild both outputs:

```bash
python scripts/build_notebooks.py               # src/*.py → executed notebooks/*.ipynb
python scripts/export_dashboard_data.py         # → dashboard/data/*.csv
python scripts/build_standalone_dashboard.py    # dashboard/page.html → dashboard/index.html
```

Edit chapters in `src/` (Jupytext percent scripts); `notebooks/` is generated from them.
Model settings live in `ml8/models.py`, datasets in `ml8/data.py`.

---

## Repository layout

```
.
├── myst.yml, intro.md          Jupyter Book 2 (MyST) configuration and landing page
├── notebooks/                  executed chapters 00–09 (generated)
├── src/                        chapter sources (Jupytext percent format)
├── ml8/                        shared code: data, from-scratch formulas, models, plot style
├── scripts/                    build_notebooks, export_dashboard_data, build_standalone_dashboard
├── dashboard/
│   ├── index.html              standalone dashboard (generated)
│   ├── page.html               dashboard page, wrapped into index.html
│   ├── dash-runtime.js         tiny runtime that loads the CSVs and runs the page's drawings
│   ├── data/                   the 18 sample-data CSV files
│   └── vendor/                 d3 v7.9.0
└── .github/workflows/deploy.yml   builds the book and publishes both to GitHub Pages
```

## Deploying a fork

1. Fork or push this repository to GitHub.
2. **Settings → Pages → Build and deployment → Source: GitHub Actions.**
3. Push to `main` (or run the workflow manually from the **Actions** tab). The site appears at
   `https://<user>.github.io/<repo>/`.

The workflow sets `BASE_URL=/<repo>` so the book's links work under the repository path.

## About the data

Everything is synthetic sample data: scikit-learn's `make_moons` and `make_blobs`, plus two
small NumPy-generated 1-D sets, all seeded. The numbers illustrate how each algorithm
behaves; they are not benchmarks.

License: CC BY 4.0 for text and figures, MIT for code.
