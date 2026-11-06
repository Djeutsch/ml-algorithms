# Eight ML Algorithms, Slowed Down

From predicting house prices to filtering spam, recommending films and recognising faces, eight algorithms do much of the everyday work of machine learning.
- Linear regression predicts a house’s price from its size.
- Logistic regression decides whether an email is spam. 
- Decision trees, random forests and gradient boosting learn chains of yes-or-no questions, the approach behind many credit-scoring and clinical risk models.
- K-nearest neighbours recommends what similar people liked. 
- K-means sorts customers, or lab samples, into groups without being told what the groups are. 
- Neural networks learn the curved, layered patterns behind image and speech recognition.

Here each one gets one formula and runs on small sample data (one toy dataset), so you can watch it learn one step at a time with a few
frames of animation, while a grid of eight buttons lights up in four colour families.

```{tip}
Prefer to explore by clicking? The [interactive dashboard](https://djeutsch.github.io/ml-algorithms/dashboard/) runs all eight algorithms on the same sample data, with sliders, Play buttons and click-to-query.
```

This book replays each frame slowly. Every chapter states the video's formula,
implements it from scratch in a few lines of NumPy, checks the result against
scikit-learn, and redraws the video's picture from real sample data.

| # | Algorithm | Video formula | Family | Task |
|---|---|---|---|---|
| 1 | [Linear regression](notebooks/01_linear_regression.ipynb) | $\hat y = wx + b$ | Linear models | Regression |
| 2 | [Logistic regression](notebooks/02_logistic_regression.ipynb) | $p = \sigma(wx + b)$ | Linear models | Classification |
| 3 | [Decision tree](notebooks/03_decision_tree.ipynb) | $G = 1 - \sum_k p_k^2$ | Tree models | Classification |
| 4 | [Random forest](notebooks/04_random_forest.ipynb) | $\hat y = \operatorname{mode}(h_1(x), \dots, h_T(x))$ | Tree models | Classification |
| 5 | [Gradient boosting](notebooks/05_gradient_boosting.ipynb) | $F_m(x) = F_{m-1}(x) + \eta\, h_m(x)$ | Tree models | Classification |
| 6 | [K-nearest neighbours](notebooks/06_knn.ipynb) | $\hat y = \operatorname{mode}\{y_i : i \in N_5(x)\}$ | Distance-based | Classification |
| 7 | [K-means](notebooks/07_kmeans.ipynb) | $\mu_j = \frac{1}{\lvert C_j\rvert}\sum_{i\in C_j} x_i$ | Distance-based | Clustering |
| 8 | [Neural network](notebooks/08_neural_network.ipynb) | $a = \sigma(Wx + b)$ | Neural networks | Classification |

## The four families

The video colours its buttons by family, and the grouping is a useful map:

- **Linear models** (blue): a weighted sum of the inputs, optionally squashed. Fast and
  interpretable, but the boundary is a straight line.
- **Tree models** (green): axis-aligned yes/no questions. One tree overfits; forests
  average many, boosting adds them in sequence.
- **Distance-based** (purple): predictions come from which points are close. KNN uses
  labels; k-means invents its own groups.
- **Neural networks** (yellow): stacked linear-then-squash layers that can learn curved
  boundaries.

## How the book is organised

- [The sample data](notebooks/00_sample_data.ipynb) introduces the four seeded datasets.
- Chapters 1–8 follow the video's order.
- [Side by side](notebooks/09_comparison.ipynb) compares every classifier on the same data.

All shared code lives in the `ml8` package next to this page: `ml8/data.py` (datasets),
`ml8/scratch.py` (from-scratch formulas), `ml8/models.py` (model settings) and
`ml8/plotting.py` (the video's dark style). Every chapter opens with an **Imports** section
showing what it loads, and [The sample data](notebooks/00_sample_data.ipynb) explains how
the imports work. `scripts/export_dashboard_data.py` writes the
CSV files behind the [companion dashboard](https://djeutsch.github.io/ml-algorithms/dashboard/) from the same code, so the two always agree.

## Running it

```bash
pip install -r requirements.txt
python scripts/build_notebooks.py      # execute every chapter into notebooks/
jupyter book start                     # live preview
jupyter book build --html              # static site in _build/html
python -m http.server 8000             # dashboard at http://localhost:8000/dashboard/
```

The repository README has the full local setup.
