"""
50 Startups - Feature progression analysis (CRISP-DM Phase 4, modeling)

How do RMSE and R2 change as a linear regression uses 1, 2, ... 5 features,
when the features are chosen by five different selection algorithms?

  1. Lasso (L1) ranking     - order in which features enter the Lasso path
  2. Forward SFS            - sklearn SequentialFeatureSelector, forward
  3. RFE                    - recursive feature elimination ranking
  4. ElasticNet ranking     - order of entry on the ElasticNet path
  5. Exhaustive search      - best subset of each size

Every algorithm only sees the TRAINING set and chooses by 5-fold CV RMSE.
The test set is used only to report how each chosen subset generalises.

Outputs (next to this script):
  <method>_progression.csv / <method>_plot.png    for each of the 5 methods
  all_5_algorithms_comparison_plot.png
  feature_selection_comparison.csv / .png

Usage:
    python feature_progression_analysis.py           # standalone
    (also called from main.py during Phase 4)
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, lasso_path, enet_path
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Profit"

METHODS = {
    "lasso_l1_ranking": "Lasso (L1 Ranking)",
    "forward_sfs": "Forward SFS",
    "rfe": "RFE",
    "elasticnet_ranking": "ElasticNet Ranking",
    "exhaustive_search": "Exhaustive Search",
}


def load_and_split(csv_path=BASE_DIR / "50_Startups.csv"):
    """One-hot encode State (drop first -> no dummy trap), then 80/20 split.
    Same random_state as main.py, so both use identical train/test rows."""
    df = pd.read_csv(csv_path)
    X = pd.get_dummies(df.drop(columns=TARGET), columns=["State"],
                       drop_first=True, dtype=float)
    y = df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def make_cv():
    return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def cv_rmse(X_train, y_train, cols):
    scores = cross_val_score(LinearRegression(), X_train[list(cols)], y_train,
                             cv=make_cv(), scoring="neg_root_mean_squared_error")
    return -scores.mean()


def path_ranking(X_train, y_train, path_fn, **kw):
    """Rank features by how early (at which penalty) they enter the
    regularisation path: earlier entry = more important."""
    Xs = StandardScaler().fit_transform(X_train)
    yc = y_train.to_numpy() - y_train.mean()
    alphas, coefs, _ = path_fn(Xs, yc, alphas=300, **kw)   # alphas decreasing
    nonzero = np.abs(coefs) > 1e-8
    entry = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), len(alphas))
    final_size = np.abs(coefs[:, -1])
    order = sorted(range(len(entry)), key=lambda i: (entry[i], -final_size[i]))
    return [X_train.columns[i] for i in order]


def subsets_lasso(X_train, y_train):
    rank = path_ranking(X_train, y_train, lasso_path)
    return [rank[:k] for k in range(1, len(rank) + 1)]


def subsets_elasticnet(X_train, y_train):
    rank = path_ranking(X_train, y_train, enet_path, l1_ratio=0.5)
    return [rank[:k] for k in range(1, len(rank) + 1)]


def subsets_rfe(X_train, y_train):
    Xs = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)
    ranking = RFE(LinearRegression(), n_features_to_select=1).fit(Xs, y_train).ranking_
    rank = [c for _, c in sorted(zip(ranking, X_train.columns))]
    return [rank[:k] for k in range(1, len(rank) + 1)]


def subsets_forward_sfs(X_train, y_train):
    n = X_train.shape[1]
    out = []
    for k in range(1, n):
        sfs = SequentialFeatureSelector(
            LinearRegression(), n_features_to_select=k, direction="forward",
            scoring="neg_root_mean_squared_error", cv=make_cv(),
        ).fit(X_train, y_train)
        chosen = list(X_train.columns[sfs.get_support()])
        prev = out[-1] if out else []
        # keep the order in which features were added
        out.append(prev + [c for c in chosen if c not in prev])
    out.append(out[-1] + [c for c in X_train.columns if c not in out[-1]])
    return out


def subsets_exhaustive(X_train, y_train):
    cols = list(X_train.columns)
    return [list(min(combinations(cols, k), key=lambda s: cv_rmse(X_train, y_train, s)))
            for k in range(1, len(cols) + 1)]


SUBSET_FNS = {
    "lasso_l1_ranking": subsets_lasso,
    "forward_sfs": subsets_forward_sfs,
    "rfe": subsets_rfe,
    "elasticnet_ranking": subsets_elasticnet,
    "exhaustive_search": subsets_exhaustive,
}


def progression(X_train, X_test, y_train, y_test, subsets):
    rows = []
    for feats in subsets:
        model = LinearRegression().fit(X_train[feats], y_train)
        pred = model.predict(X_test[feats])
        rows.append({
            "Number of Features": len(feats),
            "Selected Features": ", ".join(feats),
            "CV RMSE (train)": cv_rmse(X_train, y_train, feats),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "Test R-squared": r2_score(y_test, pred),
        })
    return pd.DataFrame(rows)


def plot_progression(df, title, path):
    k = df["Number of Features"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(k, df["Test RMSE"], "o-", label="Test")
    axes[0].plot(k, df["CV RMSE (train)"], "s--", alpha=0.7, label="CV (train)")
    axes[0].set_title(f"RMSE by Number of Features - {title}")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()
    axes[1].plot(k, df["Test R-squared"], "o-", color="tab:green")
    axes[1].set_title(f"R-squared by Number of Features - {title}")
    axes[1].set_ylabel("Test R-squared")
    for ax in axes:
        ax.set_xlabel("Number of Features")
        ax.set_xticks(k)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def run(out_dir=BASE_DIR, csv_path=BASE_DIR / "50_Startups.csv", verbose=True):
    """Run all 5 methods, write CSVs/plots, return (progressions, comparison)."""
    out_dir = Path(out_dir)
    X_train, X_test, y_train, y_test = load_and_split(csv_path)

    progressions = {}
    for key, title in METHODS.items():
        df = progression(X_train, X_test, y_train, y_test,
                         SUBSET_FNS[key](X_train, y_train))
        df.to_csv(out_dir / f"{key}_progression.csv", index=False)
        plot_progression(df, title, out_dir / f"{key}_plot.png")
        progressions[key] = df
        if verbose:
            print(f"\n--- {title} ---")
            print(df.round(4).to_string(index=False))

    # All 5 on one chart (test RMSE)
    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^", "D", "v"]
    for (key, title), m in zip(METHODS.items(), markers):
        df = progressions[key]
        ax.plot(df["Number of Features"], df["Test RMSE"], marker=m, label=title, alpha=0.85)
    ax.set_title("RMSE Comparison across 5 Feature Selection Algorithms")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Test RMSE (Lower is Better)")
    ax.set_xticks(range(1, X_train.shape[1] + 1))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "all_5_algorithms_comparison_plot.png", dpi=120)
    plt.close()

    # Each method's best subset size, chosen by CV RMSE on training data
    comp = []
    for key, title in METHODS.items():
        df = progressions[key]
        best = df.loc[df["CV RMSE (train)"].idxmin()]
        comp.append({"Method": title, "Best Number of Features": int(best["Number of Features"]),
                     "Selected Features": best["Selected Features"],
                     "CV RMSE (train)": best["CV RMSE (train)"],
                     "Test RMSE": best["Test RMSE"], "Test R-squared": best["Test R-squared"]})
    comparison = pd.DataFrame(comp)
    comparison.to_csv(out_dir / "feature_selection_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(comparison["Method"], comparison["Test RMSE"], color="tab:blue")
    axes[0].set_title("Test RMSE of each method's best subset")
    axes[0].set_xlim(comparison["Test RMSE"].min() * 0.9, comparison["Test RMSE"].max() * 1.03)
    axes[1].barh(comparison["Method"], comparison["Test R-squared"], color="tab:green")
    axes[1].set_title("Test R-squared of each method's best subset")
    axes[1].set_xlim(comparison["Test R-squared"].min() - 0.02, 1)
    for ax, col in zip(axes, ["Test RMSE", "Test R-squared"]):
        ax.invert_yaxis()
        for i, v in enumerate(comparison[col]):
            ax.text(v, i, f" {v:,.0f}" if col == "Test RMSE" else f" {v:.4f}", va="center")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_selection_comparison.png", dpi=120)
    plt.close()

    if verbose:
        print("\n--- Best subset per method (chosen by CV RMSE on training data) ---")
        print(comparison.round(4).to_string(index=False))
    return progressions, comparison


if __name__ == "__main__":
    run()
