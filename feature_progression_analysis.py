"""
Feature progression analysis (CRISP-DM Phase 4, modeling)

How do RMSE and R2 change as a linear regression uses 1, 2, ... p features,
when the features are ranked by different selection algorithms?

  1. Lasso (L1) ranking     - order in which features enter the Lasso path
  2. Forward SFS            - greedy forward selection by CV RMSE
  3. RFE                    - recursive feature elimination ranking
  4. ElasticNet ranking     - order of entry on the ElasticNet path (l1_ratio 0.5)
  5. Exhaustive search      - best subset of each size (only when p <= 14;
                              2^p - 1 subsets is infeasible beyond that)

Every algorithm only sees the TRAINING set and chooses by CV RMSE on the
folds it is given (grouped by country in main.py, so the same country is
never in both the fitting and the scoring part of a fold). The test set is
used only to report how each chosen subset generalises.

Called from main.py:
    progressions, comparison = run(Xtr, ytr, Xte, yte, folds, out_dir)
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, lasso_path, enet_path
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error

METHODS = {
    "lasso_l1_ranking": "Lasso (L1 Ranking)",
    "forward_sfs": "Forward SFS",
    "rfe": "RFE",
    "elasticnet_ranking": "ElasticNet Ranking",
    "exhaustive_search": "Exhaustive Search",
}
EXHAUSTIVE_MAX_P = 14


def cv_rmse(X, y, cols, folds):
    """Mean and SD of fold RMSE. `folds` = list of positional test indices."""
    Xc, yv, n = X[list(cols)].to_numpy(), y.to_numpy(), len(y)
    scores = []
    for test in folds:
        mask = np.ones(n, bool)
        mask[test] = False
        m = LinearRegression().fit(Xc[mask], yv[mask])
        scores.append(np.sqrt(mean_squared_error(yv[test], m.predict(Xc[test]))))
    return float(np.mean(scores)), float(np.std(scores))


def path_ranking(X, y, path_fn, **kw):
    """Earlier entry on the regularisation path = more important."""
    Xs = StandardScaler().fit_transform(X)
    yc = y.to_numpy() - y.mean()
    alphas, coefs, _ = path_fn(Xs, yc, alphas=300, **kw)   # alphas decreasing
    nonzero = np.abs(coefs) > 1e-8
    entry = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), len(alphas))
    final_size = np.abs(coefs[:, -1])
    order = sorted(range(len(entry)), key=lambda i: (entry[i], -final_size[i]))
    return [X.columns[i] for i in order]


def prefixes(rank):
    return [rank[:k] for k in range(1, len(rank) + 1)]


def subsets_lasso(X, y, folds):
    return prefixes(path_ranking(X, y, lasso_path))


def subsets_elasticnet(X, y, folds):
    return prefixes(path_ranking(X, y, enet_path, l1_ratio=0.5))


def subsets_rfe(X, y, folds):
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    ranking = RFE(LinearRegression(), n_features_to_select=1).fit(Xs, y).ranking_
    return prefixes([c for _, c in sorted(zip(ranking, X.columns))])


def subsets_forward_sfs(X, y, folds):
    chosen, out = [], []
    while len(chosen) < X.shape[1]:
        best = min((c for c in X.columns if c not in chosen),
                   key=lambda c: cv_rmse(X, y, chosen + [c], folds)[0])
        chosen.append(best)
        out.append(list(chosen))
    return out


def subsets_exhaustive(X, y, folds):
    cols = list(X.columns)
    if len(cols) > EXHAUSTIVE_MAX_P:
        return None
    return [list(min(combinations(cols, k), key=lambda s: cv_rmse(X, y, s, folds)[0]))
            for k in range(1, len(cols) + 1)]


SUBSET_FNS = {
    "lasso_l1_ranking": subsets_lasso,
    "forward_sfs": subsets_forward_sfs,
    "rfe": subsets_rfe,
    "elasticnet_ranking": subsets_elasticnet,
    "exhaustive_search": subsets_exhaustive,
}


def progression(Xtr, ytr, Xte, yte, folds, subsets):
    rows = []
    for feats in subsets:
        m = LinearRegression().fit(Xtr[feats], ytr)
        pred = m.predict(Xte[feats])
        mean, sd = cv_rmse(Xtr, ytr, feats, folds)
        rows.append({"Number of Features": len(feats), "Selected Features": ", ".join(feats),
                     "CV RMSE (train)": mean, "CV RMSE SD": sd,
                     "Test RMSE": np.sqrt(mean_squared_error(yte, pred)),
                     "Test R-squared": r2_score(yte, pred)})
    return pd.DataFrame(rows)


def one_sd_k(df):
    """Smallest k whose CV RMSE is within one fold-SD of the best."""
    best = df.loc[df["CV RMSE (train)"].idxmin()]
    ok = df[df["CV RMSE (train)"] <= best["CV RMSE (train)"] + best["CV RMSE SD"]]
    return int(best["Number of Features"]), int(ok["Number of Features"].min())


def plot_progression(df, title, path):
    k = df["Number of Features"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(k, df["Test RMSE"], "o-", label="Test")
    axes[0].plot(k, df["CV RMSE (train)"], "s--", alpha=0.7, label="CV (train, grouped)")
    axes[0].set_title(f"RMSE by Number of Features - {title}")
    axes[0].set_ylabel("RMSE (years)")
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


def run(Xtr, ytr, Xte, yte, folds, out_dir, verbose=True):
    out_dir = Path(out_dir)
    progressions = {}
    for key, title in METHODS.items():
        subsets = SUBSET_FNS[key](Xtr, ytr, folds)
        if subsets is None:
            if verbose:
                print(f"\n--- {title}: skipped ({Xtr.shape[1]} features -> "
                      f"{2 ** Xtr.shape[1] - 1:,} subsets; limit p <= {EXHAUSTIVE_MAX_P}) ---")
            continue
        df = progression(Xtr, ytr, Xte, yte, folds, subsets)
        df.to_csv(out_dir / f"{key}_progression.csv", index=False)
        plot_progression(df, title, out_dir / f"{key}_plot.png")
        progressions[key] = df
        if verbose:
            print(f"\n--- {title} ---")
            print(df.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 6))
    for (key, df), m in zip(progressions.items(), ["o", "s", "^", "D", "v"]):
        ax.plot(df["Number of Features"], df["Test RMSE"], marker=m, label=METHODS[key], alpha=0.85)
    ax.set_title("Test RMSE by Number of Features across Feature Selection Algorithms")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Test RMSE (years, lower is better)")
    ax.set_xticks(range(1, Xtr.shape[1] + 1))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "all_algorithms_comparison_plot.png", dpi=120)
    plt.close()

    comp = []
    for key, df in progressions.items():
        k_best, k_1sd = one_sd_k(df)
        r = df[df["Number of Features"] == k_1sd].iloc[0]
        comp.append({"Method": METHODS[key], "k (CV best)": k_best, "k (1-SD)": k_1sd,
                     "Selected Features (1-SD)": r["Selected Features"],
                     "CV RMSE (train)": r["CV RMSE (train)"], "Test RMSE": r["Test RMSE"],
                     "Test R-squared": r["Test R-squared"]})
    comparison = pd.DataFrame(comp)
    comparison.to_csv(out_dir / "feature_selection_comparison.csv", index=False)
    if verbose:
        print("\n--- 1-SD subset per method ---")
        print(comparison.round(4).to_string(index=False))
    return progressions, comparison
