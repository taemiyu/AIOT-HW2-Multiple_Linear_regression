"""
50 Startups - Multiple Linear Regression with scikit-learn
Methodology: CRISP-DM (Cross-Industry Standard Process for Data Mining)

  Phase 1  Business Understanding
  Phase 2  Data Understanding
  Phase 3  Data Preparation
  Phase 4  Modeling
  Phase 5  Evaluation
  Phase 6  Deployment

Dataset: 50_Startups.csv (Kaggle)
Columns: R&D Spend, Administration, Marketing Spend, State, Profit

Outputs:
    crispdm_output/                 phase plots, trained model, final_report.json
    *_progression.csv / *_plot.png  5 feature-selection algorithms (Phase 4)
    model_data.json / model_data.js data for the web report (index.html)

Usage:
    python main.py                  # uses ./50_Startups.csv (downloads it if missing)
    python main.py path/to/file.csv
"""

import sys
import json
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save plots to files, no GUI needed
import matplotlib.pyplot as plt
import joblib
import requests

import feature_progression_analysis as fpa

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "crispdm_output"
OUT_DIR.mkdir(exist_ok=True)

TARGET = "Profit"
NUMERIC = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL = ["State"]


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
# PHASE 1: BUSINESS UNDERSTANDING
# =====================================================================
banner("PHASE 1: BUSINESS UNDERSTANDING")
# 1.1 Determine business objectives  /  1.3 Determine data mining goals
business = {
    "objective": "Help a venture capital fund estimate a startup's annual "
                 "profit from its spending profile and location.",
    "business_questions": [
        "Which spending category (R&D, Administration, Marketing) drives profit most?",
        "Does the state where the startup operates affect profit?",
        "How accurately can profit be predicted for a new startup?",
    ],
    "data_mining_goal": "Build a multiple linear regression model "
                        "Profit = f(R&D, Admin, Marketing, State).",
    "success_criteria": {
        "test_R2_min": 0.90,
        "cv_R2_min": 0.85,
        "MAPE_max_pct": 10.0,
    },
}
for k, v in business.items():
    print(f"{k}: {v}")

# 1.2 Assess situation: resources, constraints, risks, costs/benefits
situation = {
    "resources": "One public CSV (50 rows), Python + scikit-learn.",
    "assumptions": "Spend figures are annual and comparable across startups.",
    "constraints": "Tiny sample -> no deep models; must stay interpretable.",
    "risks": [
        "Small sample -> unstable estimates (mitigate: cross-validation).",
        "Zero spend values may be missing data rather than true zeros.",
    ],
    "cost_benefit": "Cheap to build; wrong predictions cost the fund money, "
                    "so the model supports, not replaces, analyst judgment.",
}
print("\n--- 1.2 Situation assessment ---")
for k, v in situation.items():
    print(f"{k}: {v}")

# 1.4 Produce project plan
project_plan = [
    "Phase 2: profile data, correlations, quality checks",
    "Phase 3: clean, encode State, split train/test",
    "Phase 4: OLS on every feature subset, choose by CV R2",
    "Phase 5: test-set metrics vs success criteria, decide go/no-go",
    "Phase 6: deploy only if approved; write final report",
]
print("\n--- 1.4 Project plan ---")
for step in project_plan:
    print(" -", step)


# =====================================================================
# PHASE 2: DATA UNDERSTANDING
# =====================================================================
banner("PHASE 2: DATA UNDERSTANDING")

# 2.1 Collect initial data: use the local CSV, or download it from a public
#     GitHub copy of the Kaggle file and verify it is the known version.
DATA_URL = ("https://raw.githubusercontent.com/krishnaik06/"
            "Multiple-Linear-Regression/master/50_Startups.csv")
DATA_SHA256 = "c6075ca5a7551243"   # prefix of the verified file's SHA-256

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE_DIR / "50_Startups.csv"
if not csv_path.exists():
    print(f"{csv_path.name} not found -> downloading {DATA_URL}")
    resp = requests.get(DATA_URL, timeout=30)
    if resp.status_code != 200:
        sys.exit(f"Download failed: HTTP {resp.status_code}")
    content = resp.content
    digest = hashlib.sha256(content).hexdigest()
    if not digest.startswith(DATA_SHA256):
        sys.exit(f"Downloaded file has unexpected SHA-256 {digest[:16]}; not using it.")
    csv_path.write_bytes(content)
    print(f"Downloaded and verified ({len(content)} bytes)")
df = pd.read_csv(csv_path)
print(f"Loaded {csv_path}: {df.shape[0]} rows x {df.shape[1]} columns")

# 2.2 Describe data
print("\n--- Head ---")
print(df.head())
print("\n--- Data types ---")
print(df.dtypes)
print("\n--- Descriptive statistics ---")
print(df.describe().T)
print("\n--- State frequency ---")
print(df["State"].value_counts())

# 2.3 Explore data
corr = df[NUMERIC + [TARGET]].corr()
print("\n--- Correlation matrix ---")
print(corr.round(3))
print("\n--- Correlation with Profit (sorted) ---")
print(corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3))
print("\n--- Mean profit by State ---")
print(df.groupby("State")[TARGET].agg(["mean", "std", "count"]).round(1))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, NUMERIC):
    ax.scatter(df[col], df[TARGET], alpha=0.7)
    ax.set_xlabel(col)
    ax.set_ylabel(TARGET)
    ax.set_title(f"{col} vs Profit (r={corr.loc[col, TARGET]:.2f})")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_scatter_features_vs_profit.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)))
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
fig.colorbar(im)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_correlation_heatmap.png", dpi=120)
plt.close()

df.boxplot(column=TARGET, by="State", figsize=(6, 4))
plt.suptitle("")
plt.title("Profit by State")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_profit_by_state.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(df[TARGET], bins=10, edgecolor="black")
axes[0].set_title("Profit distribution")
axes[1].boxplot(df[TARGET])
axes[1].set_title("Profit boxplot")
counts = df["State"].value_counts()
axes[2].bar(counts.index, counts.values)
axes[2].set_title("Startups per State")
plt.tight_layout()
plt.savefig(OUT_DIR / "05_eda_distributions.png", dpi=120)
plt.close()

# 2.4 Verify data quality
print("\n--- Data quality ---")
print("Missing values per column:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
zero_counts = (df[NUMERIC] == 0).sum()
print("Zero values per numeric column (possible missing data):\n", zero_counts)

q1, q3 = df[TARGET].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(df[TARGET] < q1 - 1.5 * iqr) | (df[TARGET] > q3 + 1.5 * iqr)]
print(f"Profit outliers (IQR rule): {len(outliers)}")
if len(outliers):
    print(outliers)


# =====================================================================
# PHASE 3: DATA PREPARATION
# =====================================================================
banner("PHASE 3: DATA PREPARATION")

# 3.1 Select data
data = df[NUMERIC + CATEGORICAL + [TARGET]].copy()

# 3.2 Clean data
before = len(data)
data = data.drop_duplicates().dropna()
print(f"Rows after removing duplicates/NaN: {len(data)} (removed {before - len(data)})")
# Zeros in spend columns are kept: a startup can genuinely spend $0 on a
# category, and dropping rows from only 50 samples would cost too much data.
data["State"] = data["State"].str.strip()
# Profit outliers found in 2.4 are kept: they are real companies, not
# entry errors, and removing them would bias the model toward the middle.

# 3.3 Construct data -> no derived features; MLR uses raw spend + State.
# 3.4 Integrate data -> single source, nothing to merge.

# 3.5 Format data: split X / y and train / test BEFORE any fitting
X = data[NUMERIC + CATEGORICAL]
y = data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Encoding + scaling inside a ColumnTransformer so it is fitted on the
# training set only (prevents data leakage). drop="first" avoids the
# dummy-variable trap for linear regression.
def make_preprocessor(num_cols, cat_cols):
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
        )
    return ColumnTransformer(transformers)


def make_pipeline(num_cols, cat_cols):
    return Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols)),
        ("model", LinearRegression()),
    ])


# 3.6 Feature selection analysis (training data only, so the test set stays
#     unseen). Five views of feature importance on the encoded features.
encoder = make_preprocessor(NUMERIC, CATEGORICAL).fit(X_train)
enc_names = [n.split("__", 1)[1] for n in encoder.get_feature_names_out()]
Xtr_enc = pd.DataFrame(encoder.transform(X_train), columns=enc_names, index=X_train.index)

fs = pd.DataFrame(index=enc_names)
fs["pearson_r"] = Xtr_enc.apply(lambda c: np.corrcoef(c, y_train)[0, 1])
fs["f_score"] = SelectKBest(f_regression, k="all").fit(Xtr_enc, y_train).scores_
fs["rfe_rank"] = RFE(LinearRegression(), n_features_to_select=1).fit(Xtr_enc, y_train).ranking_
lasso = LassoCV(cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), max_iter=50000).fit(Xtr_enc, y_train)
fs["lasso_coef"] = lasso.coef_
rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE).fit(Xtr_enc, y_train)
fs["rf_importance_pct"] = rf.feature_importances_ * 100
fs = fs.sort_values("f_score", ascending=False)
print("\n--- 3.6 Feature selection analysis (training data) ---")
print(fs.round(3).to_string())


# =====================================================================
# PHASE 4: MODELING
# =====================================================================
banner("PHASE 4: MODELING")

# 4.1 Select modeling technique: Ordinary Least Squares multiple linear
#     regression (sklearn LinearRegression).
#     Modeling assumptions: linearity (checked by scatter plots in 2.3),
#     low multicollinearity (VIF below), independent errors with constant
#     variance and roughly normal distribution (residual plots in 5.1).
def vif(frame):
    """Variance inflation factor: VIF_j = 1 / (1 - R2_j), where R2_j comes from
    regressing column j on the other columns."""
    out = {}
    for col in frame.columns:
        others = frame.drop(columns=col)
        r2 = LinearRegression().fit(others, frame[col]).score(others, frame[col])
        out[col] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out)

vifs = vif(X_train[NUMERIC])
print("--- 4.1 Multicollinearity check (VIF, training data) ---")
print(vifs.round(2))
print("VIF > 10 would mean coefficients are unreliable:",
      "none" if (vifs <= 10).all() else list(vifs[vifs > 10].index))

# 4.2 Generate test design: 80/20 hold-out + 5-fold CV on training set.
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 4.3 Build models: full model + every feature subset (exhaustive
#     selection is cheap with 4 candidate features).
candidates = NUMERIC + CATEGORICAL
results = []
for k in range(1, len(candidates) + 1):
    for subset in combinations(candidates, k):
        num = [c for c in subset if c in NUMERIC]
        cat = [c for c in subset if c in CATEGORICAL]
        pipe = make_pipeline(num, cat)
        scores = cross_val_score(pipe, X_train[list(subset)], y_train,
                                 cv=cv, scoring="r2")
        results.append({
            "features": subset,
            "n_features": len(subset),
            "cv_r2_mean": scores.mean(),
            "cv_r2_std": scores.std(),
        })

results_df = (pd.DataFrame(results)
              .sort_values("cv_r2_mean", ascending=False)
              .reset_index(drop=True))
pd.set_option("display.max_colwidth", 80)
print("--- Feature subsets ranked by 5-fold CV R2 (training data) ---")
print(results_df.round(4).to_string())

# 4.4 Assess model: choose best subset; prefer fewer features when CV R2
#     is within one standard deviation of the best (parsimony rule).
best = results_df.iloc[0]
within_1sd = results_df[results_df["cv_r2_mean"] >= best["cv_r2_mean"] - best["cv_r2_std"]]
chosen = within_1sd.sort_values(["n_features", "cv_r2_mean"],
                                ascending=[True, False]).iloc[0]
selected = list(chosen["features"])
print(f"\nBest CV subset : {list(best['features'])} (R2={best['cv_r2_mean']:.4f})")
print(f"Selected (1-SD parsimony rule): {selected} (R2={chosen['cv_r2_mean']:.4f})")

sel_num = [c for c in selected if c in NUMERIC]
sel_cat = [c for c in selected if c in CATEGORICAL]

# Feature progression: 5 selection algorithms x 1..5 encoded features, each
# chosen by CV RMSE on the training set (see feature_progression_analysis.py).
print("\n--- Feature progression analysis (5 algorithms) ---")
progressions, fs_comparison = fpa.run(out_dir=BASE_DIR, csv_path=csv_path)

full_model = make_pipeline(NUMERIC, CATEGORICAL).fit(X_train, y_train)
final_model = make_pipeline(sel_num, sel_cat).fit(X_train[selected], y_train)

# Coefficients of the full model (standardized numeric inputs, so
# magnitudes are directly comparable)
names = full_model.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(full_model.named_steps["model"].coef_, index=names)
print("\n--- Full model coefficients (numeric features standardized) ---")
print(coefs.sort_values(key=abs, ascending=False).round(1))
print(f"Intercept: {full_model.named_steps['model'].intercept_:.1f}")


# =====================================================================
# PHASE 5: EVALUATION
# =====================================================================
banner("PHASE 5: EVALUATION")

# 5.1 Evaluate results: hold-out metrics, residual checks, success criteria

def evaluate(model, X_te, y_te, n_features):
    pred = model.predict(X_te)
    n = len(y_te)
    r2 = r2_score(y_te, pred)
    p = n_features
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan
    return {
        "R2": r2,
        "Adj_R2": adj_r2,
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
        "MAPE_pct": np.mean(np.abs((y_te - pred) / y_te)) * 100,
    }, pred


n_full = len(full_model.named_steps["prep"].get_feature_names_out())
n_final = len(final_model.named_steps["prep"].get_feature_names_out())
full_metrics, _ = evaluate(full_model, X_test, y_test, n_full)
final_metrics, y_pred = evaluate(final_model, X_test[selected], y_test, n_final)

baseline_pred = np.full(len(y_test), y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

comparison = pd.DataFrame({"Full model": full_metrics,
                           "Selected model": final_metrics}).T
print("--- Hold-out test metrics ---")
print(comparison.round(4))
print(f"Baseline (predict train mean) RMSE: {baseline_rmse:,.0f}")

cv_final = cross_val_score(make_pipeline(sel_num, sel_cat), X[selected], y,
                           cv=cv, scoring="r2")
print(f"5-fold CV R2 on full data (selected model): "
      f"{cv_final.mean():.4f} +/- {cv_final.std():.4f}")

# Residual diagnostics (linear-regression assumptions)
residuals = y_test - y_pred
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(y_test, y_pred)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, "r--")
axes[0].set_xlabel("Actual Profit")
axes[0].set_ylabel("Predicted Profit")
axes[0].set_title("Actual vs Predicted")
axes[1].scatter(y_pred, residuals)
axes[1].axhline(0, color="r", linestyle="--")
axes[1].set_xlabel("Predicted Profit")
axes[1].set_ylabel("Residual")
axes[1].set_title("Residuals vs Fitted (homoscedasticity)")
axes[2].hist(residuals, bins=8, edgecolor="black")
axes[2].set_title("Residual distribution (normality)")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_residual_diagnostics.png", dpi=120)
plt.close()

# Review against business success criteria
crit = business["success_criteria"]
checks = {
    f"Test R2 >= {crit['test_R2_min']}": final_metrics["R2"] >= crit["test_R2_min"],
    f"CV R2 >= {crit['cv_R2_min']}": cv_final.mean() >= crit["cv_R2_min"],
    f"MAPE <= {crit['MAPE_max_pct']}%": final_metrics["MAPE_pct"] <= crit["MAPE_max_pct"],
}
print("\n--- Success criteria review ---")
for k, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {k}")
approved = all(checks.values())
print("\nDecision:", "APPROVE for deployment" if approved
      else "REVISIT earlier phases before deployment")

print("\n--- Answers to business questions ---")
top = coefs[[c for c in coefs.index if c.startswith("num__")]].abs().idxmax()
print(f"1. Strongest spending driver: {top.replace('num__', '')}")
print(f"2. State in selected model: {'yes' if 'State' in selected else 'no'} "
      f"-> State {'adds' if 'State' in selected else 'does not add'} "
      f"meaningful predictive power")
print(f"3. Typical prediction error (MAE): ${final_metrics['MAE']:,.0f}")

# 5.2 Review process: was anything overlooked or done out of order?
process_review = {
    "Test set touched only after model selection": True,
    "Scaler/encoder fitted inside pipeline (no leakage)": True,
    "Multicollinearity acceptable (all VIF <= 10)": bool((vifs <= 10).all()),
    "Model beats naive baseline RMSE": bool(final_metrics["RMSE"] < baseline_rmse),
}
print("\n--- 5.2 Process review ---")
for k, ok in process_review.items():
    print(f"[{'OK' if ok else 'ISSUE'}] {k}")
approved = approved and all(process_review.values())

# 5.3 Determine next steps
if approved:
    next_steps = "Proceed to Phase 6 (deployment)."
else:
    failed = [k for k, ok in {**checks, **process_review}.items() if not ok]
    next_steps = ("Do not deploy. Return to Phase 3/4 (e.g. log-transform, "
                  f"interaction terms, more data). Failed: {failed}")
print("\n--- 5.3 Next steps ---")
print(next_steps)


# =====================================================================
# PHASE 6: DEPLOYMENT
# =====================================================================
banner("PHASE 6: DEPLOYMENT")

# 6.1 Plan deployment: retrain the selected pipeline on ALL data and save it.
#     CRISP-DM only deploys a model that passed Phase 5.
model_path = OUT_DIR / "startup_profit_model.joblib"
if approved:
    production_model = make_pipeline(sel_num, sel_cat).fit(X[selected], y)
    joblib.dump({"pipeline": production_model, "features": selected}, model_path)
    print(f"Model saved to {model_path}")
else:
    print("Evaluation did not approve the model -> nothing deployed.")


def predict_profit(records, path=model_path):
    """Score new startups. `records` is a list of dicts or a DataFrame."""
    bundle = joblib.load(path)
    new = pd.DataFrame(records)
    return bundle["pipeline"].predict(new[bundle["features"]])


example = [
    {"R&D Spend": 160000, "Administration": 130000,
     "Marketing Spend": 300000, "State": "New York"},
    {"R&D Spend": 50000, "Administration": 120000,
     "Marketing Spend": 150000, "State": "Florida"},
]
example_preds = []
if approved:
    for rec, p in zip(example, predict_profit(example)):
        print(f"Predicted profit for {rec}: ${p:,.0f}")
        example_preds.append({**rec, "Predicted Profit": round(float(p), 2)})

# 6.2 Plan monitoring & maintenance
# 6.3 Produce final report
report = {
    "business_objective": business["objective"],
    "situation": situation,
    "vif": {k: round(float(v), 2) for k, v in vifs.items()},
    "feature_selection": fs.round(4).to_dict(orient="index"),
    "feature_selection_comparison": fs_comparison.round(4).to_dict(orient="records"),
    "selected_features": selected,
    "test_metrics_selected": {k: round(float(v), 4) for k, v in final_metrics.items()},
    "test_metrics_full": {k: round(float(v), 4) for k, v in full_metrics.items()},
    "cv_r2_mean": round(float(cv_final.mean()), 4),
    "cv_r2_std": round(float(cv_final.std()), 4),
    "success_criteria_passed": {k: bool(v) for k, v in checks.items()},
    "process_review": process_review,
    "approved": bool(approved),
    "next_steps": next_steps,
    "monitoring_plan": [
        "Log every prediction with its inputs; compare to actual profit when reported.",
        "Retrain quarterly or when rolling MAPE exceeds 15%.",
        "Alert if new inputs fall outside training ranges (data drift).",
        "Alert on unseen State values (encoder ignores them -> baseline state).",
    ],
}
report_path = OUT_DIR / "final_report.json"
report_path.write_text(json.dumps(report, indent=2))
print(f"Final report saved to {report_path}")

# 6.4 Review project
print("\nProject review: with only 50 samples, results have high variance; "
      "collect more data before high-stakes use.")


# 6.3 (web) Export everything the report page needs. index.html loads
#     model_data.js (works from file:// and static hosting such as Vercel);
#     model_data.json is the same data for other tools.
def r(x, nd=4):
    return round(float(x), nd)

hist_counts, hist_edges = np.histogram(df[TARGET], bins=8)
state_stats = df.groupby("State")[TARGET].agg(["mean", "std", "count"])
deployed = None
if approved:
    m = production_model.named_steps["model"]
    deployed = {
        "features": selected,
        "terms": dict(zip(production_model.named_steps["prep"].get_feature_names_out().tolist(),
                          [r(c, 2) for c in m.coef_])),
        "intercept": r(m.intercept_, 2),
    }
    # Raw-unit line for a single numeric feature (Profit = a + b * x)
    if sel_num and not sel_cat and len(sel_num) == 1:
        sc = production_model.named_steps["prep"].named_transformers_["num"]
        slope = m.coef_[0] / sc.scale_[0]
        deployed["line"] = {"x": sel_num[0], "slope": r(slope, 4),
                            "intercept": r(m.intercept_ - slope * sc.mean_[0], 2)}

model_data = {
    "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "settings": {"random_state": RANDOM_STATE, "test_size": TEST_SIZE,
                 "n_train": len(X_train), "n_test": len(X_test)},
    "business": business,
    "situation": situation,
    "project_plan": project_plan,
    "data": {
        "source_url": DATA_URL,
        "sha256_prefix": DATA_SHA256,
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "zeros": {k: int(v) for k, v in zero_counts.items()},
        "state_counts": df["State"].value_counts().to_dict(),
        "outliers": outliers.reset_index().to_dict(orient="records"),
        "corr": {"names": corr.columns.tolist(), "matrix": corr.round(3).values.tolist()},
        "hist": {"counts": hist_counts.tolist(), "edges": [r(e, 0) for e in hist_edges]},
        "scatter": df[[NUMERIC[0], TARGET]].round(0).values.tolist(),
        "state_profit": [{"state": s_, "mean": r(v["mean"], 0), "std": r(v["std"], 0),
                          "count": int(v["count"])} for s_, v in state_stats.iterrows()],
    },
    # Raw rows + the exact split / CV folds, so app.html can refit any
    # feature subset in the browser on the same rows as this script.
    "raw": data.reset_index(names="row").to_dict(orient="records"),
    "train_index": X_train.index.tolist(),
    "test_index": X_test.index.tolist(),
    "cv_folds": [X_train.index[test].tolist() for _, test in cv.split(X_train)],
    "vif": {k: r(v, 2) for k, v in vifs.items()},
    "feature_selection": [{"feature": k, **{c: r(v) for c, v in row.items()}}
                          for k, row in fs.iterrows()],
    "subset_search": [{"features": list(row["features"]), "cv_r2_mean": r(row["cv_r2_mean"]),
                       "cv_r2_std": r(row["cv_r2_std"])} for _, row in results_df.iterrows()],
    "best_subset": list(best["features"]),
    "selected_features": selected,
    "progressions": {key: {"title": fpa.METHODS[key],
                           "rows": [{"k": int(row["Number of Features"]),
                                     "features": row["Selected Features"].split(", "),
                                     "cv_rmse": r(row["CV RMSE (train)"], 2),
                                     "test_rmse": r(row["Test RMSE"], 2),
                                     "test_r2": r(row["Test R-squared"])}
                                    for _, row in dfp.iterrows()]}
                     for key, dfp in progressions.items()},
    "feature_selection_comparison": fs_comparison.round(4).to_dict(orient="records"),
    "metrics": {"selected": {k: r(v) for k, v in final_metrics.items()},
                "full": {k: r(v) for k, v in full_metrics.items()},
                "baseline_rmse": r(baseline_rmse, 2),
                "cv_r2_mean": r(cv_final.mean()), "cv_r2_std": r(cv_final.std())},
    "actual_vs_pred": [[r(a, 0), r(b, 0)] for a, b in zip(y_test, y_pred)],
    "success_criteria": [{"name": k, "passed": bool(v)} for k, v in checks.items()],
    "process_review": [{"name": k, "ok": bool(v)} for k, v in process_review.items()],
    "approved": bool(approved),
    "next_steps": next_steps,
    "deployed": deployed,
    "example_predictions": example_preds,
    "monitoring_plan": report["monitoring_plan"],
}
payload = json.dumps(model_data, indent=2, ensure_ascii=False)
(BASE_DIR / "model_data.json").write_text(payload, encoding="utf-8")
(BASE_DIR / "model_data.js").write_text(
    "// Generated by main.py - do not edit by hand.\nwindow.MODEL_DATA = " + payload + ";\n",
    encoding="utf-8")
print("Web report data saved to model_data.json / model_data.js")
