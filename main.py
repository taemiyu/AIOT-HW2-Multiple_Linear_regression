"""
WHO Life Expectancy - Multiple Linear Regression
Methodology: CRISP-DM (Cross-Industry Standard Process for Data Mining)

  Phase 1  Business Understanding
  Phase 2  Data Understanding
  Phase 3  Data Preparation
  Phase 4  Modeling
  Phase 5  Evaluation
  Phase 6  Deployment

Dataset: "Life Expectancy (WHO)" on Kaggle, by KumarRajarshi
         https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
         2,938 rows (193 countries x 2000-2015), 22 columns, 20 candidate
         features + Country + target "Life expectancy".

Usage:
    python main.py            # expects datasets/LifeExpectancy_WHO.csv

Outputs:
    crispdm_output/                  phase plots, model, final_report.json, run log
    *_progression.csv / *_plot.png   feature-selection algorithms (Phase 4)
    datasets/who_prepared.csv        cleaned + imputed data used by the web app
    model_data.json / model_data.js  data for the web report (index.html, app.html)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save plots to files, no GUI needed
import matplotlib.pyplot as plt
import joblib
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import feature_progression_analysis as fpa

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "crispdm_output"
OUT_DIR.mkdir(exist_ok=True)
DATA_PATH = BASE_DIR / "datasets" / "LifeExpectancy_WHO.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who"

RANDOM_STATE = 42
TEST_SIZE = 0.2
ALPHA = 0.05
TARGET = "Life expectancy"
GROUP = "Country"


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=120)
    plt.close()


# =====================================================================
# PHASE 1: BUSINESS UNDERSTANDING
# =====================================================================
banner("PHASE 1: BUSINESS UNDERSTANDING")

# 1.1 Business objectives  /  1.3 Data mining goals
business = {
    "objective": "Help health policy analysts estimate a country's life expectancy "
                 "from health, economic and social indicators, and identify which "
                 "indicators matter most.",
    "business_questions": [
        "Which indicators are most strongly associated with life expectancy?",
        "How accurately can life expectancy be predicted for a country the model has never seen?",
        "How uncertain is a single prediction (prediction interval)?",
    ],
    "data_mining_goal": "Multiple linear regression: Life expectancy = f(indicators), "
                        "with feature selection, evaluated on unseen countries.",
    # Set before modeling; judged on countries held out from training.
    "success_criteria": {"test_R2_min": 0.80, "test_RMSE_max": 4.5, "cv_R2_min": 0.75},
}
for k, v in business.items():
    print(f"{k}: {v}")

# 1.2 Assess situation
situation = {
    "resources": "One public Kaggle CSV (2,938 rows, 22 columns), Python with "
                 "scikit-learn and statsmodels.",
    "assumptions": "Indicators are comparable across countries and years.",
    "constraints": "The model must stay interpretable (linear) for policy discussion.",
    "risks": [
        "Many missing values (44% of rows have at least one).",
        "Implausible values (e.g. average BMI above 50) indicate data-entry errors.",
        "Each country appears up to 16 times; a random split would leak the same "
        "country into training and testing.",
    ],
    "cost_benefit": "Cheap to build; useful for prioritising health spending, but "
                    "associations are not causal effects.",
}
print("\n--- 1.2 Situation assessment ---")
for k, v in situation.items():
    print(f"{k}: {v}")

# 1.4 Project plan
project_plan = [
    "Phase 2: profile data, missing values, implausible values, correlations",
    "Phase 3: clean, log-transform skewed variables, split by country, impute (same-country median, then training median)",
    "Phase 4: VIF, feature selection (Lasso, Forward SFS, RFE, ElasticNet), OLS / Ridge / Lasso",
    "Phase 5: test metrics vs success criteria, assumption checks, prediction intervals",
    "Phase 6: deploy only if approved; save model and report",
]
print("\n--- 1.4 Project plan ---")
for step in project_plan:
    print(" -", step)


# =====================================================================
# PHASE 2: DATA UNDERSTANDING
# =====================================================================
banner("PHASE 2: DATA UNDERSTANDING")

# 2.1 Collect initial data
if not DATA_PATH.exists():
    sys.exit(f"{DATA_PATH} not found. Download it from {KAGGLE_URL} "
             "and save 'Life Expectancy Data.csv' as datasets/LifeExpectancy_WHO.csv.")
raw = pd.read_csv(DATA_PATH)
raw.columns = [" ".join(c.split()) for c in raw.columns]   # strip stray spaces in names
print(f"Loaded {DATA_PATH.name}: {raw.shape[0]} rows x {raw.shape[1]} columns")
print(f"Source: {KAGGLE_URL}")

FEATURES_RAW = [c for c in raw.columns if c not in (GROUP, TARGET)]
print(f"Candidate features ({len(FEATURES_RAW)}): {FEATURES_RAW}")

# 2.2 Describe data
print("\n--- Head ---")
print(raw.head())
print("\n--- Descriptive statistics ---")
print(raw.describe().T.round(2))
print(f"\nCountries: {raw[GROUP].nunique()}, years {raw['Year'].min()}-{raw['Year'].max()}")
print("Status:", raw["Status"].value_counts().to_dict())

# 2.3 Explore data
numeric_raw = raw.select_dtypes("number")
corr_target = numeric_raw.corr()[TARGET].drop(TARGET).sort_values()
print("\n--- Correlation with life expectancy ---")
print(corr_target.round(3))

fig, ax = plt.subplots(figsize=(8, 7))
colors = ["tab:red" if v < 0 else "tab:blue" for v in corr_target]
ax.barh(corr_target.index, corr_target.values, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Pearson correlation with Life expectancy")
savefig("03_correlation_with_target.png")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(raw[TARGET].dropna(), bins=30, edgecolor="black")
ax.set_title("Life expectancy distribution")
ax.set_xlabel("Years")
savefig("02_target_distribution.png")

top6 = corr_target.abs().sort_values(ascending=False).index[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), top6):
    ax.scatter(raw[col], raw[TARGET], s=6, alpha=0.4)
    ax.set_xlabel(col)
    ax.set_ylabel(TARGET)
    ax.set_title(f"{col} (r={corr_target[col]:.2f})")
savefig("05_top_features_scatter.png")

# 2.4 Verify data quality
missing = raw.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print("\n--- Missing values ---")
print(missing)
print(f"Rows with any missing value: {int(raw.isna().any(axis=1).sum())} "
      f"({raw.isna().any(axis=1).mean() * 100:.1f}%)")
print("Duplicate rows:", int(raw.duplicated().sum()))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(missing.index[::-1], missing.values[::-1], color="tab:orange")
ax.set_title("Missing values per column")
savefig("01_missing_values.png")

quality_rules = [
    ("BMI below 10 or above 50 (not a plausible national average)",
     (raw["BMI"] < 10) | (raw["BMI"] > 50), "Drop the BMI column (too many wrong values to impute)"),
    ("Schooling = 0 years", raw["Schooling"] == 0, "Treat as missing, impute"),
    ("Income composition of resources = 0", raw["Income composition of resources"] == 0,
     "Treat as missing, impute"),
    ("Percentage expenditure = 0", raw["percentage expenditure"] == 0, "Treat as missing, impute"),
    ("Life expectancy missing (target)", raw[TARGET].isna(), "Drop row"),
]
quality = [{"issue": name, "count": int(mask.sum()), "pct": round(float(mask.mean() * 100), 1),
            "action": action} for name, mask, action in quality_rules]
print("\n--- Implausible / problematic values ---")
for q in quality:
    print(f"{q['issue']:60s} {q['count']:5d} ({q['pct']:4.1f}%) -> {q['action']}")

skewed = ["Population", "infant deaths", "under-five deaths", "Measles", "HIV/AIDS",
          "percentage expenditure", "GDP"]
skew_before = raw[skewed].skew().round(2)
print("\n--- Skewness (will be log-transformed) ---")
print(skew_before)


# =====================================================================
# PHASE 3: DATA PREPARATION
# =====================================================================
banner("PHASE 3: DATA PREPARATION")

# 3.1 Select data
data = raw.dropna(subset=[TARGET]).drop(columns=["BMI"]).reset_index(drop=True)
print(f"Dropped {len(raw) - len(data)} rows without target and the BMI column -> {data.shape}")

# 3.2 Clean data: impossible zeros -> missing
for col in ["Schooling", "Income composition of resources", "percentage expenditure"]:
    data.loc[data[col] == 0, col] = np.nan

# 3.3 Construct data: log-transform heavily skewed variables, encode Status
LOG_COLS = {c: f"log {c}" for c in skewed}
for c, new in LOG_COLS.items():
    data[new] = np.log1p(data[c])
data = data.drop(columns=list(LOG_COLS))
data["Status_Developing"] = (data["Status"] == "Developing").astype(float)
data = data.drop(columns=["Status"])
FEATURES = [c for c in data.columns if c not in (GROUP, TARGET)]
skew_after = data[list(LOG_COLS.values())].skew().round(2)
print(f"Log-transformed: {list(LOG_COLS)}")
print("Skewness after log:", skew_after.to_dict())
print(f"Final candidate features ({len(FEATURES)}): {FEATURES}")

# 3.4 Format data: split by COUNTRY so no country is in both train and test
gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(data, groups=data[GROUP]))
train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)
train_countries = sorted(data.loc[train_idx, GROUP].unique())
test_countries = sorted(data.loc[test_idx, GROUP].unique())
assert not set(train_countries) & set(test_countries)
print(f"Train: {len(train_idx)} rows / {len(train_countries)} countries; "
      f"Test: {len(test_idx)} rows / {len(test_countries)} countries (no overlap)")

# 3.5 Impute: first the median of the SAME country's other years (indicators
#     change slowly within a country; uses feature values only, never the
#     target), then the TRAINING-set median for countries with no value at all.
medians = data.loc[train_idx, FEATURES].median()
missing_before = data[FEATURES].isna().sum()
country_median = data.groupby(GROUP)[FEATURES].transform("median")
filled_by_country = int((data[FEATURES].isna() & country_median.notna()).sum().sum())
data[FEATURES] = data[FEATURES].fillna(country_median).fillna(medians)
print(f"Imputed {int(missing_before.sum())} missing cells: {filled_by_country} from the same "
      f"country's median, {int(missing_before.sum()) - filled_by_country} from training-set medians")

prepared_path = BASE_DIR / "datasets" / "who_prepared.csv"
data[[GROUP] + FEATURES + [TARGET]].to_csv(prepared_path, index=False)
print(f"Prepared data saved to {prepared_path.relative_to(BASE_DIR)}")

Xtr = data.loc[train_idx, FEATURES].reset_index(drop=True)
ytr = data.loc[train_idx, TARGET].reset_index(drop=True)
Xte = data.loc[test_idx, FEATURES].reset_index(drop=True)
yte = data.loc[test_idx, TARGET].reset_index(drop=True)
groups_tr = data.loc[train_idx, GROUP].reset_index(drop=True)

# Grouped CV folds on the training set (positions within Xtr)
gkf = GroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
folds = [np.sort(test) for _, test in gkf.split(Xtr, ytr, groups_tr)]
cv_splits = [(np.setdiff1d(np.arange(len(ytr)), f), f) for f in folds]


# =====================================================================
# PHASE 4: MODELING
# =====================================================================
banner("PHASE 4: MODELING")

# 4.1 Modeling technique: OLS multiple linear regression (+ Ridge / Lasso for
#     comparison). Assumption check before modeling: multicollinearity (VIF).
Xc = sm.add_constant(Xtr)
vifs = pd.Series([variance_inflation_factor(Xc.values, i + 1) for i in range(len(FEATURES))],
                 index=FEATURES).sort_values(ascending=False)
print("--- 4.1 VIF (all candidate features, training data) ---")
print(vifs.round(2))

# 4.2 Test design: grouped hold-out (countries) + 5-fold GroupKFold CV on train.
def cv_r2(model_factory, cols):
    s = []
    for tr, te in cv_splits:
        m = model_factory().fit(Xtr.iloc[tr][cols], ytr.iloc[tr])
        s.append(r2_score(ytr.iloc[te], m.predict(Xtr.iloc[te][cols])))
    return float(np.mean(s)), float(np.std(s))

# 4.3 Build models: feature selection by several algorithms
print("\n--- Feature progression (selection algorithms) ---")
progressions, fs_comparison = fpa.run(Xtr, ytr, Xte, yte, folds, BASE_DIR)

# 4.4 Assess: pool every (algorithm, k) result, then apply the 1-SD rule:
#     the smallest k whose CV RMSE is within one fold-SD of the overall best.
pool = pd.concat([df.assign(method=fpa.METHODS[k]) for k, df in progressions.items()], ignore_index=True)
best = pool.loc[pool["CV RMSE (train)"].idxmin()]
within = pool[pool["CV RMSE (train)"] <= best["CV RMSE (train)"] + best["CV RMSE SD"]]
chosen = within.sort_values(["Number of Features", "CV RMSE (train)"]).iloc[0]
SELECTED = chosen["Selected Features"].split(", ")
print(f"\nBest CV: {best['method']} k={int(best['Number of Features'])} "
      f"(CV RMSE {best['CV RMSE (train)']:.3f} +/- {best['CV RMSE SD']:.3f})")
print(f"Selected (1-SD rule): {chosen['method']} k={len(SELECTED)} -> {SELECTED}")

# Model comparison on the held-out countries
alphas = np.logspace(-3, 3, 60)
model_specs = {
    "OLS (all features)": (lambda: LinearRegression(), FEATURES),
    f"OLS (selected {len(SELECTED)})": (lambda: LinearRegression(), SELECTED),
    "Ridge (all, CV alpha)": (lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=cv_splits)), FEATURES),
    "Lasso (all, CV alpha)": (lambda: make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=cv_splits, max_iter=50000)), FEATURES),
}


def metrics(y_true, pred):
    return {"R2": r2_score(y_true, pred), "RMSE": float(np.sqrt(mean_squared_error(y_true, pred))),
            "MAE": mean_absolute_error(y_true, pred), "MAPE_pct": float(np.mean(np.abs((y_true - pred) / y_true)) * 100)}


models = []
for name, (factory, cols) in model_specs.items():
    m = factory().fit(Xtr[cols], ytr)
    met = metrics(yte, m.predict(Xte[cols]))
    cvm, cvs = cv_r2(lambda: LinearRegression(), cols) if name.startswith("OLS") else (np.nan, np.nan)
    models.append({"name": name, "n_features": len(cols), **met, "cv_r2_mean": cvm, "cv_r2_std": cvs})
models_df = pd.DataFrame(models)
print("\n--- Model comparison (test = unseen countries) ---")
print(models_df.round(4).to_string(index=False))

# Final OLS with statsmodels for inference (classical and cluster-robust SE)
ols = sm.OLS(ytr, sm.add_constant(Xtr[SELECTED])).fit()
ols_cluster = sm.OLS(ytr, sm.add_constant(Xtr[SELECTED])).fit(cov_type="cluster", cov_kwds={"groups": groups_tr})
ci = ols.conf_int(alpha=ALPHA)
ci_cl = ols_cluster.conf_int(alpha=ALPHA)
coef_table = pd.DataFrame({
    "coef": ols.params, "se": ols.bse, "t": ols.tvalues, "p": ols.pvalues,
    "ci_lo": ci[0], "ci_hi": ci[1], "se_cluster": ols_cluster.bse, "p_cluster": ols_cluster.pvalues,
    "ci_lo_cluster": ci_cl[0], "ci_hi_cluster": ci_cl[1],
})
std_beta = ols.params.drop("const") * Xtr[SELECTED].std() / ytr.std()
print("\n--- Final OLS (training data) ---")
print(ols.summary())
print("\n--- Cluster-robust (by country) inference ---")
print(coef_table[["coef", "se_cluster", "p_cluster"]].round(4))


# =====================================================================
# PHASE 5: EVALUATION
# =====================================================================
banner("PHASE 5: EVALUATION")

# 5.1 Evaluate results
X_te_c = sm.add_constant(Xte[SELECTED], has_constant="add")
pred_frame = ols.get_prediction(X_te_c).summary_frame(alpha=ALPHA)
y_pred = pred_frame["mean"].to_numpy()
final_metrics = metrics(yte, y_pred)
cv_final = cv_r2(lambda: LinearRegression(), SELECTED)
baseline_rmse = float(np.sqrt(mean_squared_error(yte, np.full(len(yte), ytr.mean()))))
pi_coverage = float(((yte >= pred_frame["obs_ci_lower"]) & (yte <= pred_frame["obs_ci_upper"])).mean())
print("Test metrics (selected model):", {k: round(v, 4) for k, v in final_metrics.items()})
print(f"Grouped CV R2: {cv_final[0]:.4f} +/- {cv_final[1]:.4f};  baseline RMSE {baseline_rmse:.3f}")
print(f"95% prediction interval coverage on test: {pi_coverage * 100:.1f}%")

# Prediction plot with confidence and prediction intervals (test set)
order = np.argsort(y_pred)
xs = np.arange(len(order))
fig, ax = plt.subplots(figsize=(13, 6))
ax.fill_between(xs, pred_frame["obs_ci_lower"].to_numpy()[order], pred_frame["obs_ci_upper"].to_numpy()[order],
                color="tab:green", alpha=0.18, label="95% prediction interval")
ax.fill_between(xs, pred_frame["mean_ci_lower"].to_numpy()[order], pred_frame["mean_ci_upper"].to_numpy()[order],
                color="tab:green", alpha=0.45, label="95% confidence interval (mean)")
ax.plot(xs, y_pred[order], color="darkgreen", lw=1.5, label="Prediction")
ax.scatter(xs, yte.to_numpy()[order], s=10, color="tab:orange", label="Actual", zorder=3)
ax.set_title(f"Test set (unseen countries): predictions with 95% intervals "
             f"- PI coverage {pi_coverage * 100:.1f}%")
ax.set_xlabel("Test observations, sorted by predicted life expectancy")
ax.set_ylabel("Life expectancy (years)")
ax.legend(loc="upper left")
savefig("07_prediction_intervals_test.png")

# Simple linear regression on the strongest single feature, with bands
SIMPLE_X = progressions["forward_sfs"].iloc[0]["Selected Features"]
simple = sm.OLS(ytr, sm.add_constant(Xtr[[SIMPLE_X]])).fit()
grid = np.linspace(min(Xtr[SIMPLE_X].min(), Xte[SIMPLE_X].min()), max(Xtr[SIMPLE_X].max(), Xte[SIMPLE_X].max()), 80)
grid_frame = simple.get_prediction(sm.add_constant(pd.DataFrame({SIMPLE_X: grid}), has_constant="add")).summary_frame(alpha=ALPHA)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Xtr[SIMPLE_X], ytr, s=6, alpha=0.35, label="Train")
ax.scatter(Xte[SIMPLE_X], yte, s=8, alpha=0.6, color="tab:orange", label="Test")
ax.fill_between(grid, grid_frame["obs_ci_lower"], grid_frame["obs_ci_upper"], color="tab:green", alpha=0.15, label="95% prediction interval")
ax.fill_between(grid, grid_frame["mean_ci_lower"], grid_frame["mean_ci_upper"], color="tab:green", alpha=0.45, label="95% confidence interval")
ax.plot(grid, grid_frame["mean"], color="darkgreen", lw=2, label=f"Fit (R2={simple.rsquared:.3f})")
ax.set_xlabel(SIMPLE_X)
ax.set_ylabel("Life expectancy (years)")
ax.set_title(f"Simple linear regression: Life expectancy ~ {SIMPLE_X}")
ax.legend()
savefig("08_simple_regression_intervals.png")

# Residual diagnostics
resid_tr = ols.resid
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
axes[0].scatter(yte, y_pred, s=8)
lims = [min(yte.min(), y_pred.min()), max(yte.max(), y_pred.max())]
axes[0].plot(lims, lims, "r--")
axes[0].set_title("Actual vs Predicted (test)")
axes[1].scatter(ols.fittedvalues, resid_tr, s=6, alpha=0.5)
axes[1].axhline(0, color="r", ls="--")
axes[1].set_title("Residuals vs Fitted (train)")
axes[2].hist(resid_tr, bins=40, edgecolor="black")
axes[2].set_title("Residual distribution")
stats.probplot(resid_tr, dist="norm", plot=axes[3])
axes[3].set_title("Normal Q-Q")
savefig("06_residual_diagnostics.png")

# Assumption checks (train, final model). Rows are ordered by country then year
# for Durbin-Watson, so it measures within-country autocorrelation.
order_cy = data.loc[train_idx].reset_index(drop=True).sort_values([GROUP, "Year"]).index
reset = linear_reset(ols, power=3, use_f=True)
bp = het_breuschpagan(ols.resid, ols.model.exog)
jb = jarque_bera(ols.resid)
dw = durbin_watson(ols.resid.iloc[order_cy])
sel_vif = pd.Series([variance_inflation_factor(sm.add_constant(Xtr[SELECTED]).values, i + 1)
                     for i in range(len(SELECTED))], index=SELECTED)
cooks = ols.get_influence().cooks_distance[0]
assumptions = [
    {"name": "Linearity (Ramsey RESET)", "stat": float(reset.fvalue), "p": float(reset.pvalue), "pass": bool(reset.pvalue >= ALPHA)},
    {"name": "Homoscedasticity (Breusch-Pagan)", "stat": float(bp[0]), "p": float(bp[1]), "pass": bool(bp[1] >= ALPHA)},
    {"name": "Normality of residuals (Jarque-Bera)", "stat": float(jb[0]), "p": float(jb[1]), "pass": bool(jb[1] >= ALPHA)},
    {"name": "Independence (Durbin-Watson, by country/year)", "stat": float(dw), "p": None, "pass": bool(1.5 <= dw <= 2.5)},
    {"name": "No multicollinearity (max VIF)", "stat": float(sel_vif.max()), "p": None, "pass": bool(sel_vif.max() <= 10)},
    {"name": "Influential points (Cook's D > 4/n)", "stat": int((cooks > 4 / len(cooks)).sum()), "p": None, "pass": None},
]
print("\n--- Assumption checks ---")
for a in assumptions:
    print(f"{a['name']:48s} stat={a['stat']:.3f}" + (f"  p={a['p']:.4f}" if a["p"] is not None else "") + f"  pass={a['pass']}")

crit = business["success_criteria"]
checks = {
    f"Test R2 >= {crit['test_R2_min']}": final_metrics["R2"] >= crit["test_R2_min"],
    f"Test RMSE <= {crit['test_RMSE_max']} years": final_metrics["RMSE"] <= crit["test_RMSE_max"],
    f"Grouped CV R2 >= {crit['cv_R2_min']}": cv_final[0] >= crit["cv_R2_min"],
}
print("\n--- Success criteria ---")
for k, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {k}")

# 5.2 Review process
process_review = {
    "Split by country (no country in both train and test)": not set(train_countries) & set(test_countries),
    "Fallback imputation medians computed on training rows only": True,
    "Feature selection used grouped CV on training data only": True,
    "Selected model beats the mean-only baseline": final_metrics["RMSE"] < baseline_rmse,
}
print("\n--- 5.2 Process review ---")
for k, ok in process_review.items():
    print(f"[{'OK' if ok else 'ISSUE'}] {k}")
approved = all(checks.values()) and all(process_review.values())

# 5.3 Next steps
failed_assumptions = [a["name"] for a in assumptions if a["pass"] is False]
next_steps = ("Proceed to Phase 6 (deployment)." if approved else
              "Do not deploy; revisit Phase 3/4 (more features, interactions, non-linear terms).")
if failed_assumptions:
    next_steps += (" Note: " + "; ".join(failed_assumptions) + " not satisfied - report "
                   "cluster-robust standard errors and treat p-values with care.")
print("\n--- 5.3 Next steps ---")
print(next_steps)

top = std_beta.abs().sort_values(ascending=False)
print("\n--- Answers to business questions ---")
print("1. Strongest indicators (|standardised beta|):", top.head(3).round(3).to_dict())
print(f"2. Error on unseen countries: RMSE {final_metrics['RMSE']:.2f} years, R2 {final_metrics['R2']:.3f}")
print(f"3. Typical 95% prediction interval width: "
      f"{(pred_frame['obs_ci_upper'] - pred_frame['obs_ci_lower']).median():.1f} years")


# =====================================================================
# PHASE 6: DEPLOYMENT
# =====================================================================
banner("PHASE 6: DEPLOYMENT")

all_rows = np.concatenate([train_idx, test_idx])
deploy_X = sm.add_constant(data.loc[all_rows, SELECTED])
deploy = sm.OLS(data.loc[all_rows, TARGET], deploy_X).fit() if approved else None
model_path = OUT_DIR / "life_expectancy_model.joblib"
if deploy is not None:
    bundle = {"features": SELECTED, "log_cols": LOG_COLS, "medians": medians.to_dict(),
              "params": deploy.params.to_dict(), "cov": deploy.cov_params().values.tolist(),
              "scale": float(deploy.scale), "df_resid": float(deploy.df_resid)}
    joblib.dump(bundle, model_path)
    print(f"Model saved to {model_path.relative_to(BASE_DIR)}")


def predict_life_expectancy(records, path=model_path, alpha=ALPHA):
    """Score raw WHO-style records (dicts with the original column names).
    Applies the same cleaning, log transforms and median imputation, then
    returns prediction with confidence and prediction intervals."""
    b = joblib.load(path)
    df = pd.DataFrame(records)
    df.columns = [" ".join(c.split()) for c in df.columns]
    for col in ["Schooling", "Income composition of resources", "percentage expenditure"]:
        if col in df:
            df.loc[df[col] == 0, col] = np.nan
    for c, new in b["log_cols"].items():
        if c in df:
            df[new] = np.log1p(df[c])
    if "Status" in df:
        df["Status_Developing"] = (df["Status"] == "Developing").astype(float)
    X = df.reindex(columns=b["features"]).astype(float).fillna(pd.Series(b["medians"]))
    Xc = np.column_stack([np.ones(len(X)), X.to_numpy()])
    beta = np.array([b["params"]["const"]] + [b["params"][f] for f in b["features"]])
    cov = np.array(b["cov"])
    yhat = Xc @ beta
    se_mean = np.sqrt(np.einsum("ij,jk,ik->i", Xc, cov, Xc))
    t = stats.t.ppf(1 - alpha / 2, b["df_resid"])
    se_obs = np.sqrt(se_mean ** 2 + b["scale"])
    return pd.DataFrame({"prediction": yhat, "ci_lower": yhat - t * se_mean, "ci_upper": yhat + t * se_mean,
                         "pi_lower": yhat - t * se_obs, "pi_upper": yhat + t * se_obs})


examples = []
if deploy is not None:
    sample_rows = raw[raw[GROUP].isin(test_countries) & (raw["Year"] == 2014)].dropna(subset=[TARGET]).head(3)
    preds = predict_life_expectancy(sample_rows.to_dict(orient="records"))
    for (_, r), (_, p) in zip(sample_rows.iterrows(), preds.iterrows()):
        examples.append({"country": r[GROUP], "year": int(r["Year"]), "actual": float(r[TARGET]),
                         **{k: round(float(v), 2) for k, v in p.items()}})
        print(f"{r[GROUP]} {int(r['Year'])}: actual {r[TARGET]:.1f}, predicted {p['prediction']:.1f} "
              f"(95% PI {p['pi_lower']:.1f}-{p['pi_upper']:.1f})")

monitoring_plan = [
    "Re-score when WHO publishes a new year of indicators; compare with reported life expectancy.",
    "Retrain yearly, or when the rolling RMSE on new years exceeds 5 years.",
    "Alert when inputs fall outside the training range (data drift).",
    "Check that prediction-interval coverage stays near 95%.",
]


# --------------------------------------------------------------- export
def r(x, nd=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


hist_counts, hist_edges = np.histogram(raw[TARGET].dropna(), bins=30)
train_set = set(train_idx.tolist())
corr_prepared = data.loc[train_idx, FEATURES + [TARGET]].corr()
model_data = {
    "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "dataset": {"name": "Life Expectancy (WHO)", "kaggle_url": KAGGLE_URL, "file": "datasets/LifeExpectancy_WHO.csv",
                "rows": int(raw.shape[0]), "cols": int(raw.shape[1]), "countries": int(raw[GROUP].nunique()),
                "years": [int(raw["Year"].min()), int(raw["Year"].max())], "target": TARGET,
                "features_raw": FEATURES_RAW, "features": FEATURES, "status_counts": raw["Status"].value_counts().to_dict()},
    "business": business, "situation": situation, "project_plan": project_plan,
    "data": {
        "missing": {k: int(v) for k, v in missing.items()},
        "rows_with_missing": int(raw.isna().any(axis=1).sum()),
        "quality": quality,
        "corr_target": {k: r(v, 3) for k, v in corr_target.items()},
        "target_hist": {"counts": hist_counts.tolist(), "edges": [r(e, 2) for e in hist_edges]},
        "target_stats": {k: r(v, 2) for k, v in raw[TARGET].describe().items()},
        "skew_before": {k: r(v, 2) for k, v in skew_before.items()},
        "skew_after": {k: r(v, 2) for k, v in skew_after.items()},
        "corr_prepared": {"names": corr_prepared.columns.tolist(), "matrix": corr_prepared.round(3).values.tolist()},
    },
    "prep": {"dropped_rows_target": int(len(raw) - len(data)), "dropped_columns": {"BMI": quality[0]["issue"]},
             "zero_to_missing": ["Schooling", "Income composition of resources", "percentage expenditure"],
             "log_cols": LOG_COLS, "imputed_cells": int(missing_before.sum()), "imputed_by_country": filled_by_country,
             "medians": {k: r(v, 4) for k, v in medians.items()},
             "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
             "train_countries": len(train_countries), "test_countries": len(test_countries),
             "test_country_list": test_countries},
    # positions in datasets/who_prepared.csv; folds are positions within the training list
    "split": {"train": train_idx.tolist(), "test": test_idx.tolist(), "cv_folds": [f.tolist() for f in folds]},
    "vif_all": {k: r(v, 2) for k, v in vifs.items()},
    "progressions": {k: {"title": fpa.METHODS[k], "rows": [
        {"k": int(row["Number of Features"]), "features": row["Selected Features"].split(", "),
         "cv_rmse": r(row["CV RMSE (train)"]), "cv_sd": r(row["CV RMSE SD"]),
         "test_rmse": r(row["Test RMSE"]), "test_r2": r(row["Test R-squared"])} for _, row in df.iterrows()]}
        for k, df in progressions.items()},
    "exhaustive_skipped": "exhaustive_search" not in progressions,
    "feature_selection_comparison": json.loads(fs_comparison.to_json(orient="records")),
    "final": {"method": chosen["method"], "k": len(SELECTED), "features": SELECTED,
              "cv_rmse": r(chosen["CV RMSE (train)"]), "best_method": best["method"],
              "best_k": int(best["Number of Features"]), "best_cv_rmse": r(best["CV RMSE (train)"]),
              "best_cv_sd": r(best["CV RMSE SD"])},
    "models": json.loads(models_df.to_json(orient="records")),
    "coef_table": [{"name": n, **{c: r(v, 6) for c, v in row.items()}} for n, row in coef_table.iterrows()],
    "std_beta": {k: r(v) for k, v in std_beta.items()},
    "vif_selected": {k: r(v, 2) for k, v in sel_vif.items()},
    "ols": {"r2": r(ols.rsquared), "adj_r2": r(ols.rsquared_adj), "F": r(ols.fvalue, 2), "F_p": float(ols.f_pvalue),
            "n": int(ols.nobs), "df_resid": int(ols.df_resid)},
    "metrics": {**{k: r(v) for k, v in final_metrics.items()}, "baseline_rmse": r(baseline_rmse),
                "cv_r2_mean": r(cv_final[0]), "cv_r2_std": r(cv_final[1]), "pi_coverage": r(pi_coverage)},
    "interval_plot": [{"country": data.loc[test_idx[i], GROUP], "year": int(data.loc[test_idx[i], "Year"]),
                       "actual": r(yte[i], 2), "pred": r(pred_frame["mean"].iloc[i], 2),
                       "ci_lo": r(pred_frame["mean_ci_lower"].iloc[i], 2), "ci_hi": r(pred_frame["mean_ci_upper"].iloc[i], 2),
                       "pi_lo": r(pred_frame["obs_ci_lower"].iloc[i], 2), "pi_hi": r(pred_frame["obs_ci_upper"].iloc[i], 2)}
                      for i in range(len(test_idx))],
    "simple_lr": {"feature": SIMPLE_X, "intercept": r(simple.params["const"]), "slope": r(simple.params[SIMPLE_X]),
                  "r2": r(simple.rsquared), "grid": [{"x": r(x, 4), "pred": r(f["mean"], 3), "ci_lo": r(f["mean_ci_lower"], 3),
                  "ci_hi": r(f["mean_ci_upper"], 3), "pi_lo": r(f["obs_ci_lower"], 3), "pi_hi": r(f["obs_ci_upper"], 3)}
                  for x, (_, f) in zip(grid, grid_frame.iterrows())],
                  "train_points": [[r(a, 3), r(b, 2)] for a, b in zip(Xtr[SIMPLE_X], ytr)],
                  "test_points": [[r(a, 3), r(b, 2)] for a, b in zip(Xte[SIMPLE_X], yte)]},
    "assumptions": assumptions,
    "success_criteria": [{"name": k, "passed": bool(v)} for k, v in checks.items()],
    "process_review": [{"name": k, "ok": bool(v)} for k, v in process_review.items()],
    "approved": bool(approved), "next_steps": next_steps,
    "deployed": None if deploy is None else {"features": SELECTED, "params": {k: r(v, 6) for k, v in deploy.params.items()}},
    "example_predictions": examples, "monitoring_plan": monitoring_plan,
}
payload = json.dumps(model_data, indent=1, ensure_ascii=False, allow_nan=False)
(BASE_DIR / "model_data.json").write_text(payload, encoding="utf-8")
(BASE_DIR / "model_data.js").write_text("// Generated by main.py - do not edit by hand.\nwindow.MODEL_DATA = " + payload + ";\n", encoding="utf-8")
(OUT_DIR / "final_report.json").write_text(json.dumps({k: model_data[k] for k in (
    "dataset", "business", "final", "models", "metrics", "assumptions", "success_criteria",
    "process_review", "approved", "next_steps", "monitoring_plan")}, indent=2, ensure_ascii=False), encoding="utf-8")
print("Saved model_data.json / model_data.js and crispdm_output/final_report.json")

# 6.4 Review project
print("\nProject review: associations are not causal; country-level panel data means "
      "observations within a country are correlated, so cluster-robust inference is reported.")
