"""
Build the infographic (PNG) and the written report (PDF) from model_data.json.

    python docs/build_docs.py

Writes docs/infographic.html -> docs/infographic.png / docs/infographic.pdf and
docs/report.html -> docs/HW2_report.pdf; section 2.2 is computed from
datasets/flood.csv. Rendering uses headless Google Chrome (set CHROME to
override the path). Run main.py first so model_data.json and the figures in
crispdm_output/ are up to date.
"""

import html
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
M = json.loads((ROOT / "model_data.json").read_text(encoding="utf-8"))
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

VERCEL = "https://aiot-hw2-multiple-linear-regression.vercel.app/"
PAGES = "https://taemiyu.github.io/AIOT-HW2-Multiple_Linear_regression/"
REPO = "https://github.com/taemiyu/AIOT-HW2-Multiple_Linear_regression"

D, P, MET, FIN = M["dataset"], M["prep"], M["metrics"], M["final"]
SC = M["business"]["success_criteria"]
esc = html.escape


def f(v, d=2):
    return f"{v:,.{d}f}"


def fp(p):
    return "< .001" if p < 0.001 else f"{p:.3f}".lstrip("0")


CAT = ('<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="#262826" aria-hidden="true"><path d="M4 11V3.5l4.6 3.6a9 9 0 0 1 6.8 0L20 3.5V11c.8 1.2 1.2 '
       '2.5 1.2 3.8 0 3.8-4.1 6.4-9.2 6.4s-9.2-2.6-9.2-6.4C2.8 13.5 3.2 12.2 4 11Z"/></svg>')

FEATURE_DESC = {
    "Year": "年份（2000–2015）",
    "Status": "已開發／開發中國家",
    "Adult Mortality": "成人死亡率：15–60 歲每千人死亡機率",
    "infant deaths": "每千人嬰兒死亡數",
    "Alcohol": "15 歲以上人均酒精消費（公升）",
    "percentage expenditure": "健康支出占人均 GDP 的百分比",
    "Hepatitis B": "1 歲兒童 B 型肝炎疫苗覆蓋率（%）",
    "Measles": "每千人麻疹通報病例",
    "BMI": "國民平均 BMI",
    "under-five deaths": "每千人五歲以下死亡數",
    "Polio": "1 歲兒童小兒麻痺疫苗覆蓋率（%）",
    "Total expenditure": "政府健康支出占政府總支出（%）",
    "Diphtheria": "1 歲兒童白喉（DTP3）疫苗覆蓋率（%）",
    "HIV/AIDS": "每千名活產 0–4 歲 HIV/AIDS 死亡數",
    "GDP": "人均 GDP（美元）",
    "Population": "人口數",
    "thinness 1-19 years": "10–19 歲過瘦盛行率（%）",
    "thinness 5-9 years": "5–9 歲過瘦盛行率（%）",
    "Income composition of resources": "人類發展指數中的收入組成（0–1）",
    "Schooling": "平均受教育年數",
}


# ------------------------------------------------------------------ infographic
def mini_corr():
    items = sorted(M["data"]["corr_target"].items(), key=lambda kv: -abs(kv[1]))[:6]
    rows = []
    for i, (k, v) in enumerate(items):
        y = 6 + i * 26
        w = abs(v) * 120
        color = "#4176c2" if v > 0 else "#c84f63"
        rows.append(f'<text x="226" y="{y + 13}" text-anchor="end" font-size="12" fill="#2e3a31">{esc(k)}</text>'
                    f'<rect x="234" y="{y + 2}" width="{w:.1f}" height="14" rx="3" fill="{color}"/>'
                    f'<text x="{240 + w:.1f}" y="{y + 13}" font-size="12" fill="#46534a">{v:+.2f}</text>')
    return f'<svg viewBox="0 0 420 166" width="100%">{"".join(rows)}</svg>'


def mini_progress():
    rows = M["progressions"]["forward_sfs"]["rows"]
    xs = [r["k"] for r in rows]
    ys = [r["test_rmse"] for r in rows]
    lo, hi = min(ys) - 0.3, max(ys) + 0.3
    X = lambda k: 30 + (k - 1) / (max(xs) - 1) * 330
    Y = lambda v: 110 - (v - lo) / (hi - lo) * 95
    pts = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in zip(xs, ys))
    kk = FIN["k"]
    sel = next(r for r in rows if r["k"] == kk)
    return (f'<svg viewBox="0 0 380 130" width="100%"><line x1="30" y1="110" x2="360" y2="110" stroke="#cfc8b6"/>'
            f'<polyline points="{pts}" fill="none" stroke="#3d8b62" stroke-width="2.5"/>'
            f'<circle cx="{X(kk):.1f}" cy="{Y(sel["test_rmse"]):.1f}" r="7" fill="#c06f1c"/>'
            f'<text x="{X(kk) + 10:.1f}" y="{Y(sel["test_rmse"]) - 8:.1f}" font-size="12" fill="#8a5a22">k={kk}：{sel["test_rmse"]:.2f} 歲</text>'
            f'<text x="30" y="126" font-size="11" fill="#687265">k=1</text><text x="360" y="126" font-size="11" fill="#687265" text-anchor="end">k={max(xs)}</text>'
            f'<text x="30" y="12" font-size="11" fill="#687265">Forward SFS：Test RMSE 對特徵數</text></svg>')


def mini_interval():
    pts = sorted(M["interval_plot"], key=lambda q: q["pred"])[::3]
    n = len(pts)
    lo = min(min(q["pi_lo"], q["actual"]) for q in pts) - 2
    hi = max(max(q["pi_hi"], q["actual"]) for q in pts) + 2
    X = lambda i: 10 + i / (n - 1) * 360
    Y = lambda v: 140 - (v - lo) / (hi - lo) * 130
    band = " ".join(f"{X(i):.1f},{Y(q['pi_hi']):.1f}" for i, q in enumerate(pts)) + " " + \
           " ".join(f"{X(n - 1 - i):.1f},{Y(pts[n - 1 - i]['pi_lo']):.1f}" for i in range(n))
    line = " ".join(f"{X(i):.1f},{Y(q['pred']):.1f}" for i, q in enumerate(pts))
    dots = "".join(f'<circle cx="{X(i):.1f}" cy="{Y(q["actual"]):.1f}" r="2" fill="#c06f1c"/>' for i, q in enumerate(pts))
    return (f'<svg viewBox="0 0 380 150" width="100%"><polygon points="{band}" fill="#56755b" fill-opacity="0.2"/>'
            f'<polyline points="{line}" fill="none" stroke="#44604a" stroke-width="2"/>{dots}</svg>')


def infographic():
    sel_chips = "".join(f'<span class="chip">{esc(x)}</span>' for x in FIN["features"])
    takeaways = [
        ("R² 0.850", f"只用 {FIN['k']} 個特徵，在 {P['test_countries']} 個沒看過的國家解釋 85% 的壽命差異"),
        ("依國家切分", "測試國家完全不在訓練集，避免同一國家洩漏答案"),
        ("資料清理是關鍵", "刪除 48% 錯誤的 BMI；同國中位數補值讓 R² 0.813 → 0.850"),
        ("附預測區間", f"單一預測約 ±7 歲，{MET['pi_coverage'] * 100:.1f}% 實際值落在 95% 區間內"),
    ]
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>HW2 Infographic</title>
<style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; width: 1080px; background: #f6f3ea; font-family: "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif; color: #2e3a31; }}
.wrap {{ padding: 36px 40px 40px; display: grid; gap: 18px; }}
h1 {{ font-family: "Songti TC", "Noto Serif TC", serif; font-size: 40px; margin: 0; display: flex; align-items: center; gap: 14px; }}
.sub {{ color: #687265; font-size: 15px; }}
.goal {{ background: #e3eadb; border-radius: 16px; padding: 16px 20px; font-size: 16px; line-height: 1.6; }}
.facts {{ display: grid; grid-template-columns: repeat(5, 1fr); background: #fffdf7; border: 1px solid #ddd7c8; border-radius: 14px; }}
.facts div {{ padding: 12px 14px; border-right: 1px solid #ddd7c8; }} .facts div:last-child {{ border: 0; }}
.facts small {{ display: block; font-size: 11px; letter-spacing: .08em; color: #687265; text-transform: uppercase; }}
.facts b {{ font-size: 16px; }}
.step {{ display: grid; grid-template-columns: 52px 1fr; gap: 14px; background: #fffdf7; border: 1px solid #ddd7c8; border-radius: 14px; padding: 16px 18px; }}
.num {{ width: 44px; height: 44px; border-radius: 50%; background: #56755b; color: #fffdf7; display: grid; place-items: center; font-weight: 700; font-size: 18px; }}
.step h2 {{ font-family: "Songti TC", "Noto Serif TC", serif; font-size: 21px; margin: 0 0 6px; }}
.step ul {{ margin: 0; padding-left: 1.1em; font-size: 14.5px; line-height: 1.6; }}
.cols {{ display: grid; grid-template-columns: 1.1fr 1fr; gap: 16px; align-items: start; }}
.cols3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.flow {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 14px; }}
.flow span {{ background: #e3eadb; border-radius: 8px; padding: 6px 10px; }} .flow i {{ color: #9ea397; font-style: normal; }}
.chip {{ display: inline-block; background: #e3eadb; color: #44604a; border-radius: 99px; padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 13px; font-weight: 600; }}
.kpi {{ background: #e3eadb; border-radius: 12px; padding: 10px 12px; }} .kpi small {{ font-size: 12px; color: #55665a; }}
.kpi b {{ display: block; font-family: "Songti TC", "Noto Serif TC", serif; font-size: 30px; color: #44604a; }}
.take {{ background: #2e3a31; color: #fffdf7; border-radius: 16px; padding: 18px 20px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
.take b {{ display: block; font-family: "Songti TC", "Noto Serif TC", serif; font-size: 20px; color: #e3eadb; margin-bottom: 4px; }} .take p {{ margin: 0; font-size: 13.5px; line-height: 1.5; opacity: .9; }}
.foot {{ font-size: 12px; line-height: 1.7; color: #687265; text-align: center; }}
</style></head><body><div class="wrap">
<div><h1>{CAT.format(s=44)}WHO 平均壽命預測：CRISP-DM 多元線性迴歸</h1>
<div class="sub">AIoT HW2 · Kaggle Life Expectancy (WHO) · Predict the Flood · {esc(M['generated'][:10])}</div></div>
<div class="goal"><b>目標：</b>以 {D['countries']} 國、{D['years'][0]}–{D['years'][1]} 年的健康、經濟與社會指標，建立可解釋的多元線性迴歸模型，找出影響平均壽命的關鍵因子，並在「模型沒看過的國家」上評估預測能力與不確定性。</div>
<div class="facts"><div><small>Dataset</small><b>Kaggle WHO</b></div><div><small>Records</small><b>{D['rows']:,} × {D['cols']}</b></div>
<div><small>Features</small><b>{len(D['features_raw'])} → {len(D['features'])}</b></div><div><small>Target</small><b>平均壽命（歲）</b></div><div><small>Model</small><b>OLS · {FIN['k']} 個特徵</b></div></div>

<div class="step"><div class="num">1</div><div><h2>Business Understanding 商業理解</h2><ul>
<li>協助衛生政策分析：哪些指標與平均壽命最相關？對新國家能預測多準？</li>
<li>建模前訂定成功標準：測試集 R² ≥ {SC['test_R2_min']}、RMSE ≤ {SC['test_RMSE_max']} 歲、分組 CV R² ≥ {SC['cv_R2_min']}</li></ul></div></div>

<div class="step"><div class="num">2</div><div class="cols"><div><h2>Data Understanding 資料理解</h2><ul>
<li>{len(M['data']['missing'])} 個欄位有缺值，{M['data']['rows_with_missing'] / D['rows'] * 100:.0f}% 的列至少缺一個值</li>
<li>BMI 有 48% 的值不合理（國家平均 &gt; 50 或 &lt; 10）</li>
<li>每國最多 16 年資料，同國各年份高度相似</li></ul></div>
<div>{mini_corr()}</div></div></div>

<div class="step"><div class="num">3</div><div><h2>Data Preparation 資料準備</h2>
<div class="flow"><span>刪除 BMI 與無目標值的 {P['dropped_rows_target']} 列</span><i>→</i><span>不合理的 0 改為缺值</span><i>→</i><span>{len(P['log_cols'])} 個偏態變數取 log</span><i>→</i><span>依國家切分 {P['train_countries']} / {P['test_countries']} 國</span><i>→</i><span>同國中位數補值 {P['imputed_cells']:,} 格</span></div></div></div>

<div class="step"><div class="num">4</div><div class="cols"><div><h2>Modeling 特徵選擇與建模</h2><ul>
<li>Lasso、Forward SFS、RFE、ElasticNet × k = 1–{len(D['features'])}，以依國家分組的 5-fold CV 比較</li>
<li>1-SD 精簡原則選出 {FIN['k']} 個特徵：</li></ul><div style="margin-top:6px">{sel_chips}</div>
<ul style="margin-top:6px"><li>與 Ridge、Lasso 比較；係數推論採依國家分群的穩健標準誤</li></ul></div>
<div>{mini_progress()}</div></div></div>

<div class="step"><div class="num">5</div><div><h2>Evaluation 評估（37 個沒看過的國家）</h2>
<div class="cols"><div class="cols3"><div class="kpi"><small>Test R²</small><b>{MET['R2']:.3f}</b></div><div class="kpi"><small>RMSE</small><b>{MET['RMSE']:.2f}</b></div><div class="kpi"><small>95% PI 涵蓋</small><b>{MET['pi_coverage'] * 100:.1f}%</b></div></div>
<div>{mini_interval()}<div class="sub" style="font-size:12px">測試集預測值（線）、95% 預測區間（帶）與實際值（點）</div></div></div>
<ul style="margin-top:8px"><li>三項成功標準全部通過；部分 OLS 假設未通過，已誠實揭露並改用穩健推論</li></ul></div></div>

<div class="step"><div class="num">6</div><div><h2>Deployment 部署</h2>
<div class="flow"><span>通過評估才部署</span><i>→</i><span>全部資料重新擬合</span><i>→</i><span>joblib 儲存模型</span><i>→</i><span>predict_life_expectancy() 回傳預測與區間</span><i>→</i><span>網站：報告 + 互動探索器</span></div></div></div>

<div class="step"><div class="num">+</div><div><h2>互動探索器（加分）：6 個資料集、可任意輸入</h2>
<div class="cols"><ul><li><b>第二個 Kaggle 資料集：Predict the Flood</b>（Playground Series S4E5）：20 個因子評分 → 洪水發生機率，抽樣 {FS['rows']:,} 列</li>
<li>全部 20 個特徵的 OLS：測試集 R² {FS['r2']:.3f}，95% 預測區間涵蓋 {FS['pi_cov'] * 100:.1f}%</li>
<li>另有 50 Startups、Advertising、Medical Insurance、Boston Housing；也可輸入 CSV 網址或上傳檔案</li></ul>
<div class="cols3"><div class="kpi"><small>內建資料集</small><b>6</b></div><div class="kpi"><small>特徵 &gt; 13 的 Kaggle 資料集</small><b>2</b></div><div class="kpi"><small>Flood 測試 R²</small><b>{FS['r2']:.3f}</b></div></div></div></div></div>

<div class="take">{''.join(f'<div><b>{esc(a)}</b><p>{esc(b)}</p></div>' for a, b in takeaways)}</div>
<div class="foot">資料來源：Life Expectancy (WHO) {D['kaggle_url']}<br>Predict the Flood {FLOOD_URL}<br>報告：{VERCEL} · 程式：{REPO}</div>
</div></body></html>"""


# ------------------------------------------------------------------ report
FLOOD_URL = "https://www.kaggle.com/competitions/playground-series-s4e5"


def flood_summary():
    """CRISP-DM numbers and figures for the 5,000-row Predict the Flood sample (second Kaggle dataset).

    Writes docs/figures/flood_*.png for the report's Flood section.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    df = pd.read_csv(ROOT / "datasets" / "flood.csv")
    X, y = df.drop(columns="FloodProbability"), df["FloodProbability"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Forward selection on the training set: 5-fold CV RMSE for k = 1..p, then the 1-SD rule.
    folds = list(KFold(5, shuffle=True, random_state=42).split(Xtr))

    def cv_rmse(cols):
        errs = []
        for a, b in folds:
            A, B = np.c_[np.ones(len(a)), Xtr.iloc[a][cols]], np.c_[np.ones(len(b)), Xtr.iloc[b][cols]]
            beta = np.linalg.lstsq(A, ytr.iloc[a], rcond=None)[0]
            errs.append(np.sqrt(np.mean((ytr.iloc[b] - B @ beta) ** 2)))
        return float(np.mean(errs)), float(np.std(errs))

    chosen, rest, prog = [], list(X.columns), []
    while rest:
        nxt = min(rest, key=lambda c: cv_rmse(chosen + [c])[0])
        chosen.append(nxt); rest.remove(nxt)
        prog.append((len(chosen), nxt) + cv_rmse(chosen))
    best = min(prog, key=lambda r: r[2])
    k_1sd = min(r[0] for r in prog if r[2] <= best[2] + best[3])

    fit = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
    fr = fit.get_prediction(sm.add_constant(Xte)).summary_frame(alpha=0.05)
    pred = fr["mean"].to_numpy()
    err = yte.to_numpy() - pred
    cv = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
    coef = fit.params.drop("const")
    Xc = sm.add_constant(Xtr)
    total_tr, total_te = Xtr.sum(axis=1), Xte.sum(axis=1)
    simple = sm.OLS(ytr, sm.add_constant(total_tr)).fit()
    simple_pred = simple.predict(sm.add_constant(total_te))
    pi_cov = float(((yte >= fr["obs_ci_lower"]) & (yte <= fr["obs_ci_upper"])).mean())

    fig_dir = DOCS / "figures"
    fig_dir.mkdir(exist_ok=True)
    ks = [r[0] for r in prog]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.errorbar(ks, [r[2] for r in prog], yerr=[r[3] for r in prog], color="#56755b", marker="o", ms=4, capsize=3, label="CV RMSE ± 1 SD")
    ax.axhline(best[2] + best[3], color="#c08a4b", ls="--", lw=1, label="1-SD threshold")
    ax.set_xticks(ks); ax.set_xlabel("Number of features k (forward selection order)"); ax.set_ylabel("5-fold CV RMSE (train)")
    ax.set_title("Predict the Flood: forward selection - CV RMSE keeps falling up to k = 20"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "flood_feature_selection.png", dpi=120); plt.close(fig)

    order = np.argsort(pred)
    xs = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(xs, fr["obs_ci_lower"].to_numpy()[order], fr["obs_ci_upper"].to_numpy()[order],
                    color="tab:green", alpha=0.18, label="95% prediction interval")
    ax.fill_between(xs, fr["mean_ci_lower"].to_numpy()[order], fr["mean_ci_upper"].to_numpy()[order],
                    color="tab:green", alpha=0.45, label="95% confidence interval (mean)")
    ax.plot(xs, pred[order], color="darkgreen", lw=1.5, label="Prediction")
    ax.scatter(xs, yte.to_numpy()[order], s=6, color="tab:orange", label="Actual", zorder=3)
    ax.set_title(f"Predict the Flood - test set: predictions with 95% intervals - PI coverage {pi_cov * 100:.1f}%")
    ax.set_xlabel("Test observations, sorted by predicted flood probability")
    ax.set_ylabel("FloodProbability")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(fig_dir / "flood_prediction_intervals.png", dpi=120); plt.close(fig)

    return {"rows": len(df), "features": list(X.columns), "train": len(Xtr), "test": len(Xte),
            "x_min": int(X.min().min()), "x_max": int(X.max().max()), "y_min": y.min(), "y_max": y.max(),
            "corr": X.corrwith(y).sort_values(ascending=False),
            "prog": prog, "best_k": best[0], "k_1sd": k_1sd,
            "r2": 1 - (err ** 2).sum() / ((yte - yte.mean()) ** 2).sum(), "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(np.abs(err).mean()), "baseline_rmse": float(np.sqrt(np.mean((yte - ytr.mean()) ** 2))),
            "cv_mean": cv.mean(), "cv_sd": cv.std(), "y_mean": y.mean(), "y_sd": y.std(),
            "pi_cov": pi_cov, "pi_half": float(((fr["obs_ci_upper"] - fr["obs_ci_lower"]) / 2).mean()),
            "ci_half": float(((fr["mean_ci_upper"] - fr["mean_ci_lower"]) / 2).mean()),
            "coef_min": coef.min(), "coef_max": coef.max(), "all_sig": bool((fit.pvalues.drop("const") < 0.001).all()),
            "max_vif": max(variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])),
            "bp_p": het_breuschpagan(fit.resid, fit.model.exog)[1], "jb_p": jarque_bera(fit.resid)[1],
            "dw": durbin_watson(fit.resid), "reset_p": linear_reset(fit, power=2, use_f=True).pvalue,
            "simple_r2": 1 - ((yte - simple_pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()}


FS = flood_summary()


def table(headers, rows, num_cols=()):
    th = "".join(f'<th class="{"n" if i in num_cols else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    tr = "".join("<tr>" + "".join(f'<td class="{"n" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def fig(src, caption, width="100%"):
    return f'<figure><img src="{src}" style="width:{width}"><figcaption>{caption}</figcaption></figure>'


def report():
    q_zh = [("國家平均 BMI 低於 10 或高於 50", "刪除 BMI 欄"), ("教育年數 = 0", "視為缺值後補值"),
            ("收入組成指數 = 0", "視為缺值後補值"), ("健康支出比例 = 0", "視為缺值後補值"), ("平均壽命（目標）缺值", "刪除該列")]
    quality = table(["問題", "筆數", "比例", "處理"], [[q_zh[i][0], f"{q['count']:,}", f"{q['pct']}%", q_zh[i][1]] for i, q in enumerate(M["data"]["quality"])], (1, 2))
    feats = table(["特徵", "說明", "與壽命的相關係數"], [[esc(k), FEATURE_DESC.get(k, ""), (f"{M['data']['corr_target'][k]:+.3f}" if k in M["data"]["corr_target"] else "—")] for k in D["features_raw"]], (2,))
    skew = table(["變數", "偏態（前）", "偏態（後）"], [[esc(k), f"{M['data']['skew_before'][k]:.2f}", f"{M['data']['skew_after'][v]:.2f}"] for k, v in P["log_cols"].items()], (1, 2))
    vif = table(["特徵", "VIF"], [[esc(k), f"{v:.2f}" + ("（嚴重共線）" if v > 10 else "")] for k, v in sorted(M["vif_all"].items(), key=lambda kv: -kv[1])], (1,))
    comp = table(["演算法", "k（CV 最佳）", "k（1-SD）", "選出的特徵", "CV RMSE", "Test RMSE", "Test R²"],
                 [[esc(r["Method"]), r["k (CV best)"], r["k (1-SD)"], esc(r["Selected Features (1-SD)"]), f"{r['CV RMSE (train)']:.3f}", f"{r['Test RMSE']:.3f}", f"{r['Test R-squared']:.4f}"] for r in M["feature_selection_comparison"]], (1, 2, 4, 5, 6))
    models = table(["模型", "特徵數", "Test R²", "RMSE", "MAE", "MAPE", "分組 CV R²"],
                   [[esc(m["name"]), m["n_features"], f"{m['R2']:.4f}", f"{m['RMSE']:.3f}", f"{m['MAE']:.3f}", f"{m['MAPE_pct']:.2f}%", ("—" if m["cv_r2_mean"] is None else f"{m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.3f}")] for m in M["models"]], (1, 2, 3, 4, 5, 6))
    coef = table(["變數", "係數", "SE", "p", "95% CI", "叢集穩健 SE", "叢集穩健 p", "標準化 β"],
                 [[esc("Intercept" if c["name"] == "const" else c["name"]), f"{c['coef']:.4f}", f"{c['se']:.4f}", fp(c["p"]), f"[{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]", f"{c['se_cluster']:.4f}", fp(c["p_cluster"]),
                   ("—" if c["name"] == "const" else f"{M['std_beta'][c['name']]:.3f}")] for c in M["coef_table"]], (1, 2, 3, 4, 5, 6, 7))
    a_zh = ["線性（Ramsey RESET）", "同質變異（Breusch-Pagan）", "殘差常態（Jarque-Bera）", "誤差獨立（Durbin-Watson）", "無共線性（最大 VIF）", "影響點（Cook's D > 4/n 筆數）"]
    assume = table(["假設", "統計量", "p 值", "結果"], [[a_zh[i], (str(a["stat"]) if isinstance(a["stat"], int) else f"{a['stat']:.3f}"), ("—" if a["p"] is None else fp(a["p"])), ("參考" if a["pass"] is None else ("通過" if a["pass"] else "未通過"))] for i, a in enumerate(M["assumptions"])], (1, 2))
    crit_zh = [f"測試集 R² ≥ {SC['test_R2_min']}", f"測試集 RMSE ≤ {SC['test_RMSE_max']} 歲", f"分組 CV R² ≥ {SC['cv_R2_min']}"]
    crit_v = [f"{MET['R2']:.4f}", f"{MET['RMSE']:.2f} 歲", f"{MET['cv_r2_mean']:.4f}"]
    crit = table(["成功標準", "結果", "判定"], [[crit_zh[i], crit_v[i], "通過" if c["passed"] else "未通過"] for i, c in enumerate(M["success_criteria"])], (1,))
    examples = table(["國家（2014）", "實際", "預測", "95% 預測區間"], [[esc(e["country"]), f"{e['actual']:.1f}", f"{e['prediction']:.1f}", f"{e['pi_lower']:.1f} – {e['pi_upper']:.1f}"] for e in M["example_predictions"]], (1, 2, 3))
    eq = "Life expectancy = " + f"{M['coef_table'][0]['coef']:.3f}" + "".join(f" {'−' if c['coef'] < 0 else '+'} {abs(c['coef']):.4f} × {esc(c['name'])}" for c in M["coef_table"][1:])
    sb = sorted(M["std_beta"].items(), key=lambda kv: -abs(kv[1]))
    pi_half = sum((q["pi_hi"] - q["pi_lo"]) / 2 for q in M["interval_plot"]) / len(M["interval_plot"])
    ci_half = sum((q["ci_hi"] - q["ci_lo"]) / 2 for q in M["interval_plot"]) / len(M["interval_plot"])
    nb_sources = [
        ('Kaggle：Life Expectancy (WHO) 資料集頁面', 'https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who'),
        ('Kaggle：WHO Life Expectancy - Multi Linear Regression', 'https://www.kaggle.com/code/captainozlem/who-life-expectancy-multi-linear-regression'),
        ('Kaggle：(Detailed) REGRESSION PROJECT on Life Expectancy', 'https://www.kaggle.com/code/ecedolen/detailed-regression-project-on-life-expectancy'),
        ('Enjoy Algorithms：Life Expectancy Prediction Using Machine Learning', 'https://www.enjoyalgorithms.com/blog/life-expectancy-prediction-using-linear-regression/'),
        ('NYC Data Science Academy：Modeling Life Expectancy', 'https://nycdatascience.com/blog/student-works/life-expectancy-by-country/'),
        ('CAUSE USCLAP：Analyzing Significant Predictors of Life Expectancy across Countries', 'https://www.causeweb.org/usproc/sites/default/files/usclap/2022-1/analyzing-significant-predictors-of-life-expectancy-across-countries.pdf'),
        ('arXiv 2510.00542：Interpretable Machine Learning for Life Expectancy Prediction: A Comparative Study of Linear Regression, Decision Tree, and Random Forest', 'https://arxiv.org/abs/2510.00542'),
        ('iJOE（2022）：Life Expectancy Prediction through Analysis of Immunization and HDI Factors using Machine Learning Regression Algorithms', 'https://online-journals.org/index.php/i-joe/article/view/33315'),
        ('GitHub：frinkleko/Life-expectancy（Data mining group project）', 'https://github.com/frinkleko/life-expectancy'),
    ]
    img = lambda name: f"../crispdm_output/{name}"
    flood_table = table(["項目", "結果"], [
        ["切分", f"隨機 80/20（種子 42）：訓練 {FS['train']:,} 列、測試 {FS['test']:,} 列"],
        ["模型", f"OLS，全部 {len(FS['features'])} 個特徵"],
        ["測試集 R²", f"{FS['r2']:.4f}"], ["測試集 RMSE", f"{FS['rmse']:.4f}"],
        ["5-fold CV R²", f"{FS['cv_mean']:.4f} ± {FS['cv_sd']:.3f}"],
        ["95% 預測區間涵蓋率", f"{FS['pi_cov'] * 100:.1f}%（平均約 ±{FS['pi_half']:.3f}）"]])
    flood_eval = table(["指標", "結果"], [
        ["測試集 R²", f"{FS['r2']:.4f}"], ["測試集 RMSE", f"{FS['rmse']:.4f}"], ["測試集 MAE", f"{FS['mae']:.4f}"],
        ["只猜平均的基準 RMSE", f"{FS['baseline_rmse']:.4f}"], ["5-fold CV R²", f"{FS['cv_mean']:.4f} ± {FS['cv_sd']:.3f}"],
        ["95% 預測區間涵蓋率", f"{FS['pi_cov'] * 100:.1f}%"]], (1,))
    ok = lambda b: "通過" if b else "未通過"
    flood_assume = table(["假設", "p 值／統計量", "結果"], [
        ["線性（Ramsey RESET）", fp(FS["reset_p"]), ok(FS['reset_p'] > 0.05)],
        ["同質變異（Breusch-Pagan）", fp(FS["bp_p"]), ok(FS['bp_p'] > 0.05)],
        ["殘差常態（Jarque-Bera）", fp(FS["jb_p"]), ok(FS['jb_p'] > 0.05)],
        ["誤差獨立（Durbin-Watson）", f"{FS['dw']:.3f}", ok(1.5 < FS['dw'] < 2.5)],
        ["無共線性（最大 VIF）", f"{FS['max_vif']:.2f}", ok(FS['max_vif'] < 5)]], (1,))
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>HW2 報告書：WHO 平均壽命多元線性迴歸</title>
<style>
@page {{ size: A4; margin: 18mm 16mm 18mm; }}
body {{ font-family: "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", sans-serif; font-size: 10.5pt; line-height: 1.65; color: #222; }}
h1, h2, h3 {{ font-family: "Songti TC", "Noto Serif TC", serif; color: #2e3a31; }}
h1 {{ font-size: 24pt; margin: 0 0 6px; }} h2 {{ font-size: 16pt; margin: 22px 0 8px; border-bottom: 2px solid #56755b; padding-bottom: 3px; }}
h3 {{ font-size: 12pt; margin: 14px 0 6px; }}
.cover {{ height: 245mm; display: flex; flex-direction: column; justify-content: center; gap: 10px; page-break-after: always; }}
.cover .tag {{ color: #56755b; font-weight: 700; letter-spacing: .1em; }} .cover p {{ margin: 2px 0; color: #444; }}
.cover .box {{ margin-top: 24px; background: #eef3ea; border-left: 4px solid #56755b; padding: 12px 16px; }}
.toc {{ page-break-after: always; }} .toc ol {{ line-height: 2; }}
.page {{ page-break-before: always; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6px 0 10px; page-break-inside: avoid; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 4px 6px; text-align: left; vertical-align: top; }} th {{ background: #eef3ea; }}
td.n, th.n {{ text-align: right; white-space: nowrap; }}
figure {{ margin: 8px 0 12px; page-break-inside: avoid; }} figure img {{ border: 1px solid #ddd; }} figcaption {{ font-size: 9pt; color: #555; }}
.eq {{ font-family: Menlo, monospace; font-size: 9pt; background: #f4f4f4; padding: 6px 8px; overflow-wrap: anywhere; word-break: break-word; }}
.note {{ font-size: 9.5pt; color: #444; background: #f8f6ef; border-left: 3px solid #c08a4b; padding: 6px 10px; }}
blockquote {{ margin: 6px 0; padding: 8px 12px; background: #f4f4f4; border-left: 3px solid #999; }}
a {{ color: #44604a; word-break: break-all; }}
</style></head><body>

<div class="cover">
  <div class="tag">AIoT · HW2 · Multiple Linear Regression</div>
  <h1>WHO 平均壽命預測：<br>以 CRISP-DM 進行多元線性迴歸分析</h1>
  <p>資料集：Kaggle「Life Expectancy (WHO)」（{D['countries']} 國、{D['years'][0]}–{D['years'][1]} 年、{len(D['features_raw'])} 個候選特徵）</p>
  <p>第二個資料集（互動探索器）：Kaggle「Predict the Flood」（Playground Series S4E5，{len(FS['features'])} 個特徵）</p>
  <p>作者：taemiyu　　日期：{esc(M['generated'][:10])}</p>
  <div class="box">
    <p>網頁報告：<a href="{VERCEL}">{VERCEL}</a></p>
    <p>互動探索器：<a href="{VERCEL}app">{VERCEL}app</a></p>
    <p>GitHub Pages：<a href="{PAGES}">{PAGES}</a></p>
    <p>程式碼：<a href="{REPO}">{REPO}</a></p>
  </div>
</div>

<div class="toc"><h2>目錄</h2><ol>
<li>摘要</li><li>資料集</li><li>Business Understanding 商業理解</li><li>Data Understanding 資料理解</li>
<li>Data Preparation 資料準備</li><li>Modeling 建模與特徵選擇</li><li>Evaluation 評估（含預測區間圖）</li>
<li>Deployment 部署</li><li>結論、限制與未來改進</li><li>第二個資料集：Predict the Flood 的 CRISP-DM（精簡版）</li><li>GPT 輔助內容</li><li>NotebookLM 摘要</li><li>參考資料</li></ol></div>

<h2>1. 摘要</h2>
<p>本報告以 Kaggle 的 WHO Life Expectancy 資料集（{D['rows']:,} 列、{D['cols']} 欄），依 CRISP-DM 六階段建立多元線性迴歸模型預測各國平均壽命。資料準備階段處理了大量缺值與不合理值（刪除 48% 錯誤的 BMI 欄、偏態變數取 log、同國中位數補值），並<b>依國家切分訓練／測試集</b>，避免同一國家的不同年份洩漏答案。以 Lasso、Forward SFS、RFE、ElasticNet 四種演算法搭配依國家分組的交叉驗證進行特徵選擇，依 1-SD 精簡原則選出 {FIN['k']} 個特徵：{esc('、'.join(FIN['features']))}。最終模型在 {P['test_countries']} 個模型沒看過的國家上達到 <b>R² {MET['R2']:.3f}、RMSE {MET['RMSE']:.2f} 歲</b>，95% 預測區間涵蓋 {MET['pi_coverage'] * 100:.1f}% 的實際值，三項成功標準全部通過。此外，互動探索器加入第二個特徵數大於 13 的 Kaggle 資料集「Predict the Flood」（{len(FS['features'])} 個特徵），全部特徵的 OLS 在測試集達到 R² {FS['r2']:.3f}（見 2.2 節，完整流程見第 10 節）。</p>

<h2>2. 資料集</h2>
<h3>2.1 主資料集：Life Expectancy (WHO)</h3>
<p>來源：<a href="{D['kaggle_url']}">{D['kaggle_url']}</a>（Kaggle，作者 KumarRajarshi；原始資料來自 WHO Global Health Observatory 與聯合國）。每列為一個國家在一個年份的資料，預測目標為平均壽命（Life expectancy，歲）。候選特徵共 {len(D['features_raw'])} 個（不含國家名稱與目標），符合作業「特徵數大於 13」的要求：</p>
{feats}
<h3>2.2 第二個 Kaggle 資料集：Predict the Flood（互動探索器）</h3>
<p>來源：<a href="{FLOOD_URL}">{FLOOD_URL}</a>（Kaggle Playground Series Season 4 Episode 5，2024；以 Flood Prediction Factors 資料集為基礎產生的合成資料）。目標為洪水發生機率（FloodProbability，平均 {FS['y_mean']:.3f}、標準差 {FS['y_sd']:.3f}），候選特徵 {len(FS['features'])} 個，同樣符合「特徵數大於 13」：</p>
<p class="eq">{esc('、'.join(FS['features']))}</p>
<p>原始訓練資料有 1,117,957 列，超過瀏覽器端計算的上限，因此以隨機種子 42 抽樣 {FS['rows']:,} 列（<span class="eq">datasets/flood.csv</span>），抽樣後目標的平均與標準差和原始資料一致。資料沒有缺值，各特徵都是 0–16 的整數評分，不需要額外清理。</p>
{flood_table}
<p>上表為 Python（statsmodels）重算的結果，與探索器一致（探索器測試集 R² 0.844、5-fold CV R² 0.830）。20 個特徵的係數都為正且大小相近（{FS['coef_min']:.4f}–{FS['coef_max']:.4f}）{'，全部 p &lt; .001' if FS['all_sig'] else ''}，因此特徵選擇的 1-SD 原則會保留全部 20 個特徵：少用任何一個因子，誤差都會變大。這和 WHO 資料集「少數特徵就足夠」的結果形成對比。在探索器切換到這個資料集，可以看到特徵選擇比較、係數表，以及含信賴區間與預測區間的預測圖。依 CRISP-DM 六階段的完整說明見第 10 節。</p>

<h2 class="page">3. Business Understanding 商業理解</h2>
<p><b>商業目標：</b>協助衛生政策分析者，從各國的健康、經濟與社會指標估計平均壽命，並找出最關鍵的指標。</p>
<p><b>商業問題：</b>（1）哪些指標與平均壽命最相關？（2）對「沒看過的國家」能預測多準？（3）單一預測的不確定性有多大？</p>
<p><b>資料探勘目標：</b>建立可解釋的多元線性迴歸模型 <span class="eq">Life expectancy = f(指標)</span>，並以未見過的國家評估。</p>
<p><b>成功標準（建模前訂定）：</b>測試集 R² ≥ {SC['test_R2_min']}、測試集 RMSE ≤ {SC['test_RMSE_max']} 歲、分組 5-fold CV R² ≥ {SC['cv_R2_min']}。</p>
<p><b>現況評估：</b>資源為一份公開資料集與 Python（scikit-learn、statsmodels）；模型須可解釋；主要風險為缺值多、不合理數值、以及同一國家重複出現造成的資料洩漏。</p>

<h2>4. Data Understanding 資料理解</h2>
<p>{len(M['data']['missing'])} 個欄位有缺值，{M['data']['rows_with_missing']:,} 列（{M['data']['rows_with_missing'] / D['rows'] * 100:.0f}%）至少缺一個值，因此不能直接刪除有缺值的列。與平均壽命相關最強的是受教育年數（r = {M['data']['corr_target']['Schooling']:+.2f}）、收入組成指數（{M['data']['corr_target']['Income composition of resources']:+.2f}）與成人死亡率（{M['data']['corr_target']['Adult Mortality']:+.2f}）。</p>
{fig(img('01_missing_values.png'), '圖 1　各欄缺值數', '80%')}
{fig(img('03_correlation_with_target.png'), '圖 2　各特徵與平均壽命的 Pearson 相關係數', '80%')}
{fig(img('02_target_distribution.png'), f"圖 3　平均壽命分布（平均 {M['data']['target_stats']['mean']:.1f} 歲，範圍 {M['data']['target_stats']['min']}–{M['data']['target_stats']['max']} 歲）", '70%')}
{fig(img('05_top_features_scatter.png'), '圖 4　相關最強的 6 個特徵與平均壽命的散佈圖')}
<h3>不合理數值與處理</h3>{quality}

<h2 class="page">5. Data Preparation 資料準備</h2>
<ol>
<li><b>選擇：</b>刪除 {P['dropped_rows_target']} 列沒有目標值的資料；刪除 BMI 欄（48% 的值不可能是國家平均）。</li>
<li><b>清理：</b>教育年數、收入組成指數、健康支出比例為 0 者視為缺值。</li>
<li><b>建構：</b>{len(P['log_cols'])} 個嚴重偏態的變數取 log(1+x)；Status 轉為 0/1（Status_Developing）。最終候選特徵 {len(D['features'])} 個。</li>
<li><b>切分：</b>以 GroupShuffleSplit <b>依國家</b> 80/20 切分（訓練 {P['train_countries']} 國 {P['train_rows']:,} 列、測試 {P['test_countries']} 國 {P['test_rows']} 列），兩者沒有重疊的國家；交叉驗證同樣使用 GroupKFold。</li>
<li><b>補值：</b>共 {P['imputed_cells']:,} 格。先用同一國家其他年份的中位數（{P['imputed_by_country']:,} 格），該國完全沒有資料時才用訓練集中位數（{P['imputed_cells'] - P['imputed_by_country']:,} 格）；只使用特徵值、不使用目標值，沒有資料洩漏。開發過程中比較：只用訓練集中位數時測試集 R² 為 0.813，改用同國中位數後提升至 {MET['R2']:.3f}。</li>
</ol>
<div class="note">為什麼依國家切分：同一國家各年份的資料非常相似，隨機切分會讓模型在測試時「看過」同一國家，使分數被高估。依國家切分的結果更能代表「預測一個新國家」的實際表現。</div>
<h3>偏態變數取 log 的效果</h3>{skew}

<h2 class="page">6. Modeling 建模與特徵選擇</h2>
<h3>6.1 共線性檢查（VIF）</h3>
<p>嬰兒死亡數與五歲以下死亡數幾乎是同一件事，VIF 超過 190；特徵選擇會只留下其中之一。</p>{vif}
<h3>6.2 特徵選擇</h3>
<p>四種演算法（Lasso 路徑、Forward SFS、RFE、ElasticNet 路徑）各自只看訓練集，以依國家分組的 5-fold CV RMSE 比較 k = 1 到 {len(D['features'])} 個特徵。{len(D['features'])} 個特徵的窮舉搜尋需 {2 ** len(D['features']) - 1:,} 種組合，計算量過大，因此略過。CV RMSE 最低的是 {esc(FIN['best_method'])} k = {FIN['best_k']}（{FIN['best_cv_rmse']:.3f} ± {FIN['best_cv_sd']:.3f} 歲）；依 <b>1-SD 精簡原則</b>（CV RMSE 落在最佳值一個標準差內、特徵最少），選出 <b>{esc(FIN['method'])}，k = {FIN['k']}</b>。</p>
{fig('../all_algorithms_comparison_plot.png', '圖 5　各演算法的 Test RMSE 對特徵數', '85%')}
{comp}
<h3>6.3 模型比較</h3>{models}
<p>Ridge 與 Lasso 以分組交叉驗證挑選正則化強度，表現與全部特徵的 OLS 相近；精簡模型只用 {FIN['k']} 個特徵，誤差只多約 0.26 歲，但更容易解釋。</p>
<h3>6.4 最終模型</h3><p class="eq">{eq}</p>{coef}
<p>訓練集 n = {M['ols']['n']}，R² = {M['ols']['r2']:.4f}，調整後 R² = {M['ols']['adj_r2']:.4f}，F = {M['ols']['F']:,.2f}（p < .001）。因同一國家內的誤差相關，推論以依國家分群的叢集穩健標準誤為準；四個係數仍全部顯著。</p>

<h2 class="page">7. Evaluation 評估</h2>
<p>在 {P['test_countries']} 個沒看過的國家上：R² <b>{MET['R2']:.4f}</b>、RMSE <b>{MET['RMSE']:.2f} 歲</b>、MAE {MET['MAE']:.2f} 歲、MAPE {MET['MAPE_pct']:.2f}%；只猜平均的基準 RMSE 為 {MET['baseline_rmse']:.2f} 歲。分組 CV R² 為 {MET['cv_r2_mean']:.3f} ± {MET['cv_r2_std']:.3f}。</p>
{crit}
<h3>7.1 預測圖：95% 信賴區間與預測區間</h3>
{fig(img('07_prediction_intervals_test.png'), f"圖 6　測試集預測值（依預測值排序）與 95% 信賴區間（深色，平均約 ±{ci_half:.2f} 歲）、95% 預測區間（淺色，平均約 ±{pi_half:.2f} 歲）。實際值有 {MET['pi_coverage'] * 100:.1f}% 落在預測區間內。")}
{fig(img('08_simple_regression_intervals.png'), f"圖 7　單純線性迴歸：Life expectancy ~ {esc(M['simple_lr']['feature'])}（R² = {M['simple_lr']['r2']:.3f}），含 95% 信賴區間與預測區間", '90%')}
<h3>7.2 殘差診斷與假設檢定</h3>
{fig(img('06_residual_diagnostics.png'), '圖 8　實際 vs 預測、殘差 vs 預測值、殘差分布、常態 Q-Q 圖')}
{assume}
<p>真實的跨國資料很難完全符合 OLS 假設：殘差有異質變異與非常態，且同一國家內的誤差高度相關（Durbin-Watson {M['assumptions'][3]['stat']:.2f}）。因此係數推論改用叢集穩健標準誤，並將非線性項與混合效果模型列為後續改進方向。最終特徵的最大 VIF 為 {max(M['vif_selected'].values()):.2f}，沒有共線性問題。</p>
<h3>7.3 商業問題的答案</h3>
<ol><li><b>最重要的指標：</b>{'、'.join(f"{esc(k)}（標準化 β {v:+.3f}）" for k, v in sb)}。</li>
<li><b>預測準確度：</b>對沒看過的國家，平均誤差約 {MET['MAE']:.1f} 歲（RMSE {MET['RMSE']:.2f} 歲）。</li>
<li><b>不確定性：</b>單一國家年份的 95% 預測區間約 ±{pi_half:.1f} 歲；平均值的信賴區間約 ±{ci_half:.2f} 歲。</li></ol>

<h2 class="page">8. Deployment 部署</h2>
<p>通過評估後，以全部資料重新擬合最終模型並用 joblib 儲存（<span class="eq">crispdm_output/life_expectancy_model.joblib</span>）。<span class="eq">predict_life_expectancy()</span> 接受原始欄位，套用相同的清理、log 轉換與補值，回傳預測值、95% 信賴區間與預測區間。成果另部署為網站（報告與互動探索器）。</p>
{examples}
<p class="note">Belgium 2014 年的實際值 89.0 歲明顯偏高（比利時實際約 81 歲），是原始資料的錯誤；模型預測約 80 歲反而合理，再次說明資料品質檢查的重要。</p>
<p><b>監控計畫：</b>新年度資料公布時重新預測並比對；每年重新訓練，或滾動 RMSE 超過 5 歲時立即重訓；輸入超出訓練範圍時警示；定期檢查預測區間涵蓋率是否維持約 95%。</p>

<h2>9. 結論、限制與未來改進</h2>
<ul><li>只用 {FIN['k']} 個特徵的線性模型，就能解釋沒看過的國家約 {MET['R2'] * 100:.0f}% 的平均壽命差異；收入組成、HIV/AIDS 與成人死亡率是最主要的因子。</li>
<li>依國家切分與同國中位數補值，是讓評估可信、模型更準的兩個關鍵。</li>
<li>限制：資料有錯誤與大量缺值；相關不等於因果；OLS 假設未全部滿足；未納入環境污染等其他因子。</li>
<li>未來改進：極端值縮尾處理、交互作用與非線性項、考慮國家內相關的混合效果模型，以及與隨機森林等非線性模型比較。</li></ul>

<h2 class="page">10. 第二個資料集：Predict the Flood 的 CRISP-DM（精簡版）</h2>
<p class="note">WHO 是本作業的主資料集（第 3–9 節）。本節用同樣的六個階段，精簡說明第二個 Kaggle 資料集的分析；數字由 Python（statsmodels、scikit-learn）重算，互動版本見探索器的「Predict the Flood」。</p>
<h3>10.1 Business Understanding 商業理解</h3>
<p><b>目標：</b>依 20 個洪水風險因子的評分，估計一個地區的洪水發生機率，並找出影響最大的因子。<b>問題：</b>（1）哪些因子最重要？（2）線性模型能預測多準？（3）單一預測的不確定性有多大？<b>評估方式：</b>以測試集 R²、RMSE 與「只猜平均」的基準比較，並用 5-fold CV 確認結果穩定。</p>
<h3>10.2 Data Understanding 資料理解</h3>
<p>抽樣後 {FS['rows']:,} 列、{len(FS['features'])} 個特徵，沒有缺值；每個特徵都是 {FS['x_min']}–{FS['x_max']} 的整數評分。目標 FloodProbability 介於 {FS['y_min']:.3f}–{FS['y_max']:.3f}（平均 {FS['y_mean']:.3f}、標準差 {FS['y_sd']:.3f}）。每個因子與目標的相關係數都只有 {FS['corr'].min():.2f}–{FS['corr'].max():.2f}（最高 {esc(FS['corr'].index[0])} r = {FS['corr'].iloc[0]:.3f}，最低 {esc(FS['corr'].index[-1])} r = {FS['corr'].iloc[-1]:.3f}），沒有單一主導的因子。這是 Kaggle 依 Flood Prediction Factors 資料集產生的合成資料。</p>
<h3>10.3 Data Preparation 資料準備</h3>
<ol><li>刪除流水號 id 欄。</li>
<li>原始 1,117,957 列超過瀏覽器端上限，以隨機種子 42 抽樣 {FS['rows']:,} 列；抽樣後目標的平均與標準差和原始資料一致。</li>
<li>沒有缺值或不合理值，所有特徵已是數值評分，不需要補值、編碼或取 log。</li>
<li>隨機 80/20 切分（種子 42）：訓練 {FS['train']:,} 列、測試 {FS['test']:,} 列。每列是獨立樣本，沒有 WHO「同一國家重複出現」的問題，因此不需要分組切分。</li></ol>
<h3>10.4 Modeling 建模與特徵選擇</h3>
<p>最大 VIF 只有 {FS['max_vif']:.2f}，特徵之間幾乎互相獨立。以 Forward SFS 在訓練集計算 k = 1 到 {len(FS['features'])} 的 5-fold CV RMSE：k = 1 時為 {FS['prog'][0][2]:.4f}，之後每多一個特徵誤差都再下降，到 k = {len(FS['features'])} 時為 {FS['prog'][-1][2]:.4f}。CV 最佳是 k = {FS['best_k']}，依 1-SD 精簡原則也是 k = {FS['k_1sd']}，所以最終模型為<b>全部 {len(FS['features'])} 個特徵的 OLS</b>。係數全部為正且大小相近（{FS['coef_min']:.4f}–{FS['coef_max']:.4f}）{'，全部 p &lt; .001' if FS['all_sig'] else ''}：任一因子的評分每增加 1 分，洪水機率約增加 0.5 個百分點。</p>
{fig('figures/flood_feature_selection.png', '圖 9　Predict the Flood：Forward SFS 的 5-fold CV RMSE 對特徵數（誤差棒為 ±1 SD，虛線為 1-SD 門檻）', '90%')}
<h3>10.5 Evaluation 評估</h3>
{flood_eval}
{fig('figures/flood_prediction_intervals.png', f"圖 10　Predict the Flood 測試集預測值（依預測值排序）與 95% 信賴區間（深色，平均約 ±{FS['ci_half']:.4f}）、95% 預測區間（淺色，平均約 ±{FS['pi_half']:.3f}）。實際值有 {FS['pi_cov'] * 100:.1f}% 落在預測區間內。")}
{flood_assume}
<p>RMSE 從只猜平均的 {FS['baseline_rmse']:.4f} 降到 {FS['rmse']:.4f}，預測區間涵蓋率接近 95%。殘差有異質變異、非常態，RESET 也顯示有輕微的非線性：洪水機率在極高與極低端較平緩。另外，只用「20 個因子評分的加總」做單純線性迴歸，測試集 R² 就有 {FS['simple_r2']:.3f}，表示洪水機率主要取決於總風險分數，各因子的貢獻大致相同。</p>
<h3>10.6 Deployment 部署</h3>
<p>部署於互動探索器：資料集選單選「Predict the Flood」，即可在瀏覽器中重跑特徵選擇、查看係數表、假設檢定與含信賴區間、預測區間的預測圖，並用滑桿即時預測。<b>限制：</b>這是合成資料，結論不能直接套用到真實地區；實際應用需要以真實觀測資料重新訓練，並持續監控預測區間的涵蓋率。<b>未來改進：</b>加入總分的二次項或改用非線性模型，處理兩端較平緩的關係。</p>

<h2 class="page">11. GPT 輔助內容（AI 工具：Claude）</h2>
<p>依作業規定標示：以下列出本專案中由生成式 AI 工具協助完成的部分；與 AI 的完整對話紀錄另以 PDF 繳交。</p>
<p><b>使用的 AI 工具：</b></p>
<ul><li><b>Claude（Claude Code）</b>：主要開發工具，負責程式撰寫、資料分析、網頁製作與報告文字草稿（含本報告書）。</li>
<li><b>NotebookLM</b>：研究網路上同主題的解法並產生摘要（見第 12 節）。</li></ul>
<p><b>AI 協助的部分：</b>資料集挑選建議、資料清理與補值策略、CRISP-DM 程式（main.py、feature_progression_analysis.py）、瀏覽器端統計引擎與特徵選擇演算法、網頁版面與圖表、報告文字草稿。</p>
<p><b>本人負責與確認的部分：</b></p>
<ul><li><b>作業方向：</b>依作業要求提出需求，包括 CRISP-DM 流程、加分項目（資料集下拉選單、任意輸入資料集）、指定 Boston Housing 必須出現在選單中。</li>
<li><b>資料集決定：</b>參考作業說明中提供的資料集清單與 Claude 的建議，選定 WHO Life Expectancy，並自行從 Kaggle 下載資料；之後再從作業清單選定第二個 Kaggle 資料集 Predict the Flood，自行接受競賽規則並下載。</li>
<li><b>專案與部署：</b>建立 GitHub repository，將 HW2 與 HW1 分開管理；在 Vercel 與 GitHub Pages 完成部署設定。</li>
<li><b>網站設計方向：</b>指定版面參考網站、選擇配色風格與黑貓元素，並多次檢視畫面、提出修改。</li>
<li><b>研究：</b>使用 NotebookLM 整理同主題文獻並產生摘要。</li>
<li><b>檢查與確認：</b>檢視分析結果、網站與報告內容，決定最終繳交版本。</li></ul>
<p><b>驗證方式：</b>網頁計算結果已與 Python（statsmodels、scikit-learn）逐項比對一致。</p>

<h2>12. NotebookLM 摘要</h2>
<p>使用 NotebookLM 研究網路上同主題（WHO Life Expectancy 迴歸分析）的解法後產生的摘要：</p>
<blockquote><b>資料清理</b>多採用插值法填補缺漏值、縮尾處理（Winsorization）極端值，並校正數據末端零被截斷的錯誤。<b>核心預測因子</b>包含成人死亡率、受教育年數、HIV/AIDS 死亡率、資源收入組成（HDI）及疫苗覆蓋率。<b>使用模型</b>涵蓋多元線性迴歸、Ridge 與 Lasso 正則化模型及隨機森林；其中隨機森林與二次多項式迴歸<b>評估結果</b>最佳（R² 達 0.92 以上）。<b>常見限制</b>則為原始資料缺漏率高、變數間存在高度共線性，以及未納入環境污染等潛在因子。</blockquote>
<h3>NotebookLM 使用的來源</h3><ol>{''.join(f'<li>{esc(t)}：<a href="{u}">{u}</a></li>' for t, u in nb_sources)}</ol>
<h3>與本專案的比較（本專案自行分析，非 NotebookLM 內容）</h3>
<ul><li><b>預測因子一致：</b>文獻指出的成人死亡率、HIV/AIDS、收入組成指數與疫苗覆蓋率，正是本專案選出的 4 個特徵。</li>
<li><b>R² 較低的原因：</b>文獻的 R² 0.92 以上多半來自隨機切分；本專案依國家切分，測試國家完全沒出現在訓練中，R² {MET['R2']:.3f} 是更接近實際應用的估計。</li>
<li><b>模型選擇：</b>隨機森林與多項式迴歸分數較高但不再是線性模型、較難解釋；本專案依作業要求使用線性迴歸，並以 Ridge／Lasso 確認正則化沒有明顯改善。</li>
<li><b>資料清理：</b>文獻用插值與縮尾處理；本專案用同國中位數補值、刪除錯誤過多的 BMI 欄、偏態變數取 log。縮尾處理與非線性項可列為後續改進方向。</li></ul>

<h2>13. 參考資料</h2><ol>
<li>KumarRajarshi. Life Expectancy (WHO). Kaggle. <a href="{D['kaggle_url']}">{D['kaggle_url']}</a></li>
<li>Kaggle. Predict the Flood（Playground Series Season 4, Episode 5）. <a href="{FLOOD_URL}">{FLOOD_URL}</a></li>
<li>Chapman, P. et al. (2000). CRISP-DM 1.0: Step-by-step data mining guide.</li>
<li>Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, 2825–2830.</li>
<li>Seabold, S. &amp; Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python.</li>
<li>版面參考：AIoT-DIC2 Multiple LR Explorer <a href="https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/">https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/</a>；StatsUnlock Multiple Linear Regression Calculator <a href="https://statsunlock.com/multiple-linear-regression-calculator/">https://statsunlock.com/multiple-linear-regression-calculator/</a></li>
<li>第 12 節所列 NotebookLM 來源。</li></ol>
</body></html>"""


def chrome(args):
    prof = tempfile.mkdtemp()
    try:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}", *args],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    finally:
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    (DOCS / "infographic.html").write_text(infographic(), encoding="utf-8")
    (DOCS / "report.html").write_text(report(), encoding="utf-8")
    chrome(["--hide-scrollbars", "--window-size=1080,2800", "--force-device-scale-factor=2", "--virtual-time-budget=5000",
            f"--screenshot={DOCS / 'infographic.png'}", (DOCS / "infographic.html").as_uri()])
    # Trim the empty background below the content (the window is taller than the page).
    from PIL import Image, ImageChops
    im = Image.open(DOCS / "infographic.png").convert("RGB")
    diff = ImageChops.difference(im, Image.new("RGB", im.size, (246, 243, 234)))
    bbox = diff.getbbox()
    if bbox:
        im = im.crop((0, 0, im.width, min(im.height, bbox[3] + 40)))
        im.save(DOCS / "infographic.png", optimize=True)
    # Same infographic as a one-page PDF sized to the content (CSS px = PNG px / 2).
    page_h = int(im.height / 2) + 120   # print layout runs slightly taller than the screenshot
    with tempfile.TemporaryDirectory() as tmp:
        printable = Path(tmp) / "infographic_print.html"
        printable.write_text(infographic().replace(
            "</style>", f"@page {{ size: 1080px {page_h}px; margin: 0; }}\nhtml {{ background: #f6f3ea; }}\nhtml, body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}\n</style>", 1),
            encoding="utf-8")
        chrome(["--no-pdf-header-footer", "--virtual-time-budget=5000", f"--print-to-pdf={DOCS / 'infographic.pdf'}", printable.as_uri()])
    chrome(["--no-pdf-header-footer", "--virtual-time-budget=8000", f"--print-to-pdf={DOCS / 'HW2_report.pdf'}",
            (DOCS / "report.html").as_uri()])
    print("Wrote docs/infographic.html, docs/infographic.png, docs/infographic.pdf, docs/report.html, docs/HW2_report.pdf")
