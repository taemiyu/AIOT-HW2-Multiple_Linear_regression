/*
 * regression.js - ordinary least squares and the statistics around it, in
 * plain JavaScript, for app.html. Validated against numpy/scipy on the
 * 50 Startups data (see README).
 *
 *   Regression.fit(X, y, alpha)  X: array of rows (no intercept column)
 *   Regression.dist              t / F / chi-square CDFs and the t quantile
 */
(function (root) {
  "use strict";

  // ---------- special functions ----------
  function lnGamma(z) {
    const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
      771.32342877765313, -176.61502916214059, 12.507343278686905,
      -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
    z -= 1;
    let x = c[0];
    for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  // Regularized incomplete beta I_x(a, b), continued fraction (Numerical Recipes).
  function betacf(x, a, b) {
    const MAXIT = 300, EPS = 1e-15, FPMIN = 1e-300;
    let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }
  function ibeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(x, a, b) / a : 1 - bt * betacf(1 - x, b, a) / b;
  }
  // Regularized lower incomplete gamma P(a, x).
  function igammaP(a, x) {
    if (x <= 0) return 0;
    if (x < a + 1) {
      let sum = 1 / a, del = sum, ap = a;
      for (let n = 0; n < 500; n++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
      return sum * Math.exp(-x + a * Math.log(x) - lnGamma(a));
    }
    let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (let i = 1; i < 500; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300;
      c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - lnGamma(a)) * h;
  }

  const dist = {
    // Two-sided tail helpers return upper-tail probabilities directly to
    // keep precision for tiny p-values.
    tSF(t, df) { const x = df / (df + t * t); const tail = 0.5 * ibeta(x, df / 2, 0.5); return t >= 0 ? tail : 1 - tail; },
    tCDF(t, df) { return 1 - dist.tSF(t, df); },
    tInv(p, df) {              // quantile, by bisection on the CDF
      let lo = -1e3, hi = 1e3;
      for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (dist.tCDF(mid, df) < p) lo = mid; else hi = mid; }
      return (lo + hi) / 2;
    },
    fSF(f, d1, d2) { if (f <= 0) return 1; return ibeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2); },
    chi2SF(x, k) { return x <= 0 ? 1 : 1 - igammaP(k / 2, x / 2); },
  };

  // ---------- linear algebra ----------
  function invert(A) {
    const n = A.length, M = A.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (Math.abs(M[piv][c]) < 1e-12) throw new Error("singular matrix");
      [M[c], M[piv]] = [M[piv], M[c]];
      const d = M[c][c];
      for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c];
        if (f) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j];
      }
    }
    return M.map(r => r.slice(n));
  }

  // OLS core on a design matrix WITH intercept column. Columns are scaled
  // to unit max before inverting (raw spends are ~1e5, so X'X would be
  // badly conditioned), then the results are mapped back.
  function core(Xd, y) {
    const n = Xd.length, p = Xd[0].length;
    const s = Array.from({ length: p }, (_, j) => Math.max(...Xd.map(r => Math.abs(r[j]))) || 1);
    const Z = Xd.map(r => r.map((v, j) => v / s[j]));
    const ZtZ = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => Z.reduce((acc, r) => acc + r[i] * r[j], 0)));
    const inv = invert(ZtZ);
    const Zty = Array.from({ length: p }, (_, i) => Z.reduce((acc, r, k) => acc + r[i] * y[k], 0));
    const g = inv.map(row => row.reduce((acc, v, j) => acc + v * Zty[j], 0));
    const b = g.map((v, j) => v / s[j]);
    const XtXi = inv.map((row, i) => row.map((v, j) => v / (s[i] * s[j])));
    const yhat = Xd.map(r => r.reduce((acc, v, j) => acc + v * b[j], 0));
    return { b, XtXi, yhat, resid: y.map((v, i) => v - yhat[i]), n, p };
  }
  const withIntercept = X => X.map(r => [1, ...r]);
  const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
  const sd = a => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };
  const r2Of = (y, fit) => { const m = mean(y); let sse = 0, sst = 0; y.forEach((v, i) => { sse += (v - fit[i]) ** 2; sst += (v - m) ** 2; }); return 1 - sse / sst; };

  function fit(X, y, alpha = 0.05) {
    const k = X[0].length, Xd = withIntercept(X);
    const c = core(Xd, y), n = c.n, dfe = n - k - 1;
    const sse = c.resid.reduce((s, e) => s + e * e, 0);
    const ym = mean(y), sst = y.reduce((s, v) => s + (v - ym) ** 2, 0), ssr = sst - sse;
    const mse = sse / dfe, tc = dist.tInv(1 - alpha / 2, dfe);
    const se = c.XtXi.map((row, i) => Math.sqrt(mse * row[i]));
    const t = c.b.map((b, i) => b / se[i]);
    const p = t.map(v => Math.min(1, 2 * dist.tSF(Math.abs(v), dfe)));
    const F = (ssr / k) / mse, Fp = dist.fSF(F, k, dfe);
    const r2 = 1 - sse / sst, adjR2 = 1 - (1 - r2) * (n - 1) / dfe;

    const cols = Array.from({ length: k }, (_, j) => X.map(r => r[j]));
    const vif = cols.map((col, j) => {
      if (k === 1) return 1;
      const others = X.map(r => r.filter((_, i) => i !== j));
      const r2j = r2Of(col, core(withIntercept(others), col).yhat);
      return 1 / (1 - r2j);
    });
    const sy = sd(y), stdBeta = cols.map((col, j) => c.b[j + 1] * sd(col) / sy);

    // Breusch-Pagan (Koenker studentized): n * R^2 of e^2 on X
    const e2 = c.resid.map(e => e * e);
    const bp = n * r2Of(e2, core(Xd, e2).yhat), bpP = dist.chi2SF(bp, k);
    // Jarque-Bera on residuals (population moments)
    const em = mean(c.resid);
    const m2 = mean(c.resid.map(e => (e - em) ** 2)), m3 = mean(c.resid.map(e => (e - em) ** 3)), m4 = mean(c.resid.map(e => (e - em) ** 4));
    const skew = m3 / m2 ** 1.5, kurt = m4 / m2 ** 2;
    const jb = n / 6 * (skew ** 2 + (kurt - 3) ** 2 / 4), jbP = dist.chi2SF(jb, 2);
    // Durbin-Watson (rows in dataset order)
    let dwNum = 0;
    for (let i = 1; i < n; i++) dwNum += (c.resid[i] - c.resid[i - 1]) ** 2;
    const dw = dwNum / sse;
    // Ramsey RESET with yhat^2 and yhat^3
    // (Not computable when yhat takes only a few values, e.g. State alone:
    // yhat^2 and yhat^3 are then collinear with the dummies.)
    let resetF = NaN, resetP = NaN;
    try {
      const z = c.yhat.map(v => v / 1e5);
      const cu = core(X.map((r, i) => [1, ...r, z[i] ** 2, z[i] ** 3]), y);
      const sseU = cu.resid.reduce((s, e) => s + e * e, 0), dfu = n - k - 3;
      resetF = ((sse - sseU) / 2) / (sseU / dfu);
      resetP = dist.fSF(resetF, 2, dfu);
    } catch (e) { /* leave as NaN */ }
    // Leverage and Cook's distance
    const h = Xd.map(r => r.reduce((acc, v, i) => acc + v * r.reduce((a2, w, j) => a2 + c.XtXi[i][j] * w, 0), 0));
    const cook = c.resid.map((e, i) => e * e / ((k + 1) * mse) * h[i] / (1 - h[i]) ** 2);

    function predict(x) { return c.b[0] + x.reduce((s, v, j) => s + v * c.b[j + 1], 0); }
    function interval(x) {
      const x0 = [1, ...x];
      const q = x0.reduce((acc, v, i) => acc + v * x0.reduce((a2, w, j) => a2 + c.XtXi[i][j] * w, 0), 0);
      const yhat = predict(x), half = tc * Math.sqrt(mse * (1 + q)), halfMean = tc * Math.sqrt(mse * q);
      return { yhat, piLo: yhat - half, piHi: yhat + half, ciLo: yhat - halfMean, ciHi: yhat + halfMean };
    }

    return {
      n, k, dfe, alpha, tCrit: tc,
      coef: c.b, se, t, p, ciLo: c.b.map((b, i) => b - tc * se[i]), ciHi: c.b.map((b, i) => b + tc * se[i]),
      vif, stdBeta, sse, ssr, sst, mse, F, Fp, r2, adjR2,
      fitted: c.yhat, resid: c.resid, leverage: h, cook,
      tests: { bp, bpP, jb, jbP, skew, kurt, dw, resetF, resetP },
      predict, interval,
    };
  }

  // Coefficients only - for cross-validation loops where the full set of
  // statistics in fit() would be wasted work.
  function lite(X, y) {
    const c = core(withIntercept(X), y);
    return { coef: c.b, predict: x => c.b[0] + x.reduce((s, v, j) => s + v * c.b[j + 1], 0) };
  }

  root.Regression = { fit, lite, dist, r2Of, mean, sd };
  if (typeof module !== "undefined") module.exports = root.Regression;
})(typeof window !== "undefined" ? window : globalThis);
