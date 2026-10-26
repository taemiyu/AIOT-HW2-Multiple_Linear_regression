/*
 * selection.js - dataset handling and feature-selection algorithms for app.html.
 *
 *   Data.parseCSV(text)                 -> { header, rows }
 *   Data.profile(header, rows)          -> column types (numeric / categorical / id)
 *   Data.encoder(columns, records)      -> one-hot (drop first level) design builder
 *   Data.split(n, testFrac, seed)       -> reproducible train/test split
 *   Data.kfold(m, k, seed)              -> reproducible CV folds
 *   Select.run(X, y, folds, opts)       -> 5 algorithms x k = 1..p progressions
 *
 * The algorithms mirror feature_progression_analysis.py; on 50 Startups they
 * reproduce the Python progressions exactly (see README).
 */
(function (root) {
  "use strict";
  const R = root.Regression;

  // ---------------------------------------------------------------- CSV
  function parseCSV(text) {
    text = text.replace(/^﻿/, "");
    const rows = [];
    let row = [], cell = "", q = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) {
        if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; }
        else cell += ch;
      } else if (ch === '"') q = true;
      else if (ch === ",") { row.push(cell); cell = ""; }
      else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && text[i + 1] === "\n") i++;
        row.push(cell); cell = "";
        if (row.length > 1 || row[0].trim() !== "") rows.push(row);
        row = [];
      } else cell += ch;
    }
    if (cell !== "" || row.length) { row.push(cell); if (row.length > 1 || row[0].trim() !== "") rows.push(row); }
    if (rows.length < 2) throw new Error("CSV 至少需要標題列和一列資料。");
    const header = rows[0].map((h, i) => h.trim() || `Unnamed: ${i}`);
    const width = header.length;
    const body = rows.slice(1).filter(r => r.length === width);
    const skipped = rows.length - 1 - body.length;
    return { header, rows: body.map(r => r.map(c => c.trim())), skipped };
  }

  const MISSING = new Set(["", "na", "nan", "null", "none", "?"]);
  const isMissing = v => MISSING.has(String(v).toLowerCase());
  const NUM_RE = /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/;

  function profile(header, rows) {
    const n = rows.length;
    return header.map((name, j) => {
      const vals = rows.map(r => r[j]).filter(v => !isMissing(v));
      const numeric = vals.length > 0 && vals.every(v => NUM_RE.test(v.replace(/,/g, "")));
      const levels = numeric ? null : [...new Set(vals)].sort();
      const col = { name, index: j, type: numeric ? "numeric" : "categorical", levels, missing: n - vals.length, note: "" };
      // Row-number / ID columns carry no signal; exclude them by default.
      const lname = name.toLowerCase();
      if (/^(unnamed: ?\d+|id|index|row|#)$/.test(lname)) { col.type = "id"; col.note = "流水號／ID 欄位"; }
      else if (numeric) {
        const nums = vals.map(Number);
        const seq0 = nums.every((v, i) => v === i), seq1 = nums.every((v, i) => v === i + 1);
        if (vals.length === n && n > 5 && (seq0 || seq1)) { col.type = "id"; col.note = "值為 1, 2, 3… 的流水號"; }
      } else if (levels.length > Math.min(30, Math.max(2, n * 0.5))) { col.type = "id"; col.note = `類別數過多（${levels.length}）`; }
      else if (levels.length < 2) { col.type = "id"; col.note = "只有一種值"; }
      return col;
    });
  }

  // Typed records (only rows complete in the needed columns)
  function records(header, rows, cols, needed) {
    const idx = needed.map(name => header.indexOf(name));
    const kind = Object.fromEntries(cols.map(c => [c.name, c.type]));
    const out = [];
    let dropped = 0;
    rows.forEach((r, i) => {
      if (idx.some(j => isMissing(r[j]))) { dropped++; return; }
      const rec = { __row: i };
      needed.forEach((name, t) => { const v = r[idx[t]]; rec[name] = kind[name] === "numeric" ? Number(v.replace(/,/g, "")) : v; });
      out.push(rec);
    });
    return { records: out, dropped };
  }

  // One-hot encoding, levels sorted, first level dropped (as pandas
  // get_dummies(drop_first=True) / OneHotEncoder(drop="first")).
  function encoder(features, cols) {
    const byName = Object.fromEntries(cols.map(c => [c.name, c]));
    const names = [], group = [], parts = [];
    features.forEach(f => {
      const c = byName[f];
      if (c.type === "numeric") { names.push(f); group.push(f); parts.push(r => [r[f]]); }
      else {
        const lv = c.levels.slice(1);
        lv.forEach(l => { names.push(`${f}_${l}`); group.push(f); });
        parts.push(r => lv.map(l => (r[f] === l ? 1 : 0)));
      }
    });
    return { names, group, baseline: Object.fromEntries(features.filter(f => byName[f].type !== "numeric").map(f => [f, byName[f].levels[0]])), row: r => parts.flatMap(p => p(r)) };
  }

  // Seeded PRNG (mulberry32) for reproducible splits in the browser.
  function rng(seed) {
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function shuffle(arr, seed) {
    const a = arr.slice(), r = rng(seed);
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }
  function split(n, testFrac = 0.2, seed = 42) {
    const p = shuffle([...Array(n).keys()], seed), nt = Math.max(1, Math.round(n * testFrac));
    return { test: p.slice(0, nt).sort((a, b) => a - b), train: p.slice(nt).sort((a, b) => a - b) };
  }
  // Folds as arrays of positions 0..m-1 within the training set.
  function kfold(m, k = 5, seed = 42) {
    const p = shuffle([...Array(m).keys()], seed + 1);
    const folds = Array.from({ length: k }, () => []);
    p.forEach((v, i) => folds[i % k].push(v));
    return folds.map(f => f.sort((a, b) => a - b));
  }

  // ---------------------------------------------------------------- algorithms
  const pick = (X, cols) => X.map(r => cols.map(j => r[j]));

  // Cross-validation fits thousands of subsets on the same folds (exhaustive
  // search: 2^p - 1). Each fold's X'X and X'y (with intercept, columns scaled
  // to unit max for conditioning) are computed once; a subset then only needs
  // a small solve on the matching sub-matrix.
  const foldCache = new WeakMap();
  function foldGrams(X, y, folds) {
    let byFolds = foldCache.get(X);
    if (!byFolds) { byFolds = new WeakMap(); foldCache.set(X, byFolds); }
    let g = byFolds.get(folds);
    if (g && g.y === y) return g;
    const n = X.length, p = X[0].length;
    const s = [1, ...Array.from({ length: p }, (_, j) => Math.max(...X.map(r => Math.abs(r[j]))) || 1)];
    const Z = X.map(r => [1, ...r.map((v, j) => v / s[j + 1])]);
    g = folds.map(f => {
      const inF = new Set(f), G = Array.from({ length: p + 1 }, () => new Float64Array(p + 1)), c = new Float64Array(p + 1);
      for (let i = 0; i < n; i++) {
        if (inF.has(i)) continue;
        const z = Z[i];
        for (let a = 0; a <= p; a++) { c[a] += z[a] * y[i]; for (let b = a; b <= p; b++) G[a][b] += z[a] * z[b]; }
      }
      for (let a = 0; a <= p; a++) for (let b = 0; b < a; b++) G[a][b] = G[b][a];
      return { G, c, test: f };
    });
    g.Z = Z; g.y = y;
    byFolds.set(folds, g);
    return g;
  }
  function solve(A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      if (Math.abs(M[piv][c]) < 1e-12) throw new Error("singular matrix");
      [M[c], M[piv]] = [M[piv], M[c]];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; if (f) for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let v = M[r][n]; for (let j = r + 1; j < n; j++) v -= M[r][j] * x[j]; x[r] = v / M[r][r]; }
    return x;
  }
  function cvFoldPredictions(X, y, cols, folds) {
    const g = foldGrams(X, y, folds), idx = [0, ...cols.map(j => j + 1)];
    return g.map(({ G, c, test }) => {
      const b = solve(idx.map(a => idx.map(bb => G[a][bb])), idx.map(a => c[a]));
      return test.map(i => idx.reduce((sum, a, t) => sum + b[t] * g.Z[i][a], 0));
    });
  }
  function cvRMSE(X, y, cols, folds) { return cvRMSEStats(X, y, cols, folds).mean; }
  function cvRMSEStats(X, y, cols, folds) {
    const preds = cvFoldPredictions(X, y, cols, folds);
    const scores = folds.map((f, k) => Math.sqrt(R.mean(f.map((i, t) => (y[i] - preds[k][t]) ** 2))));
    const mu = R.mean(scores);
    return { mean: mu, std: Math.sqrt(R.mean(scores.map(v => (v - mu) ** 2))) };
  }
  function cvR2(X, y, cols, folds) {
    const preds = cvFoldPredictions(X, y, cols, folds);
    const s = folds.map((f, k) => R.r2Of(f.map(i => y[i]), preds[k]));
    const mu = R.mean(s);
    return { mean: mu, std: Math.sqrt(R.mean(s.map(v => (v - mu) ** 2))), folds: s };
  }

  function standardize(X) {
    const p = X[0].length, n = X.length;
    const mu = Array.from({ length: p }, (_, j) => R.mean(X.map(r => r[j])));
    const sd = Array.from({ length: p }, (_, j) => Math.sqrt(R.mean(X.map(r => (r[j] - mu[j]) ** 2))) || 1);
    return X.map(r => r.map((v, j) => (v - mu[j]) / sd[j]));
  }

  // Coordinate descent on the Gram matrix along a log-spaced penalty path
  // (same grid as sklearn: 300 alphas, eps = 1e-3). Returns features ordered
  // by the step at which they first become non-zero.
  function pathOrder(X, y, l1Ratio) {
    const Xs = standardize(X), n = Xs.length, p = Xs[0].length, ym = R.mean(y), yc = y.map(v => v - ym);
    const G = Array.from({ length: p }, (_, i) => Array.from({ length: p }, (_, j) => Xs.reduce((s, r) => s + r[i] * r[j], 0) / n));
    const c = Array.from({ length: p }, (_, j) => Xs.reduce((s, r, i) => s + r[j] * yc[i], 0) / n);
    const aMax = Math.max(...c.map(Math.abs)) / l1Ratio, N = 300;
    const alphas = Array.from({ length: N }, (_, i) => aMax * Math.pow(10, -3 * i / (N - 1)));
    const w = new Array(p).fill(0), entry = new Array(p).fill(N);
    alphas.forEach((a, t) => {
      for (let it = 0; it < 10000; it++) {
        let maxd = 0, maxw = 0;
        for (let j = 0; j < p; j++) {
          let rho = c[j];
          for (let k = 0; k < p; k++) if (k !== j) rho -= G[j][k] * w[k];
          const z = Math.abs(rho) - a * l1Ratio;
          const nw = z > 0 ? Math.sign(rho) * z / (G[j][j] + a * (1 - l1Ratio)) : 0;
          maxd = Math.max(maxd, Math.abs(nw - w[j])); w[j] = nw; maxw = Math.max(maxw, Math.abs(nw));
        }
        if (maxd <= 1e-12 * Math.max(1, maxw)) break;
      }
      w.forEach((v, j) => { if (entry[j] === N && Math.abs(v) > 1e-8) entry[j] = t; });
    });
    return [...Array(p).keys()].sort((i, j) => entry[i] - entry[j] || Math.abs(w[j]) - Math.abs(w[i]));
  }

  function rfeOrder(X, y) {
    const Xs = standardize(X);
    let remaining = [...Array(X[0].length).keys()];
    const eliminated = [];
    while (remaining.length > 1) {
      const m = R.lite(pick(Xs, remaining), y);
      let worst = 0;
      remaining.forEach((_, i) => { if (Math.abs(m.coef[i + 1]) < Math.abs(m.coef[worst + 1])) worst = i; });
      eliminated.push(remaining[worst]);
      remaining = remaining.filter((_, i) => i !== worst);
    }
    return [remaining[0], ...eliminated.reverse()];
  }

  function forwardSFS(X, y, folds) {
    const p = X[0].length, chosen = [];
    while (chosen.length < p) {
      let best = null, bestS = Infinity;
      for (let j = 0; j < p; j++) {
        if (chosen.includes(j)) continue;
        const s = cvRMSE(X, y, [...chosen, j], folds);
        if (s < bestS - 1e-9) { bestS = s; best = j; }
      }
      chosen.push(best);
    }
    return chosen.map((_, k) => chosen.slice(0, k + 1));
  }

  function* combos(p, k, start = 0, acc = []) {
    if (acc.length === k) { yield acc.slice(); return; }
    for (let j = start; j < p; j++) { acc.push(j); yield* combos(p, k, j + 1, acc); acc.pop(); }
  }
  function exhaustive(X, y, folds) {
    const p = X[0].length, out = [];
    for (let k = 1; k <= p; k++) {
      let best = null, bestS = Infinity;
      for (const s of combos(p, k)) { const v = cvRMSE(X, y, s, folds); if (v < bestS - 1e-9) { bestS = v; best = s; } }
      out.push(best);
    }
    return out;
  }

  const METHODS = [
    { key: "lasso", title: "Lasso", sub: "L1 Ranking" },
    { key: "sfs", title: "Forward SFS", sub: "Greedy" },
    { key: "rfe", title: "RFE", sub: "Backward" },
    { key: "enet", title: "ElasticNet", sub: "L1 + L2" },
    { key: "exhaustive", title: "Exhaustive Search", sub: "Best Subset" },
  ];
  const prefixes = order => order.map((_, k) => order.slice(0, k + 1));

  // Budget guard: exhaustive search fits 2^p - 1 subsets x folds.
  function exhaustiveAllowed(p, n) { return p <= 14 && (Math.pow(2, p) - 1) * n <= 4e6; }

  function run(Xtr, ytr, Xte, yte, folds) {
    const p = Xtr[0].length, out = {};
    const subsets = {
      lasso: () => prefixes(pathOrder(Xtr, ytr, 1)),
      sfs: () => forwardSFS(Xtr, ytr, folds),
      rfe: () => prefixes(rfeOrder(Xtr, ytr)),
      enet: () => prefixes(pathOrder(Xtr, ytr, 0.5)),
      exhaustive: () => (exhaustiveAllowed(p, Xtr.length) ? exhaustive(Xtr, ytr, folds) : null),
    };
    METHODS.forEach(({ key }) => {
      const subs = subsets[key]();
      if (!subs) { out[key] = null; return; }
      out[key] = subs.map(cols => {
        const m = R.lite(pick(Xtr, cols), ytr), pred = pick(Xte, cols).map(x => m.predict(x));
        const err = yte.map((v, i) => v - pred[i]);
        const cv = cvRMSEStats(Xtr, ytr, cols, folds);
        return { k: cols.length, cols, cvRmse: cv.mean, cvSd: cv.std, testRmse: Math.sqrt(R.mean(err.map(e => e * e))), testR2: R.r2Of(yte, pred) };
      });
    });
    return out;
  }

  // k with the lowest CV RMSE, and the smallest k within one fold-SD of it.
  function bestK(rows) {
    const best = rows.reduce((a, b) => (b.cvRmse < a.cvRmse ? b : a));
    const parsimonious = rows.find(r => r.cvRmse <= best.cvRmse + best.cvSd);
    return { cvBest: best.k, oneSd: parsimonious.k };
  }

  root.Data = { parseCSV, profile, records, encoder, split, kfold, isMissing };
  root.Select = { METHODS, run, bestK, cvRMSE, cvR2, exhaustiveAllowed, pick };
})(typeof window !== "undefined" ? window : globalThis);
