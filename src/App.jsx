import { useState, useEffect, useRef, useCallback, useMemo } from "react";

function useXLSX() {
  const [ready, setReady] = useState(!!window.XLSX);
  useEffect(() => {
    if (window.XLSX) {
      setReady(true);
      return;
    }
    const s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js";
    s.onload = () => setReady(true);
    document.head.appendChild(s);
  }, []);
  return ready;
}

const C = {
  navy: "#0D1B2A",
  ink: "#1C2B3A",
  mid: "#274060",
  blue: "#2C6FAC",
  pos: "#C0392B",
  neg: "#1A5F96",
  posShap: "#C0392B",
  negShap: "#1A5F96",
  bg: "#F2F5F9",
  paper: "#FFFFFF",
  gray: "#5D6D7E",
  light: "#E8EDF3",
  border: "#CDD5DF",
  accent: "#2980B9"
};

const TAB_COLS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78"];
const dlBtn = { marginTop: 6, padding: "4px 14px", fontSize: 12, background: C.navy, color: "#fff", border: "none", borderRadius: 4, cursor: "pointer", fontFamily: "monospace", letterSpacing: 0.3 };

const cardStyle = {
  background: C.paper,
  borderRadius: 8,
  padding: "16px 18px",
  boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
  marginBottom: 14
};

const vMean = (a) => a.reduce((s, v) => s + v, 0) / Math.max(a.length, 1);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const cleanModelLabel = (name) => (name || "").replace(/-like\b/gi, "").trim();

function vStd(a, m) {
  const mn = m ?? vMean(a);
  return Math.sqrt(a.reduce((s, v) => s + (v - mn) * (v - mn), 0) / Math.max(a.length, 1)) || 1;
}

// ── Matrix Math ──────────────────────────────────────────────────────────────
const mmul = (A, B) => A.map(rA => B[0].map((_, j) => rA.reduce((s, v, k) => s + v * B[k][j], 0)));
const mT = A => A[0].map((_, j) => A.map(r => r[j]));
function minv(M) {
  const n = M.length, A = M.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => i === j ? 1 : 0)]);
  for (let c = 0; c < n; c++) {
    let mx = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[mx][c])) mx = r;
    [A[c], A[mx]] = [A[mx], A[c]];
    const p = A[c][c] || 1e-14; for (let j = 0; j < 2 * n; j++) A[c][j] /= p;
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j]; }
  }
  return A.map(r => r.slice(n));
}

// ── Color Functions ──────────────────────────────────────────────────────────
function viridis(t) {
  const s = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
  t = Math.max(0, Math.min(1, t)); const idx = t * (s.length - 1), lo = Math.floor(idx), hi = Math.min(lo + 1, s.length - 1), f = idx - lo;
  return `rgb(${Math.round(s[lo][0] + f * (s[hi][0] - s[lo][0]))},${Math.round(s[lo][1] + f * (s[hi][1] - s[lo][1]))},${Math.round(s[lo][2] + f * (s[hi][2] - s[lo][2]))})`;
}

function rwb(t) {
  t = Math.max(0, Math.min(1, t));
  if (t < 0.5) { const f = t / 0.5; return `rgb(${Math.round(31 + f * 224)},${Math.round(119 + f * 136)},${Math.round(180 + f * 75)})`; }
  else { const f = (t - 0.5) / 0.5; return `rgb(255,${Math.round(255 - f * 216)},${Math.round(255 - f * 215)})`; }
}

const normVal = (v, mn, mx) => mx === mn ? 0.5 : (v - mn) / (mx - mn);

function seeded(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function shuffleIdx(n, seed = 42) {
  const arr = Array.from({ length: n }, (_, i) => i);
  const rnd = seeded(seed);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

function subsetRows(X, ids) {
  return ids.map((i) => X[i]);
}

function subsetVals(y, ids) {
  return ids.map((i) => y[i]);
}

function makeKFolds(n, k, shuffle = true, seed = 42) {
  const idx = shuffle ? shuffleIdx(n, seed) : Array.from({ length: n }, (_, i) => i);
  const folds = [];
  const foldSize = Math.floor(n / k);
  let start = 0;
  for (let i = 0; i < k; i++) {
    const end = i === k - 1 ? n : start + foldSize;
    const test = idx.slice(start, end);
    const testSet = new Set(test);
    const train = idx.filter((id) => !testSet.has(id));
    folds.push({ train, test });
    start = end;
  }
  return folds;
}

function normalizeMatrix(X) {
  const p = X[0].length;
  const means = Array.from({ length: p }, (_, j) => vMean(X.map((r) => r[j])));
  const stds = Array.from({ length: p }, (_, j) => vStd(X.map((r) => r[j]), means[j]));
  const Xn = X.map((r) => r.map((v, j) => (v - means[j]) / stds[j]));
  return { Xn, means, stds };
}

function distance2(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

function quantileThresholds(vals, bins = 14) {
  const sorted = [...vals].sort((a, b) => a - b);
  const out = [];
  for (let b = 1; b < bins; b++) {
    const q = b / bins;
    const pos = Math.floor(q * (sorted.length - 1));
    out.push(sorted[pos]);
  }
  return [...new Set(out)];
}

function fitDecisionStump(X, y, opts = {}) {
  const n = X.length;
  const p = X[0].length;
  const bins = clamp(Math.round(opts.bins ?? 14), 6, 40);
  const weights = opts.weights || Array.from({ length: n }, () => 1 / n);
  const featurePool = opts.featurePool || Array.from({ length: p }, (_, j) => j);

  let best = {
    feature: 0,
    threshold: 0,
    leftValue: vMean(y),
    rightValue: vMean(y),
    loss: Number.POSITIVE_INFINITY
  };

  for (const j of featurePool) {
    const col = X.map((r) => r[j]);
    const ths = quantileThresholds(col, bins);
    if (ths.length === 0) continue;

    for (const t of ths) {
      let wl = 0;
      let wr = 0;
      let yl = 0;
      let yr = 0;
      for (let i = 0; i < n; i++) {
        const w = weights[i];
        if (X[i][j] <= t) {
          wl += w;
          yl += w * y[i];
        } else {
          wr += w;
          yr += w * y[i];
        }
      }
      if (wl <= 1e-12 || wr <= 1e-12) continue;

      const lv = yl / wl;
      const rv = yr / wr;
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const pred = X[i][j] <= t ? lv : rv;
        const e = y[i] - pred;
        loss += weights[i] * e * e;
      }
      if (loss < best.loss) best = { feature: j, threshold: t, leftValue: lv, rightValue: rv, loss };
    }
  }

  return best;
}

function stumpPredictRow(stump, row) {
  return row[stump.feature] <= stump.threshold ? stump.leftValue : stump.rightValue;
}

function stumpPredict(stump, X) {
  return X.map((row) => stumpPredictRow(stump, row));
}

function trainAdaBoostRegressor(X, y, params = {}, seed = 42) {
  const n = X.length;
  const nEstimators = clamp(Math.round(params.nEstimators ?? 60), 5, 220);
  const learningRate = clamp(params.learningRate ?? 0.2, 0.005, 1.5);
  const bins = clamp(Math.round(params.bins ?? 14), 6, 40);

  let w = Array.from({ length: n }, () => 1 / n);
  const learners = [];
  const alphas = [];

  for (let m = 0; m < nEstimators; m++) {
    const stump = fitDecisionStump(X, y, { weights: w, bins, seed: seed + m });
    const pred = stumpPredict(stump, X);
    const errs = pred.map((v, i) => Math.abs(y[i] - v));
    const maxErr = Math.max(...errs, 1e-9);
    const normErr = errs.map((e) => e / maxErr);

    let weightedErr = 0;
    for (let i = 0; i < n; i++) weightedErr += w[i] * normErr[i];
    weightedErr = clamp(weightedErr, 1e-9, 0.499);

    const beta = weightedErr / (1 - weightedErr);
    const alpha = learningRate * Math.log((1 - weightedErr) / weightedErr);
    for (let i = 0; i < n; i++) w[i] *= Math.pow(beta, 1 - normErr[i]);
    const ws = w.reduce((s, v) => s + v, 0) || 1;
    w = w.map((v) => v / ws);

    learners.push(stump);
    alphas.push(alpha);
  }

  const alphaSum = alphas.reduce((s, v) => s + Math.abs(v), 0) || 1;
  return {
    type: "AdaBoost",
    params: { nEstimators, learningRate, bins },
    predictRow: (row) => {
      let s = 0;
      for (let i = 0; i < learners.length; i++) s += alphas[i] * stumpPredictRow(learners[i], row);
      return s / alphaSum;
    },
    predict: (Xnew) => Xnew.map((row) => {
      let s = 0;
      for (let i = 0; i < learners.length; i++) s += alphas[i] * stumpPredictRow(learners[i], row);
      return s / alphaSum;
    })
  };
}

function trainXGBoostLike(X, y, params = {}, seed = 42) {
  const n = X.length;
  const nEstimators = clamp(Math.round(params.nEstimators ?? 100), 10, 300);
  const learningRate = clamp(params.learningRate ?? 0.08, 0.005, 0.8);
  const bins = clamp(Math.round(params.bins ?? 16), 6, 50);
  const subsample = clamp(params.subsample ?? 0.8, 0.4, 1);

  const base = vMean(y);
  let pred = Array.from({ length: n }, () => base);
  const learners = [];

  for (let m = 0; m < nEstimators; m++) {
    const rnd = seeded(seed + m * 17 + 11);
    const ids = [];
    for (let i = 0; i < n; i++) if (rnd() < subsample) ids.push(i);
    if (ids.length < 4) continue;

    const residual = ids.map((i) => y[i] - pred[i]);
    const Xsub = subsetRows(X, ids);
    const stump = fitDecisionStump(Xsub, residual, { bins });
    const fullPred = stumpPredict(stump, X);
    for (let i = 0; i < n; i++) pred[i] += learningRate * fullPred[i];
    learners.push(stump);
  }

  return {
    type: "XGBoost-like",
    params: { nEstimators, learningRate, bins, subsample },
    predictRow: (row) => {
      let v = base;
      for (const st of learners) v += learningRate * stumpPredictRow(st, row);
      return v;
    },
    predict: (Xnew) => Xnew.map((row) => {
      let v = base;
      for (const st of learners) v += learningRate * stumpPredictRow(st, row);
      return v;
    })
  };
}

function trainCatBoostLike(X, y, params = {}, seed = 42) {
  const n = X.length;
  const p = X[0].length;
  const nEstimators = clamp(Math.round(params.nEstimators ?? 110), 10, 320);
  const learningRate = clamp(params.learningRate ?? 0.06, 0.005, 0.7);
  const bins = clamp(Math.round(params.bins ?? 14), 6, 40);
  const featureSubsample = clamp(params.featureSubsample ?? 0.7, 0.2, 1);

  const base = vMean(y);
  let pred = Array.from({ length: n }, () => base);
  const learners = [];

  for (let m = 0; m < nEstimators; m++) {
    const rnd = seeded(seed + m * 29 + 7);
    const permutation = shuffleIdx(n, seed + m * 101 + 3);

    const subsetCount = Math.max(1, Math.round(p * featureSubsample));
    const allFeat = Array.from({ length: p }, (_, j) => j);
    for (let j = allFeat.length - 1; j > 0; j--) {
      const r = Math.floor(rnd() * (j + 1));
      [allFeat[j], allFeat[r]] = [allFeat[r], allFeat[j]];
    }
    const featPool = allFeat.slice(0, subsetCount);

    const trainCount = Math.max(4, Math.floor(0.8 * n));
    const ids = permutation.slice(0, trainCount);
    const Xsub = subsetRows(X, ids);
    const residual = ids.map((i) => y[i] - pred[i]);

    const stump = fitDecisionStump(Xsub, residual, { bins, featurePool: featPool });
    const fullPred = stumpPredict(stump, X);
    for (let i = 0; i < n; i++) pred[i] += learningRate * fullPred[i];
    learners.push(stump);
  }

  return {
    type: "CatBoost-like",
    params: { nEstimators, learningRate, bins, featureSubsample },
    predictRow: (row) => {
      let v = base;
      for (const st of learners) v += learningRate * stumpPredictRow(st, row);
      return v;
    },
    predict: (Xnew) => Xnew.map((row) => {
      let v = base;
      for (const st of learners) v += learningRate * stumpPredictRow(st, row);
      return v;
    })
  };
}

function calcMetrics(y, yp) {
  const n = y.length;
  const ym = vMean(y);
  let sse = 0;
  let sae = 0;
  let mape = 0;
  let smape = 0;
  let mbe = 0;

  for (let i = 0; i < n; i++) {
    const e = y[i] - yp[i];
    const ae = Math.abs(e);
    sse += e * e;
    sae += ae;
    mbe += yp[i] - y[i];
    const den = Math.abs(y[i]) < 1e-9 ? 1 : Math.abs(y[i]);
    mape += (ae / den) * 100;
    const sden = Math.abs(y[i]) + Math.abs(yp[i]) + 1e-9;
    smape += (2 * ae / sden) * 100;
  }

  const rmse = Math.sqrt(sse / n);
  const mae = sae / n;
  const mse = sse / n;
  const ssTot = y.reduce((s, v) => s + (v - ym) * (v - ym), 0);
  const r2 = 1 - sse / Math.max(ssTot, 1e-9);
  const yVar = y.reduce((s, v) => s + (v - ym) * (v - ym), 0) / Math.max(n, 1);
  const varErr = yp.reduce((s, v, i) => s + (v - y[i]) * (v - y[i]), 0) / Math.max(n, 1);
  const vaf = 100 * (1 - varErr / Math.max(yVar, 1e-9));

  return {
    RMSE: rmse,
    MAE: mae,
    MAD: mae,
    MSE: mse,
    R2: r2,
    MAPE: mape / n,
    sMAPE: smape / n,
    MBE: mbe / n,
    VAF: vaf
  };
}

function runKFold(trainFn, X, y, cvOpts) {
  const n = y.length;
  const folds = makeKFolds(n, cvOpts.k, cvOpts.shuffle, cvOpts.seed);
  const oof = Array.from({ length: n }, () => null);

  for (let f = 0; f < folds.length; f++) {
    const tr = folds[f].train;
    const te = folds[f].test;
    const model = trainFn(subsetRows(X, tr), subsetVals(y, tr), cvOpts.seed + f);
    const pred = model.predict(subsetRows(X, te));
    te.forEach((id, i) => {
      oof[id] = pred[i];
    });
  }

  const validY = [];
  const validP = [];
  for (let i = 0; i < n; i++) {
    if (oof[i] !== null && Number.isFinite(oof[i])) {
      validY.push(y[i]);
      validP.push(oof[i]);
    }
  }
  return { oof, metrics: calcMetrics(validY, validP) };
}

function decodeParams(vec, spec) {
  const out = {};
  spec.forEach((s, i) => {
    const v = s.min + vec[i] * (s.max - s.min);
    out[s.name] = s.int ? Math.round(v) : v;
  });
  return out;
}

function differentialEvolution({ objective, dims, popSize = 12, generations = 10, F = 0.7, CR = 0.9, seed = 42 }) {
  const rnd = seeded(seed);
  let pop = Array.from({ length: popSize }, () => Array.from({ length: dims }, () => rnd()));
  let scores = pop.map((v) => objective(v));

  for (let g = 0; g < generations; g++) {
    for (let i = 0; i < popSize; i++) {
      const pool = Array.from({ length: popSize }, (_, k) => k).filter((k) => k !== i);
      for (let k = pool.length - 1; k > 0; k--) {
        const j = Math.floor(rnd() * (k + 1));
        [pool[k], pool[j]] = [pool[j], pool[k]];
      }
      const [a, b, c] = pool.slice(0, 3);
      const trial = [];
      const jRand = Math.floor(rnd() * dims);
      for (let d = 0; d < dims; d++) {
        if (rnd() < CR || d === jRand) trial[d] = clamp(pop[a][d] + F * (pop[b][d] - pop[c][d]), 0, 1);
        else trial[d] = pop[i][d];
      }

      const sTrial = objective(trial);
      if (sTrial <= scores[i]) {
        pop[i] = trial;
        scores[i] = sTrial;
      }
    }
  }

  let bestI = 0;
  for (let i = 1; i < popSize; i++) if (scores[i] < scores[bestI]) bestI = i;
  return { vector: pop[bestI], score: scores[bestI] };
}

function particleSwarmOptimization({ objective, dims, popSize = 12, generations = 10, w = 0.72, c1 = 1.5, c2 = 1.5, seed = 42 }) {
  const rnd = seeded(seed);
  let pos = Array.from({ length: popSize }, () => Array.from({ length: dims }, () => rnd()));
  let vel = Array.from({ length: popSize }, () => Array.from({ length: dims }, () => (rnd() - 0.5) * 0.2));
  let pBest = pos.map((p) => [...p]);
  let pBestScore = pos.map((p) => objective(p));
  let gBestI = 0;
  for (let i = 1; i < popSize; i++) if (pBestScore[i] < pBestScore[gBestI]) gBestI = i;
  let gBest = [...pBest[gBestI]];
  let gBestScore = pBestScore[gBestI];

  for (let g = 0; g < generations; g++) {
    for (let i = 0; i < popSize; i++) {
      for (let d = 0; d < dims; d++) {
        const r1 = rnd();
        const r2 = rnd();
        vel[i][d] = w * vel[i][d] + c1 * r1 * (pBest[i][d] - pos[i][d]) + c2 * r2 * (gBest[d] - pos[i][d]);
        pos[i][d] = clamp(pos[i][d] + vel[i][d], 0, 1);
      }
      const s = objective(pos[i]);
      if (s < pBestScore[i]) {
        pBest[i] = [...pos[i]];
        pBestScore[i] = s;
      }
      if (s < gBestScore) {
        gBest = [...pos[i]];
        gBestScore = s;
      }
    }
  }

  return { vector: gBest, score: gBestScore };
}

function geneticAlgorithm({ objective, dims, popSize = 12, generations = 10, mutationRate = 0.1, crossoverRate = 0.85, seed = 42 }) {
  const rnd = seeded(seed);
  const tournament = (population, scores, k = 3) => {
    let best = null;
    for (let i = 0; i < k; i++) {
      const idx = Math.floor(rnd() * population.length);
      if (best === null || scores[idx] < scores[best]) best = idx;
    }
    return population[best];
  };

  let pop = Array.from({ length: popSize }, () => Array.from({ length: dims }, () => rnd()));
  let scores = pop.map((v) => objective(v));

  for (let g = 0; g < generations; g++) {
    let eliteI = 0;
    for (let i = 1; i < popSize; i++) if (scores[i] < scores[eliteI]) eliteI = i;
    const next = [[...pop[eliteI]]];

    while (next.length < popSize) {
      const p1 = tournament(pop, scores);
      const p2 = tournament(pop, scores);
      const child = [];
      for (let d = 0; d < dims; d++) {
        let gene = rnd() < crossoverRate ? (rnd() < 0.5 ? p1[d] : p2[d]) : p1[d];
        if (rnd() < mutationRate) gene = clamp(gene + (rnd() - 0.5) * 0.4, 0, 1);
        child.push(gene);
      }
      next.push(child);
    }
    pop = next;
    scores = pop.map((v) => objective(v));
  }

  let bestI = 0;
  for (let i = 1; i < popSize; i++) if (scores[i] < scores[bestI]) bestI = i;
  return { vector: pop[bestI], score: scores[bestI] };
}

function optimizeHyperparams({ algorithm, objective, dims, popSize, generations, paramA, paramB, seed }) {
  if (algorithm === "pso") {
    return particleSwarmOptimization({
      objective,
      dims,
      popSize,
      generations,
      w: clamp(paramA, 0.1, 1),
      c1: clamp(paramB, 0.3, 2.5),
      c2: clamp(paramB, 0.3, 2.5),
      seed
    });
  }

  if (algorithm === "ga") {
    return geneticAlgorithm({
      objective,
      dims,
      popSize,
      generations,
      mutationRate: clamp(paramA, 0.01, 0.6),
      crossoverRate: clamp(paramB, 0.2, 1),
      seed
    });
  }

  return differentialEvolution({
    objective,
    dims,
    popSize,
    generations,
    F: clamp(paramA, 0.1, 1.2),
    CR: clamp(paramB, 0.1, 1),
    seed
  });
}

function computeApproxShap(model, X) {
  const n = X.length;
  const p = X[0].length;
  const basePoint = X[0].map((_, j) => vMean(X.map((r) => r[j])));
  const shapVals = Array.from({ length: n }, () => Array.from({ length: p }, () => 0));

  for (let i = 0; i < n; i++) {
    const row = X[i];
    const full = model.predictRow(row);
    for (let j = 0; j < p; j++) {
      const masked = [...row];
      masked[j] = basePoint[j];
      const reduced = model.predictRow(masked);
      shapVals[i][j] = full - reduced;
    }
  }

  const meanAbs = Array.from({ length: p }, (_, j) => vMean(shapVals.map((r) => Math.abs(r[j]))));
  const meanSigned = Array.from({ length: p }, (_, j) => vMean(shapVals.map((r) => r[j])));
  const orderImp = [...meanAbs.map((v, i) => ({ v, i }))].sort((a, b) => b.v - a.v).map((d) => d.i);
  return { shapVals, meanAbs, meanSigned, orderImp };
}

function profileNumericData(rows, columns) {
  const stats = columns.map((c) => {
    const vals = rows.map((r) => r[c]).filter((v) => Number.isFinite(v));
    const sorted = [...vals].sort((a, b) => a - b);
    const n = vals.length;
    const mean = vMean(vals);
    const std = vStd(vals, mean);
    const q1 = sorted[Math.floor(0.25 * (n - 1))] ?? 0;
    const median = sorted[Math.floor(0.5 * (n - 1))] ?? 0;
    const q3 = sorted[Math.floor(0.75 * (n - 1))] ?? 0;
    return {
      feature: c,
      count: n,
      missing: rows.length - n,
      mean,
      std,
      min: sorted[0] ?? 0,
      q1,
      median,
      q3,
      max: sorted[n - 1] ?? 0
    };
  });

  const p = columns.length;
  const corr = Array.from({ length: p }, () => Array.from({ length: p }, () => 0));
  for (let i = 0; i < p; i++) {
    corr[i][i] = 1;
    for (let j = i + 1; j < p; j++) {
      const x = rows.map((r) => r[columns[i]]);
      const y = rows.map((r) => r[columns[j]]);
      const xm = vMean(x);
      const ym = vMean(y);
      let num = 0;
      let dx = 0;
      let dy = 0;
      for (let k = 0; k < x.length; k++) {
        const a = x[k] - xm;
        const b = y[k] - ym;
        num += a * b;
        dx += a * a;
        dy += b * b;
      }
      const c = num / Math.sqrt((dx || 1e-9) * (dy || 1e-9));
      corr[i][j] = c;
      corr[j][i] = c;
    }
  }
  return { stats, corr, columns };
}

function runKMeans(X, k, maxIter = 80, seed = 42) {
  const n = X.length;
  const p = X[0].length;
  const rndIdx = shuffleIdx(n, seed).slice(0, k);
  let centroids = rndIdx.map((i) => [...X[i]]);
  let labels = Array.from({ length: n }, () => 0);

  for (let it = 0; it < maxIter; it++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      let best = 0;
      let bestD = Number.POSITIVE_INFINITY;
      for (let c = 0; c < k; c++) {
        const d = distance2(X[i], centroids[c]);
        if (d < bestD) {
          bestD = d;
          best = c;
        }
      }
      if (labels[i] !== best) changed = true;
      labels[i] = best;
    }

    const sums = Array.from({ length: k }, () => Array.from({ length: p }, () => 0));
    const cnt = Array.from({ length: k }, () => 0);
    for (let i = 0; i < n; i++) {
      cnt[labels[i]] += 1;
      for (let j = 0; j < p; j++) sums[labels[i]][j] += X[i][j];
    }

    for (let c = 0; c < k; c++) {
      if (cnt[c] === 0) {
        centroids[c] = [...X[Math.floor(seeded(seed + it + c)() * n)]];
      } else {
        centroids[c] = centroids[c].map((_, j) => sums[c][j] / cnt[c]);
      }
    }

    if (!changed) break;
  }

  let inertia = 0;
  for (let i = 0; i < n; i++) inertia += distance2(X[i], centroids[labels[i]]);
  return { labels, centroids, inertia };
}

function runFuzzyCMeans(X, k, maxIter = 80, m = 2, seed = 42) {
  const n = X.length;
  const p = X[0].length;
  const rnd = seeded(seed);

  let U = Array.from({ length: n }, () => {
    const row = Array.from({ length: k }, () => rnd() + 1e-6);
    const s = row.reduce((a, b) => a + b, 0);
    return row.map((v) => v / s);
  });

  let centroids = Array.from({ length: k }, () => Array.from({ length: p }, () => 0));

  for (let it = 0; it < maxIter; it++) {
    for (let c = 0; c < k; c++) {
      let den = 0;
      const num = Array.from({ length: p }, () => 0);
      for (let i = 0; i < n; i++) {
        const u = Math.pow(U[i][c], m);
        den += u;
        for (let j = 0; j < p; j++) num[j] += u * X[i][j];
      }
      centroids[c] = num.map((v) => v / Math.max(den, 1e-9));
    }

    for (let i = 0; i < n; i++) {
      for (let c = 0; c < k; c++) {
        const dci = Math.sqrt(distance2(X[i], centroids[c])) + 1e-9;
        let den = 0;
        for (let cj = 0; cj < k; cj++) {
          const dji = Math.sqrt(distance2(X[i], centroids[cj])) + 1e-9;
          den += Math.pow(dci / dji, 2 / (m - 1));
        }
        U[i][c] = 1 / den;
      }
    }
  }

  const labels = U.map((row) => {
    let best = 0;
    for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
    return best;
  });

  let inertia = 0;
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) inertia += Math.pow(U[i][c], m) * distance2(X[i], centroids[c]);
  }

  return { labels, centroids, inertia, membership: U };
}

function computeClusteringMetrics(X, labels, centroids) {
  const n = X.length;
  const k = centroids.length;
  const groups = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) groups[labels[i]].push(i);

  const sampleIdx = n > 220 ? shuffleIdx(n, 123).slice(0, 220) : Array.from({ length: n }, (_, i) => i);
  let silSum = 0;
  for (const i of sampleIdx) {
    const ci = labels[i];
    const same = groups[ci];
    let a = 0;
    if (same.length > 1) {
      for (const j of same) if (j !== i) a += Math.sqrt(distance2(X[i], X[j]));
      a /= same.length - 1;
    }
    let b = Number.POSITIVE_INFINITY;
    for (let c = 0; c < k; c++) {
      if (c === ci || groups[c].length === 0) continue;
      let d = 0;
      for (const j of groups[c]) d += Math.sqrt(distance2(X[i], X[j]));
      d /= groups[c].length;
      if (d < b) b = d;
    }
    const s = (b - a) / Math.max(a, b, 1e-9);
    silSum += s;
  }
  const silhouette = silSum / Math.max(sampleIdx.length, 1);

  const scat = Array.from({ length: k }, () => 0);
  for (let c = 0; c < k; c++) {
    if (!groups[c].length) continue;
    let s = 0;
    for (const i of groups[c]) s += Math.sqrt(distance2(X[i], centroids[c]));
    scat[c] = s / groups[c].length;
  }

  let db = 0;
  for (let i = 0; i < k; i++) {
    let maxR = 0;
    for (let j = 0; j < k; j++) {
      if (i === j) continue;
      const m = Math.sqrt(distance2(centroids[i], centroids[j]));
      const r = (scat[i] + scat[j]) / Math.max(m, 1e-9);
      if (r > maxR) maxR = r;
    }
    db += maxR;
  }
  db /= Math.max(k, 1);

  const globalMean = Array.from({ length: X[0].length }, (_, j) => vMean(X.map((r) => r[j])));
  let between = 0;
  let within = 0;
  for (let c = 0; c < k; c++) {
    const size = groups[c].length;
    if (!size) continue;
    between += size * distance2(centroids[c], globalMean);
    for (const i of groups[c]) within += distance2(X[i], centroids[c]);
  }
  const calinskiHarabasz = ((n - k) / Math.max(k - 1, 1)) * (between / Math.max(within, 1e-9));

  return {
    Silhouette: silhouette,
    DaviesBouldin: db,
    CalinskiHarabasz: calinskiHarabasz,
    Inertia: within
  };
}

function featureClusterSeparation(X, labels, featureNames) {
  const n = X.length;
  const p = X[0].length;
  const k = Math.max(...labels) + 1;
  const meanAll = Array.from({ length: p }, (_, j) => vMean(X.map((r) => r[j])));
  const scores = [];

  for (let j = 0; j < p; j++) {
    let ssBetween = 0;
    let ssTotal = 0;
    const vals = X.map((r) => r[j]);
    for (let i = 0; i < n; i++) ssTotal += (vals[i] - meanAll[j]) * (vals[i] - meanAll[j]);

    for (let c = 0; c < k; c++) {
      const idx = labels.map((v, i) => (v === c ? i : -1)).filter((v) => v >= 0);
      if (!idx.length) continue;
      const mc = vMean(idx.map((i) => X[i][j]));
      ssBetween += idx.length * (mc - meanAll[j]) * (mc - meanAll[j]);
    }
    const eta2 = ssBetween / Math.max(ssTotal, 1e-9);
    scores.push({ feature: featureNames[j], score: eta2 });
  }

  return scores.sort((a, b) => b.score - a.score);
}

function pca2D(X) {
  const n = X.length;
  const p = X[0].length;
  const mean = Array.from({ length: p }, (_, j) => vMean(X.map((r) => r[j])));
  const centered = X.map((r) => r.map((v, j) => v - mean[j]));

  const cov = Array.from({ length: p }, () => Array.from({ length: p }, () => 0));
  for (let i = 0; i < p; i++) {
    for (let j = i; j < p; j++) {
      let s = 0;
      for (let r = 0; r < n; r++) s += centered[r][i] * centered[r][j];
      cov[i][j] = s / Math.max(n - 1, 1);
      cov[j][i] = cov[i][j];
    }
  }

  const power = (M, initSeed = 11) => {
    const rnd = seeded(initSeed);
    let v = Array.from({ length: p }, () => rnd());
    for (let it = 0; it < 40; it++) {
      const mv = Array.from({ length: p }, (_, i) => M[i].reduce((s, x, j) => s + x * v[j], 0));
      const norm = Math.sqrt(mv.reduce((s, x) => s + x * x, 0)) || 1;
      v = mv.map((x) => x / norm);
    }
    return v;
  };

  const v1 = power(cov, 17);
  const l1 = v1.reduce((s, vi, i) => s + vi * cov[i].reduce((a, x, j) => a + x * v1[j], 0), 0);
  const def = cov.map((row, i) => row.map((x, j) => x - l1 * v1[i] * v1[j]));
  const v2 = power(def, 23);

  return centered.map((row) => ({
    x: row.reduce((s, v, i) => s + v * v1[i], 0),
    y: row.reduce((s, v, i) => s + v * v2[i], 0)
  }));
}

function formatMetric(v) {
  if (!Number.isFinite(v)) return "-";
  return Math.abs(v) >= 100 ? v.toFixed(2) : v.toFixed(4);
}

function serializeSvg(svgEl) {
  const clone = svgEl.cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", "0");
  bg.setAttribute("y", "0");
  bg.setAttribute("width", "100%");
  bg.setAttribute("height", "100%");
  bg.setAttribute("fill", "white");
  clone.insertBefore(bg, clone.firstChild);
  return new XMLSerializer().serializeToString(clone);
}

function findChartSvg(container) {
  if (!container) return null;
  return container.querySelector("svg");
}

function downloadSvgFromContainer(container, fileName = "research_visual") {
  const svg = findChartSvg(container);
  if (!svg) {
    alert("No SVG visual found in this tab to export.");
    return;
  }
  const xml = serializeSvg(svg);
  const blob = new Blob([xml], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${fileName}.svg`;
  a.click();
  URL.revokeObjectURL(url);
}

function svgToPngBlob(svg, scale = 4) {
  return new Promise((resolve, reject) => {
    const xml = serializeSvg(svg);
    const blob = new Blob([xml], { type: "image/svg+xml" });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      const rect = svg.getBoundingClientRect();
      const w = Math.max(1, Math.round((rect.width || svg.viewBox.baseVal.width || 1200) * scale));
      const h = Math.max(1, Math.round((rect.height || svg.viewBox.baseVal.height || 800) * scale));
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
      ctx.drawImage(img, 0, 0, w, h);
      canvas.toBlob((pngBlob) => {
        URL.revokeObjectURL(url);
        if (!pngBlob) {
          reject(new Error("Failed to render PNG."));
          return;
        }
        resolve(pngBlob);
      }, "image/png");
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Unable to rasterize SVG."));
    };
    img.src = url;
  });
}

async function downloadPngFromContainer(container, fileName = "research_visual", scale = 4) {
  const svg = findChartSvg(container);
  if (!svg) {
    alert("No SVG visual found in this tab to export.");
    return;
  }
  try {
    const blob = await svgToPngBlob(svg, scale);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${fileName}.png`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert(`PNG export failed: ${e.message}`);
  }
}

async function copyPngFromContainer(container, scale = 4) {
  const svg = findChartSvg(container);
  if (!svg) {
    alert("No SVG visual found in this tab to clip/copy.");
    return;
  }
  if (!navigator.clipboard || !window.ClipboardItem) {
    alert("Clipboard image copy is not supported in this browser.");
    return;
  }
  try {
    const blob = await svgToPngBlob(svg, scale);
    await navigator.clipboard.write([new window.ClipboardItem({ "image/png": blob })]);
    alert("Visual copied to clipboard as high-fidelity PNG.");
  } catch (e) {
    alert(`Clipboard copy failed: ${e.message}`);
  }
}

// ── OLS + Analytical SHAP ─────────────────────────────────────────────────────
function computeAnalyticalShap(Xraw, y) {
  const n = Xraw.length, p = Xraw[0].length;
  const means = Array.from({ length: p }, (_, j) => vMean(Xraw.map(r => r[j])));
  const stds = Array.from({ length: p }, (_, j) => vStd(Xraw.map(r => r[j]), means[j]));
  const Xsc = Xraw.map(r => r.map((v, j) => (v - means[j]) / stds[j]));
  const Xaug = Xsc.map(r => [1, ...r]);
  const Xt = mT(Xaug); let XtX = mmul(Xt, Xaug);
  for (let i = 1; i <= p; i++) XtX[i][i] += 0.001; // ridge
  const Xty = Xt.map(r => r.reduce((s, v, i) => s + v * y[i], 0));
  const beta = mmul(minv(XtX), Xty.map(v => [v])).map(r => r[0]);
  const intercept = beta[0], coefs = beta.slice(1);
  const yPred = Xaug.map(r => r.reduce((s, v, i) => s + v * beta[i], 0));
  const ym = vMean(y), ssR = y.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0), ssT = y.reduce((s, v) => s + (v - ym) ** 2, 0);
  const r2 = 1 - ssR / ssT, adjR2 = 1 - (1 - r2) * (n - 1) / (n - p - 1), rmse = Math.sqrt(ssR / n);
  const shapVals = Xsc.map(r => r.map((v, j) => v * coefs[j]));
  const meanAbs = coefs.map((_, j) => vMean(shapVals.map(r => Math.abs(r[j]))));
  const orderImp = [...meanAbs.map((v, i) => ({ v, i }))].sort((a, b) => b.v - a.v).map(d => d.i);
  return { intercept, coefs, r2, adjR2, rmse, yPred, shapVals, meanAbs, orderImp, Xsc, Xraw, means, stds, n, p, ym };
}

// ── Helper Components ────────────────────────────────────────────────────────
function FigTitle({ num, title, desc }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
        <span style={{ fontFamily: "Georgia,serif", fontWeight: 700, color: C.navy, fontSize: 13 }}>Figure {num}.</span>
        <span style={{ fontFamily: "Georgia,serif", fontWeight: 600, color: C.ink, fontSize: 13 }}>{title}</span>
      </div>
      {desc && <div style={{ fontSize: 11, color: C.gray, marginTop: 3, fontStyle: "italic", fontFamily: "Georgia,serif" }}>{desc}</div>}
    </div>
  );
}

function dlSVG(el, name) {
  if (!el) return;
  const svg = el.tagName === "svg" ? el : el.querySelector("svg");
  if (!svg) return;
  const clone = svg.cloneNode(true);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  const xml = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([xml], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${name}.svg`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── SHAP Visualization Components ────────────────────────────────────────────
function OlsFeatureImportance({ features, meanAbs, coefs, orderImp, target }) {
  const ref = useRef();
  const sorted = [...orderImp].reverse();
  const W = 560, bH = 30, pL = 178, pR = 90, pT = 28, pB = 44;
  const H = sorted.length * bH + pT + pB;
  const maxV = Math.max(...meanAbs) * 1.22;
  const sx = v => pL + (v / maxV) * (W - pL - pR);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(t => ({ t, x: sx(t * maxV), v: (t * maxV).toFixed(2) }));
  return (
    <div>
      <FigTitle num={1} title="SHAP Feature Importance (Analytical)" desc={`Mean absolute SHAP value for each feature predicting ${target}. Red = net positive effect; Blue = net negative.`} />
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block" }} xmlns="http://www.w3.org/2000/svg">
          {sorted.map((fi, i) => {
            const y0 = pT + i * bH, fn = features[fi], v = meanAbs[fi], cl = coefs[fi] >= 0 ? C.posShap : C.negShap, bw = sx(v) - pL;
            return (<g key={fi}>
              <text x={pL - 8} y={y0 + bH / 2 + 4} textAnchor="end" fontSize={11} fill={C.navy} fontFamily="Georgia,serif">{fn.length > 25 ? fn.slice(0, 23) + "…" : fn}</text>
              <rect x={pL} y={y0 + 5} width={Math.max(bw, 2)} height={bH - 10} fill={cl} opacity={0.84} rx={3} />
              <text x={pL + bw + 5} y={y0 + bH / 2 + 4} fontSize={10} fill={C.gray} fontFamily="monospace">{v.toFixed(3)}</text>
            </g>);
          })}
          <line x1={pL} y1={pT + sorted.length * bH} x2={W - pR} y2={pT + sorted.length * bH} stroke={C.border} strokeWidth={1} />
          {ticks.map(({ t, x, v }) => <g key={t}>
            <line x1={x} y1={pT + sorted.length * bH} x2={x} y2={pT + sorted.length * bH + 5} stroke={C.gray} strokeWidth={0.8} />
            <text x={x} y={pT + sorted.length * bH + 15} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="monospace">{v}</text>
          </g>)}
          <text x={pL + (W - pL - pR) / 2} y={H - 5} textAnchor="middle" fontSize={11} fill={C.gray} fontFamily="Georgia,serif">Mean |SHAP value|</text>
          <rect x={W - pR + 5} y={pT} width={11} height={11} fill={C.posShap} opacity={0.84} rx={1} />
          <text x={W - pR + 19} y={pT + 9} fontSize={9} fill={C.gray}>Positive</text>
          <rect x={W - pR + 5} y={pT + 16} width={11} height={11} fill={C.negShap} opacity={0.84} rx={1} />
          <text x={W - pR + 19} y={pT + 25} fontSize={9} fill={C.gray}>Negative</text>
        </svg>
      </div>
      <button style={dlBtn} onClick={() => dlSVG(ref.current, "Fig1_SHAP_Importance")}>↓ Download SVG</button>
    </div>
  );
}

function OlsShapBeeswarm({ features, shapVals, Xraw, orderImp, target }) {
  const ref = useRef();
  const n = shapVals.length, p = features.length;
  const W = 580, pH = 32, pL = 175, pR = 50, pT = 28, pB = 44;
  const H = p * pH + pT + pB;
  const allSV = shapVals.flatMap(r => r); const mnSV = Math.min(...allSV), mxSV = Math.max(...allSV);
  const svRange = Math.max(Math.abs(mnSV), Math.abs(mxSV));
  const innerW = W - pL - pR;
  const sx = v => pL + innerW / 2 + v / svRange * (innerW / 2) * 0.92;
  const rng = (seed) => {
    let s = seed; return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  };
  const cbW = 14, cbH = Math.min(p * pH - 8, 80);
  const cbX = W - pR - cbW - 4, cbY = pT + (p * pH - cbH) / 2;
  return (
    <div>
      <FigTitle num={2} title="SHAP Beeswarm Summary" desc={`Each dot is one observation. Horizontal position = SHAP value (impact on ${target}). Colour = feature value magnitude (low → blue → yellow → high).`} />
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block" }} xmlns="http://www.w3.org/2000/svg">
          <line x1={pL + innerW / 2} y1={pT} x2={pL + innerW / 2} y2={pT + p * pH} stroke={C.border} strokeWidth={1} strokeDasharray="3,3" />
          {orderImp.map((fi, row) => {
            const y0 = pT + row * pH + pH / 2;
            const sv = shapVals.map(r => r[fi]);
            const fv = Xraw.map(r => r[fi]);
            const fvMn = Math.min(...fv), fvMx = Math.max(...fv);
            const rand = rng(fi * 137);
            const sorted = [...sv.map((v, i) => ({ v, fv: fv[i], i }))].sort((a, b) => a.v - b.v);
            return (<g key={fi}>
              <text x={pL - 8} y={y0 + 4} textAnchor="end" fontSize={11} fill={C.navy} fontFamily="Georgia,serif">
                {features[fi].length > 24 ? features[fi].slice(0, 22) + "…" : features[fi]}
              </text>
              {sorted.map(({ v, fv: fval }, si) => {
                const cx = sx(v), jit = (rand() - 0.5) * pH * 0.7;
                const col = viridis(normVal(fval, fvMn, fvMx));
                return <circle key={si} cx={cx} cy={y0 + jit} r={3.5} fill={col} opacity={0.85} stroke="white" strokeWidth={0.5} />;
              })}
            </g>);
          })}
          <line x1={pL} y1={pT + p * pH} x2={W - pR - cbW - 12} y2={pT + p * pH} stroke={C.border} strokeWidth={1} />
          {[-1, -0.5, 0, 0.5, 1].map(t => {
            const x = sx(t * svRange), v = (t * svRange).toFixed(2);
            return (<g key={t}>
              <line x1={x} y1={pT + p * pH} x2={x} y2={pT + p * pH + 5} stroke={C.gray} strokeWidth={0.8} />
              <text x={x} y={pT + p * pH + 15} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="monospace">{v}</text>
            </g>);
          })}
          <text x={pL + innerW / 2 - 20} y={H - 5} textAnchor="middle" fontSize={11} fill={C.gray} fontFamily="Georgia,serif">SHAP value (impact on model output)</text>
          {Array.from({ length: cbH }, (_, k) => {
            const t = 1 - k / cbH; return <rect key={k} x={cbX} y={cbY + k} width={cbW} height={1.5} fill={viridis(t)} />;
          })}
          <rect x={cbX} y={cbY} width={cbW} height={cbH} fill="none" stroke={C.border} strokeWidth={0.5} />
          <text x={cbX + cbW + 3} y={cbY + 6} fontSize={8} fill={C.gray}>High</text>
          <text x={cbX + cbW + 3} y={cbY + cbH + 4} fontSize={8} fill={C.gray}>Low</text>
          <text x={cbX + cbW / 2} y={cbY - 4} textAnchor="middle" fontSize={8} fill={C.gray}>Feature</text>
          <text x={cbX + cbW / 2} y={cbY - 14} textAnchor="middle" fontSize={8} fill={C.gray}>value</text>
        </svg>
      </div>
      <button style={dlBtn} onClick={() => dlSVG(ref.current, "Fig2_SHAP_Beeswarm")}>↓ Download SVG</button>
    </div>
  );
}

function OlsShapDependence({ features, shapVals, Xraw, orderImp, target }) {
  const ref = useRef();
  const top4 = orderImp.slice(0, 4);
  const cols = 2, rows = 2;
  const cW = 260, cH = 200, pad = { t: 30, r: 15, b: 44, l: 52 }, gap = 20;
  const W = cols * cW + gap, H = rows * cH + gap;
  const allColors = Xraw.map(r => r[orderImp[0]]);
  const acMin = Math.min(...allColors), acMax = Math.max(...allColors);
  return (
    <div>
      <FigTitle num={3} title="SHAP Dependence Plots — Top 4 Features" desc={`Feature value (x) vs. SHAP contribution (y) for the four most influential features. Colour indicates feature value magnitude.`} />
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block" }} xmlns="http://www.w3.org/2000/svg">
          {top4.map((fi, idx) => {
            const col = idx % 2, row = Math.floor(idx / 2);
            const ox = col * (cW + gap), oy = row * (cH + gap);
            const iW = cW - pad.l - pad.r, iH = cH - pad.t - pad.b;
            const fvArr = Xraw.map(r => r[fi]);
            const svArr = shapVals.map(r => r[fi]);
            const fvMin = Math.min(...fvArr), fvMax = Math.max(...fvArr);
            const svMin = Math.min(...svArr), svMax = Math.max(...svArr);
            const svPad = (svMax - svMin) * 0.12 || 0.5;
            const sx2 = v => ox + pad.l + (v - fvMin) / (fvMax - fvMin || 1) * iW;
            const sy2 = v => oy + pad.t + iH - (v - (svMin - svPad)) / ((svMax + svPad) - (svMin - svPad)) * iH;
            const n = fvArr.length;
            const xm = vMean(fvArr), ym2 = vMean(svArr);
            const slope = fvArr.reduce((s, v, i) => s + (v - xm) * (svArr[i] - ym2), 0) / (fvArr.reduce((s, v) => s + (v - xm) ** 2, 0) || 1);
            const intcpt = ym2 - slope * xm;
            const x1l = fvMin, x2l = fvMax, y1l = slope * x1l + intcpt, y2l = slope * x2l + intcpt;
            return (<g key={fi}>
              <rect x={ox + pad.l} y={oy + pad.t} width={iW} height={iH} fill={C.bg} rx={2} />
              {svMin < 0 && svMax > 0 && <line x1={ox + pad.l} y1={sy2(0)} x2={ox + pad.l + iW} y2={sy2(0)} stroke={C.border} strokeWidth={0.8} strokeDasharray="3,2" />}
              <line x1={sx2(x1l)} y1={sy2(y1l)} x2={sx2(x2l)} y2={sy2(y2l)} stroke={C.ink} strokeWidth={1.3} strokeDasharray="4,3" opacity={0.6} />
              {fvArr.map((fv, i) => <circle key={i} cx={sx2(fv)} cy={sy2(svArr[i])} r={3.5} fill={viridis(normVal(fv, fvMin, fvMax))} opacity={0.8} stroke="white" strokeWidth={0.4} />)}
              <line x1={ox + pad.l} y1={oy + pad.t + iH} x2={ox + pad.l + iW} y2={oy + pad.t + iH} stroke={C.border} strokeWidth={1} />
              <line x1={ox + pad.l} y1={oy + pad.t} x2={ox + pad.l} y2={oy + pad.t + iH} stroke={C.border} strokeWidth={1} />
              <text x={ox + pad.l + iW / 2} y={oy + cH - 4} textAnchor="middle" fontSize={10} fill={C.gray} fontFamily="Georgia,serif">{features[fi].length > 20 ? features[fi].slice(0, 18) + "…" : features[fi]}</text>
              <text x={ox + 8} y={oy + pad.t + iH / 2} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="Georgia,serif" transform={`rotate(-90,${ox + 8},${oy + pad.t + iH / 2})`}>SHAP</text>
              <text x={ox + pad.l + iW / 2} y={oy + pad.t - 8} textAnchor="middle" fontSize={11} fill={C.ink} fontWeight="bold" fontFamily="Georgia,serif">{features[fi].length > 20 ? features[fi].slice(0, 18) + "…" : features[fi]}</text>
              {[svMin, ym2, svMax].map((v, k) => <g key={k}>
                <line x1={ox + pad.l - 3} y1={sy2(v)} x2={ox + pad.l} y2={sy2(v)} stroke={C.gray} strokeWidth={0.7} />
                <text x={ox + pad.l - 5} y={sy2(v) + 3} textAnchor="end" fontSize={8} fill={C.gray} fontFamily="monospace">{v.toFixed(1)}</text>
              </g>)}
            </g>);
          })}
        </svg>
      </div>
      <button style={dlBtn} onClick={() => dlSVG(ref.current, "Fig3_SHAP_Dependence")}>↓ Download SVG</button>
    </div>
  );
}

function OlsShapWaterfall({ features, shapVals, orderImp, rowIndex, target }) {
  const ref = useRef();
  const row = shapVals[rowIndex] || [];
  const top = orderImp.slice(0, 8);
  let running = 0;
  const steps = top.map((fi) => {
    const start = running;
    const delta = row[fi] || 0;
    const end = start + delta;
    running = end;
    return { fi, start, end, delta };
  });
  const lo = Math.min(0, ...steps.map((s) => Math.min(s.start, s.end)));
  const hi = Math.max(0, ...steps.map((s) => Math.max(s.start, s.end)));
  const W = 620, H = 360, pL = 100, pR = 22, pT = 26, pB = 58;
  const innerW = W - pL - pR;
  const rowH = (H - pT - pB) / Math.max(steps.length, 1);
  const sx = (v) => pL + ((v - lo) / (hi - lo || 1)) * innerW;

  return (
    <div>
      <FigTitle num={4} title="SHAP Local Waterfall (One Observation)" desc={`Step-by-step contribution to ${target} for row #${rowIndex + 1}. Red bars increase prediction, blue bars decrease it.`} />
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block" }} xmlns="http://www.w3.org/2000/svg">
          <line x1={sx(0)} y1={pT} x2={sx(0)} y2={H - pB} stroke={C.border} strokeWidth={1} strokeDasharray="4,3" />
          {steps.map((s, i) => {
            const y = pT + i * rowH + 4;
            const x0 = sx(s.start);
            const x1 = sx(s.end);
            const x = Math.min(x0, x1);
            const w = Math.max(Math.abs(x1 - x0), 1.5);
            const col = s.delta >= 0 ? C.posShap : C.negShap;
            return (
              <g key={s.fi}>
                <text x={pL - 8} y={y + rowH / 2 + 3} textAnchor="end" fontSize={10} fill={C.navy} fontFamily="Georgia,serif">
                  {features[s.fi].length > 24 ? `${features[s.fi].slice(0, 22)}…` : features[s.fi]}
                </text>
                <rect x={x} y={y} width={w} height={rowH - 8} fill={col} opacity={0.84} rx={3} />
                <text x={x + w + 4} y={y + rowH / 2 + 3} fontSize={9} fill={C.gray} fontFamily="monospace">
                  {s.delta >= 0 ? "+" : ""}{s.delta.toFixed(3)}
                </text>
              </g>
            );
          })}
          <line x1={pL} y1={H - pB} x2={W - pR} y2={H - pB} stroke={C.border} strokeWidth={1} />
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const v = lo + t * (hi - lo);
            const x = sx(v);
            return (
              <g key={t}>
                <line x1={x} y1={H - pB} x2={x} y2={H - pB + 5} stroke={C.gray} strokeWidth={0.8} />
                <text x={x} y={H - pB + 16} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="monospace">{v.toFixed(2)}</text>
              </g>
            );
          })}
          <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fill={C.gray} fontFamily="Georgia,serif">Cumulative SHAP contribution</text>
        </svg>
      </div>
      <button style={dlBtn} onClick={() => dlSVG(ref.current, "Fig4_SHAP_Local_Waterfall")}>↓ Download SVG</button>
    </div>
  );
}

function OlsShapForcePlot({ features, shapVals, orderImp, rowIndex, target }) {
  const ref = useRef();
  const row = shapVals[rowIndex] || [];
  const top = orderImp.slice(0, 10);
  const points = [{ label: "Base", v: 0 }];
  let run = 0;
  top.forEach((fi) => {
    run += row[fi] || 0;
    points.push({ label: features[fi], v: run, d: row[fi] || 0 });
  });
  const mn = Math.min(...points.map((p) => p.v), 0);
  const mx = Math.max(...points.map((p) => p.v), 0);
  const W = 700, H = 250, pL = 42, pR = 24, pT = 46, pB = 54;
  const sx = (v) => pL + ((v - mn) / (mx - mn || 1)) * (W - pL - pR);
  const y = H / 2;
  return (
    <div>
      <FigTitle num={6} title="SHAP Force Plot (Local Explanation)" desc={`Cumulative SHAP pushes for row #${rowIndex + 1} on ${target}. Red pushes prediction higher, blue pushes lower.`} />
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block" }} xmlns="http://www.w3.org/2000/svg">
          <line x1={pL} y1={y} x2={W - pR} y2={y} stroke={C.border} strokeWidth={1} />
          <line x1={sx(0)} y1={pT - 8} x2={sx(0)} y2={H - pB + 8} stroke={C.gray} strokeDasharray="3,3" />
          {top.map((fi, i) => {
            const d = row[fi] || 0;
            const x0 = sx(points[i].v);
            const x1 = sx(points[i + 1].v);
            const x = Math.min(x0, x1);
            const w = Math.max(Math.abs(x1 - x0), 2);
            const col = d >= 0 ? C.posShap : C.negShap;
            return (
              <g key={fi}>
                <rect x={x} y={y - 14} width={w} height={28} fill={col} opacity={0.82} rx={3} />
                {w > 36 && <text x={x + w / 2} y={y + 3} textAnchor="middle" fontSize={8} fill="white">{features[fi].slice(0, 8)}</text>}
              </g>
            );
          })}
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const v = mn + t * (mx - mn);
            const x = sx(v);
            return (
              <g key={t}>
                <line x1={x} y1={H - pB} x2={x} y2={H - pB + 5} stroke={C.gray} />
                <text x={x} y={H - pB + 16} textAnchor="middle" fontSize={9} fill={C.gray} fontFamily="monospace">{v.toFixed(2)}</text>
              </g>
            );
          })}
          <text x={sx(0)} y={pT - 14} textAnchor="middle" fontSize={9} fill={C.gray}>Base</text>
          <text x={sx(run)} y={pT - 14} textAnchor="middle" fontSize={9} fill={C.navy}>Final</text>
          <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fill={C.gray}>Model output shift from SHAP contributions</text>
        </svg>
      </div>
      <button style={dlBtn} onClick={() => dlSVG(ref.current, "Fig6_SHAP_Force_Plot")}>↓ Download SVG</button>
    </div>
  );
}

function OlsSignedImpactBars({ features, meanAbs, meanSigned, orderImp, target }) {
  const ref = useRef();
  const sorted = [...orderImp].reverse().slice(0, 12);
  const W = 600, bH = 28, pL = 180, pR = 30, pT = 26, pB = 42;
  const H = sorted.length * bH + pT + pB;
  const lim = Math.max(...sorted.map((fi) => Math.abs(meanSigned[fi])), 1e-9) * 1.2;
  const center = pL + (W - pL - pR) / 2;
  const sx = (v) => center + (v / lim) * ((W - pL - pR) / 2) * 0.92;
  return (
    <div>
      <FigTitle num={5} title="SHAP Signed Direction (Top Features)" desc={`Average SHAP direction for ${target}. Right side increases prediction; left side decreases prediction.`} />
      <div ref={ref}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block" }} xmlns="http://www.w3.org/2000/svg">
          <line x1={center} y1={pT} x2={center} y2={H - pB} stroke={C.border} strokeDasharray="3,3" />
          {sorted.map((fi, i) => {
            const y = pT + i * bH + 5;
            const x0 = sx(0), x1 = sx(meanSigned[fi]);
            const x = Math.min(x0, x1), w = Math.max(Math.abs(x1 - x0), 1.5);
            const col = meanSigned[fi] >= 0 ? C.posShap : C.negShap;
            return (
              <g key={fi}>
                <text x={pL - 8} y={y + bH / 2 + 3} textAnchor="end" fontSize={10} fill={C.navy}>{features[fi]}</text>
                <rect x={x} y={y} width={w} height={bH - 10} fill={col} opacity={0.86} rx={3} />
                <text x={x1 + (meanSigned[fi] >= 0 ? 4 : -4)} y={y + bH / 2 + 3} textAnchor={meanSigned[fi] >= 0 ? "start" : "end"} fontSize={9} fill={C.gray} fontFamily="monospace">{meanSigned[fi].toFixed(3)}</text>
              </g>
            );
          })}
          <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fill={C.gray}>Mean SHAP value (signed)</text>
        </svg>
      </div>
      <button style={dlBtn} onClick={() => dlSVG(ref.current, "Fig5_SHAP_Signed_Direction")}>↓ Download SVG</button>
    </div>
  );
}

function OlsShapImpactDistribution({ shapVals }) {
  const flat = shapVals.flat();
  const bins = 24;
  const mn = Math.min(...flat), mx = Math.max(...flat);
  const step = (mx - mn || 1) / bins;
  const hist = Array.from({ length: bins }, () => 0);
  flat.forEach((v) => {
    const id = clamp(Math.floor((v - mn) / step), 0, bins - 1);
    hist[id] += 1;
  });
  const maxH = Math.max(...hist, 1);
  const negShare = flat.filter((v) => v < 0).length / Math.max(flat.length, 1);
  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>SHAP Impact Distribution</div>
      <div style={{ fontSize: 11, color: C.gray, marginBottom: 8 }}>Negative share: {(negShare * 100).toFixed(1)}% · Positive share: {((1 - negShare) * 100).toFixed(1)}%</div>
      <div style={{ display: "flex", gap: 3, alignItems: "flex-end", height: 170, padding: "8px 6px", border: `1px solid ${C.border}`, borderRadius: 6, background: "#FBFCFE" }}>
        {hist.map((h, i) => <div key={i} style={{ flex: 1, height: `${(h / maxH) * 100}%`, background: i < bins / 2 ? C.negShap : C.posShap, opacity: 0.78, borderRadius: 2 }} />)}
      </div>
    </div>
  );
}

function OlsShapTopTable({ features, meanAbs, meanSigned, orderImp }) {
  const top = [...orderImp].slice(0, 15);
  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>SHAP Ranked Summary Table</div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ background: C.navy, color: "white" }}>
              <th style={{ padding: "7px 8px", textAlign: "left" }}>Rank</th>
              <th style={{ padding: "7px 8px", textAlign: "left" }}>Feature</th>
              <th style={{ padding: "7px 8px", textAlign: "left" }}>Mean |SHAP|</th>
              <th style={{ padding: "7px 8px", textAlign: "left" }}>Mean SHAP (signed)</th>
              <th style={{ padding: "7px 8px", textAlign: "left" }}>Direction</th>
            </tr>
          </thead>
          <tbody>
            {top.map((fi, i) => (
              <tr key={fi} style={{ background: i % 2 === 0 ? C.paper : C.bg }}>
                <td style={{ padding: "6px 8px", fontFamily: "monospace" }}>{i + 1}</td>
                <td style={{ padding: "6px 8px", color: C.navy, fontWeight: 700 }}>{features[fi]}</td>
                <td style={{ padding: "6px 8px", fontFamily: "monospace" }}>{meanAbs[fi].toFixed(4)}</td>
                <td style={{ padding: "6px 8px", fontFamily: "monospace" }}>{meanSigned[fi].toFixed(4)}</td>
                <td style={{ padding: "6px 8px", color: meanSigned[fi] >= 0 ? C.posShap : C.negShap, fontWeight: 700 }}>{meanSigned[fi] >= 0 ? "Positive" : "Negative"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function OlsShapCumulativeCoverage({ features, meanAbs, orderImp }) {
  const top = [...orderImp].slice(0, 15);
  const total = meanAbs.reduce((s, v) => s + v, 0) || 1;
  let running = 0;
  const data = top.map((fi, i) => {
    running += meanAbs[fi];
    return { rank: i + 1, feature: features[fi], share: running / total };
  });
  const W = 620, H = 320, pL = 50, pR = 20, pT = 20, pB = 50;
  const sx = (i) => pL + ((i - 1) / Math.max(data.length - 1, 1)) * (W - pL - pR);
  const sy = (v) => H - pB - v * (H - pT - pB);
  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>SHAP Cumulative Coverage Curve</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block", background: C.paper }} xmlns="http://www.w3.org/2000/svg">
        <line x1={pL} y1={H - pB} x2={W - pR} y2={H - pB} stroke={C.border} />
        <line x1={pL} y1={H - pB} x2={pL} y2={pT} stroke={C.border} />
        {data.length > 1 && <polyline fill="none" stroke={C.blue} strokeWidth={2.2} points={data.map((d) => `${sx(d.rank)},${sy(d.share)}`).join(" ")} />}
        {data.map((d) => <circle key={d.rank} cx={sx(d.rank)} cy={sy(d.share)} r={3} fill={C.navy} />)}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => <g key={t}>
          <line x1={pL - 4} y1={sy(t)} x2={pL} y2={sy(t)} stroke={C.gray} />
          <text x={pL - 8} y={sy(t) + 3} textAnchor="end" fontSize={9} fill={C.gray}>{Math.round(t * 100)}%</text>
        </g>)}
        <text x={W / 2} y={H - 10} textAnchor="middle" fontSize={11} fill={C.gray}>Top-ranked features included</text>
      </svg>
    </div>
  );
}

function OlsShapSignBalance({ features, shapVals, orderImp }) {
  const top = [...orderImp].slice(0, 12);
  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>SHAP Sign Balance by Feature</div>
      {top.map((fi) => {
        const vals = shapVals.map((r) => r[fi]);
        const pos = vals.filter((v) => v > 0).length / Math.max(vals.length, 1);
        const neg = vals.filter((v) => v < 0).length / Math.max(vals.length, 1);
        const neu = Math.max(0, 1 - pos - neg);
        return (
          <div key={fi} style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 11, color: C.ink, marginBottom: 3 }}>{features[fi]}</div>
            <div style={{ display: "flex", height: 14, borderRadius: 4, overflow: "hidden", background: C.light }}>
              <div style={{ width: `${neg * 100}%`, background: C.negShap }} />
              <div style={{ width: `${neu * 100}%`, background: "#D9E2EC" }} />
              <div style={{ width: `${pos * 100}%`, background: C.posShap }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function OlsShapCohortContrast({ features, shapVals, orderImp, y }) {
  const top = [...orderImp].slice(0, 10);
  const sorted = [...y].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(0.25 * (sorted.length - 1))] ?? 0;
  const q3 = sorted[Math.floor(0.75 * (sorted.length - 1))] ?? 0;
  const lowIds = y.map((v, i) => ({ v, i })).filter((d) => d.v <= q1).map((d) => d.i);
  const highIds = y.map((v, i) => ({ v, i })).filter((d) => d.v >= q3).map((d) => d.i);
  const rows = top.map((fi) => {
    const low = vMean(lowIds.map((i) => shapVals[i][fi] || 0));
    const high = vMean(highIds.map((i) => shapVals[i][fi] || 0));
    return { fi, low, high, delta: high - low };
  });
  const lim = Math.max(...rows.map((r) => Math.max(Math.abs(r.low), Math.abs(r.high))), 1e-9) * 1.2;
  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>SHAP Cohort Contrast (Top vs Bottom Quartile Target)</div>
      {rows.map((r) => (
        <div key={r.fi} style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 11, color: C.ink, marginBottom: 3 }}>{features[r.fi]}</div>
          <div style={{ display: "grid", gridTemplateColumns: "54px 1fr 54px", gap: 6, alignItems: "center" }}>
            <div style={{ fontSize: 10, color: C.gray }}>Low Q</div>
            <div style={{ height: 10, background: C.light, borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${(Math.abs(r.low) / lim) * 100}%`, height: "100%", background: C.negShap, opacity: 0.85 }} />
            </div>
            <div style={{ fontFamily: "monospace", fontSize: 10, color: C.gray }}>{r.low.toFixed(3)}</div>
            <div style={{ fontSize: 10, color: C.gray }}>High Q</div>
            <div style={{ height: 10, background: C.light, borderRadius: 3, overflow: "hidden" }}>
              <div style={{ width: `${(Math.abs(r.high) / lim) * 100}%`, height: "100%", background: C.posShap, opacity: 0.85 }} />
            </div>
            <div style={{ fontFamily: "monospace", fontSize: 10, color: C.gray }}>{r.high.toFixed(3)}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function InsightNarrative({ categoryTab, activeTab, targetCol, selectedResult, selectedCluster }) {
  const narratives = {
    profile: `Data Profile summarizes data quality, spread, and central tendency so you can spot missing values, skew, and outliers before modeling ${targetCol || "the target"}.`,
    corr: "Correlation Heatmap highlights linear relationships; strong off-diagonal values may indicate redundancy or multicollinearity.",
    shap_importance: `Feature Importance ranks global drivers of ${targetCol || "the target"} by mean |SHAP|, showing which variables consistently matter most.`,
    shap_signed: "Signed SHAP bars summarize average direction by feature, showing which predictors tend to push predictions up versus down.",
    shap_beeswarm: `Beeswarm shows the full distribution of effects: horizontal spread = impact strength, color = raw feature magnitude, and sign indicates push up/down on ${targetCol || "prediction"}.`,
    shap_heatmap: "SHAP heatmap compares contribution patterns across observations and top features, helping identify row-level regimes.",
    shap_dependence: "Dependence plots reveal response shape for top features and help detect monotonic trends, thresholds, and non-linear behavior.",
    shap_distribution: "Impact distribution summarizes how SHAP contributions are spread globally and whether negative vs positive contributions are balanced.",
    shap_table: "SHAP ranked table provides publication-ready numeric summaries of magnitude and direction for the most influential predictors.",
    shap_waterfall: `Local Waterfall explains one row at a time as a narrative of additive contributions, ideal for case-level explanation and reporting.`,
    shap_force: "Force plot shows cumulative local pushes from top features for one observation, making the final prediction shift easy to track.",
    shap_coverage: "Cumulative coverage shows how quickly top-ranked features account for total SHAP importance mass.",
    shap_sign_balance: "Sign-balance reveals how often each feature pushes predictions up, down, or near-zero across the dataset.",
    shap_cohort: "Cohort contrast compares average SHAP effects between low-target and high-target groups to expose regime-specific drivers.",
    metrics: `Model Metrics table compares candidate regressors. Prioritize lower error metrics and stable R2/VAF when selecting a production baseline.`,
    pred: `${selectedResult?.name || "Selected model"} prediction scatter checks calibration. Tight clustering around the diagonal indicates stronger generalization.`,
    residual: "Residual analysis helps identify bias pockets and heteroscedasticity; wide structured residuals suggest model misspecification.",
    cluster_metrics: "Clustering metrics compare separation/compactness trade-offs across methods. Higher Silhouette and lower Davies-Bouldin are generally preferred.",
    cluster_scatter: `${selectedCluster?.name || "Selected model"} PCA cluster map provides a 2D structure check; overlap can indicate weak separability.`,
    cluster_sizes: "Cluster size chart checks balance. Very small clusters can indicate outlier groups or an overly granular k.",
    cluster_importance: "Feature separation scores indicate which variables most distinguish clusters, supporting interpretable segmentation narratives."
  };
  return (
    <div style={{ ...cardStyle, marginBottom: 10, borderLeft: `4px solid ${C.blue}` }}>
      <div style={{ fontSize: 12, fontWeight: 700, color: C.navy, marginBottom: 4 }}>Narrative</div>
      <div style={{ fontSize: 12, lineHeight: 1.55, color: C.gray }}>
        {narratives[activeTab] || "Select a view to see an interpretation narrative for the current visualization."}
      </div>
      {categoryTab === "shap" && (
        <div style={{ fontSize: 11, color: C.gray, marginTop: 8 }}>
          Reporting tip: combine global (importance/beeswarm/dependence) with local (waterfall) explanations for a complete SHAP story.
        </div>
      )}
    </div>
  );
}

function buildReportParagraph({ activeTab, analysisMode, targetCol, selectedResult, selectedCluster, shapRowIndex }) {
  const byTab = {
    profile: `Data profiling was conducted prior to modeling to validate completeness, spread, and potential outliers for variables used in predicting ${targetCol || "the target outcome"}. This step supports trustworthy downstream interpretation by flagging quality risks before algorithm fitting.`,
    corr: `Correlation structure was reviewed to identify strongly related predictors that may influence model stability. The heatmap was used as an initial diagnostic to detect redundancy patterns that could affect both predictive performance and interpretability.`,
    shap_importance: `Global SHAP importance analysis indicates the relative influence of each feature on predictions of ${targetCol || "the target outcome"}. Features with higher mean absolute SHAP values were interpreted as primary drivers and prioritized in the explanatory narrative.`,
    shap_signed: `Signed SHAP feature profiles were reviewed to determine each predictor's net directional tendency on model output, supporting interpretation beyond absolute importance alone.`,
    shap_beeswarm: `The SHAP beeswarm plot was used to assess both magnitude and direction of feature effects across all observations. Horizontal dispersion reflects effect strength, while point distribution around zero highlights whether features systematically increase or decrease model output.`,
    shap_heatmap: `SHAP heatmap patterns were reviewed across observations to identify consistent contribution signatures and potential sub-groups with similar explanatory structure.`,
    shap_dependence: `SHAP dependence plots were examined for top-ranked predictors to characterize effect shape and non-linearity. These plots support interpretation of monotonic trends, threshold behavior, and heterogeneous local responses within the modeled population.`,
    shap_distribution: `The aggregate SHAP contribution distribution was inspected to understand global balance, tail behavior, and overall concentration of positive versus negative impacts.`,
    shap_table: `A ranked SHAP summary table was generated to provide report-ready numerical evidence of feature importance and directionality for transparent documentation.`,
    shap_waterfall: `A local SHAP waterfall explanation was generated for observation #${shapRowIndex + 1}, showing additive feature-level contributions to that case prediction. This observation-level view complements global findings and supports transparent case-by-case reporting.`,
    shap_force: `A local SHAP force plot for observation #${shapRowIndex + 1} was used to trace cumulative positive and negative pushes from influential features to the final model output.`,
    shap_coverage: `A cumulative SHAP coverage curve was used to quantify how many top-ranked features capture most explanatory mass, informing parsimonious feature reporting.`,
    shap_sign_balance: `SHAP sign-balance profiling was used to estimate directional consistency of each feature's contribution across observations.`,
    shap_cohort: `SHAP cohort contrast compared feature contributions between low and high target quartiles to identify segment-dependent explanatory shifts.`,
    metrics: `${cleanModelLabel(selectedResult?.name) || "The selected model"} was benchmarked using standard regression metrics (RMSE, MAE, MAPE, sMAPE, MBE, R2, VAF). Model comparison was used to balance predictive accuracy and robustness before selecting a preferred analytical baseline.`,
    pred: `Observed-versus-predicted scatter for ${cleanModelLabel(selectedResult?.name) || "the selected model"} was reviewed as a calibration diagnostic. Proximity of points to the identity line was interpreted as evidence of stronger generalization in out-of-fold evaluation.`,
    residual: `Residual behavior was analyzed to assess bias and variance structure. Systematic patterns or widening spread were treated as potential indicators of misspecification, heteroscedasticity, or segment-specific performance limitations.`,
    cluster_metrics: `${selectedCluster?.name || "The selected clustering model"} was assessed using internal validity metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz, and inertia where applicable) to evaluate compactness and separation quality across candidate solutions.`,
    cluster_scatter: `PCA-based cluster projection was reviewed to visually assess separation quality for ${selectedCluster?.name || "the selected clustering model"}. Overlap patterns were interpreted as potential evidence of weak boundary definition between latent segments.`,
    cluster_sizes: `Cluster membership counts were examined to evaluate segment balance and practical interpretability. Highly imbalanced group sizes were treated as potential outlier-driven or over-partitioned solutions requiring additional validation.`,
    cluster_importance: `Feature separation scores were used to identify variables most responsible for differentiating cluster assignments. These drivers were used to convert cluster outputs into a domain-facing segmentation narrative.`
  };

  const modeSentence = analysisMode === "regression"
    ? "The workflow was executed in regression mode with cross-validated evaluation."
    : "The workflow was executed in clustering mode with model-level comparative diagnostics.";

  return `${modeSentence} ${byTab[activeTab] || "This view summarizes the current analytical result and its interpretive meaning."}`;
}

async function copyTextToClipboard(text) {
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (_) {
    // no-op fallback below
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  const ok = document.execCommand("copy");
  document.body.removeChild(ta);
  return ok;
}

function downloadTextFile(text, filename) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function UploadPanel({ onData }) {
  const [drag, setDrag] = useState(false);
  const ref = useRef();

  const process = useCallback(async (file) => {
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    let rows;

    if (ext === "csv") {
      const text = await file.text();
      const lines = text.trim().split(/\r?\n/);
      const headers = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
      rows = lines.slice(1).filter((l) => l.trim()).map((line) => {
        const vals = line.split(",").map((v) => v.trim().replace(/^"|"$/g, ""));
        const obj = {};
        headers.forEach((h, i) => {
          obj[h] = vals[i] === "" || Number.isNaN(Number(vals[i])) ? vals[i] : Number(vals[i]);
        });
        return obj;
      });
    } else if (["xlsx", "xls"].includes(ext) && window.XLSX) {
      const buf = await file.arrayBuffer();
      const wb = window.XLSX.read(buf, { type: "array" });
      rows = window.XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { defval: null });
    } else {
      alert("Use CSV/XLSX and ensure XLSX library is loaded.");
      return;
    }

    if (rows?.length) onData(rows, file.name);
  }, [onData]);

  return (
    <div
      style={{ border: `2px dashed ${drag ? C.blue : C.border}`, borderRadius: 10, padding: "24px 16px", textAlign: "center", background: drag ? "#EAF4FB" : C.paper, cursor: "pointer", transition: "all 0.2s" }}
      onClick={() => ref.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        process(e.dataTransfer.files[0]);
      }}
    >
      <input ref={ref} type="file" accept=".csv,.xlsx,.xls" style={{ display: "none" }} onChange={(e) => process(e.target.files[0])} />
      <div style={{ fontSize: 32, marginBottom: 6 }}>Data</div>
      <div style={{ fontWeight: 700, color: C.navy, fontSize: 14 }}>Drop file or click to upload</div>
      <div style={{ fontSize: 12, color: C.gray, marginTop: 3 }}>CSV - XLSX - XLS</div>
    </div>
  );
}

function MetricsTable({ rows }) {
  const cols = ["RMSE", "MAE", "MAPE", "sMAPE", "MBE", "R2", "VAF"];
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ background: C.navy, color: "white" }}>
            <th style={{ padding: "7px 8px", textAlign: "left" }}>Model Group</th>
            <th style={{ padding: "7px 8px", textAlign: "left" }}>Model</th>
            {cols.map((c) => <th key={c} style={{ padding: "7px 8px", textAlign: "left" }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.key} style={{ background: i % 2 === 0 ? C.paper : C.bg }}>
              <td style={{ padding: "6px 8px", color: C.gray }}>{r.group}</td>
              <td style={{ padding: "6px 8px", fontWeight: 700, color: C.navy }}>{cleanModelLabel(r.name)}</td>
              {cols.map((c) => <td key={c} style={{ padding: "6px 8px", fontFamily: "monospace" }}>{formatMetric(r.metrics[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ClusterMetricsTable({ rows }) {
  const cols = ["Silhouette", "DaviesBouldin", "CalinskiHarabasz", "Inertia"];
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ background: C.navy, color: "white" }}>
            <th style={{ padding: "7px 8px", textAlign: "left" }}>Model</th>
            {cols.map((c) => <th key={c} style={{ padding: "7px 8px", textAlign: "left" }}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.key} style={{ background: i % 2 === 0 ? C.paper : C.bg }}>
              <td style={{ padding: "6px 8px", fontWeight: 700, color: C.navy }}>{r.name}</td>
              {cols.map((c) => <td key={c} style={{ padding: "6px 8px", fontFamily: "monospace" }}>{formatMetric(r.metrics[c])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ImportanceBar({ features, meanAbs, meanSigned, title = "Feature Importance" }) {
  const order = [...meanAbs.map((v, i) => ({ v, i }))].sort((a, b) => b.v - a.v).map((d) => d.i);
  const top = order.slice(0, 12);
  const maxV = Math.max(...top.map((i) => meanAbs[i]), 1e-9);

  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>{title}</div>
      {top.map((fi) => {
        const w = (meanAbs[fi] / maxV) * 100;
        const col = meanSigned[fi] >= 0 ? C.pos : C.neg;
        return (
          <div key={fi} style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 11, color: C.ink, marginBottom: 3 }}>{features[fi]}</div>
            <div style={{ height: 14, background: C.light, borderRadius: 4, overflow: "hidden" }}>
              <div style={{ width: `${w}%`, height: "100%", background: col }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ShapBeeswarm({ features, shapVals, Xraw, orderImp }) {
  const top = orderImp.slice(0, 8);
  const W = 640;
  const H = 320;
  const pL = 170;
  const pR = 26;
  const pT = 20;
  const pB = 40;
  const rowH = (H - pT - pB) / Math.max(top.length, 1);
  const all = shapVals.flat();
  const mn = Math.min(...all);
  const mx = Math.max(...all);
  const sx = (v) => pL + ((v - mn) / (mx - mn || 1)) * (W - pL - pR);

  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>SHAP Beeswarm Summary</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block", background: C.paper }} xmlns="http://www.w3.org/2000/svg">
        {top.map((fi, row) => {
          const y = pT + rowH * row + rowH / 2;
          const vals = Xraw.map((r) => r[fi]);
          const fmin = Math.min(...vals);
          const fmax = Math.max(...vals);
          return (
            <g key={fi}>
              <text x={pL - 8} y={y + 4} textAnchor="end" fontSize={11} fill={C.ink}>{features[fi]}</text>
              {shapVals.map((r, i) => {
                const t = (vals[i] - fmin) / (fmax - fmin || 1);
                const c = `rgb(${Math.round(40 + 215 * t)},${Math.round(130 + 90 * (1 - t))},${Math.round(210 - 180 * t)})`;
                const jitter = (((i * 17 + fi * 11) % 100) / 100 - 0.5) * rowH * 0.65;
                return <circle key={i} cx={sx(r[fi])} cy={y + jitter} r={2.7} fill={c} opacity={0.85} />;
              })}
            </g>
          );
        })}
        <line x1={sx(0)} y1={pT} x2={sx(0)} y2={H - pB} stroke={C.border} strokeDasharray="3,3" />
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fill={C.gray}>SHAP value</text>
      </svg>
    </div>
  );
}

function ShapHeatmap({ features, shapVals, orderImp }) {
  const top = orderImp.slice(0, 10);
  const n = shapVals.length;
  const W = Math.min(760, Math.max(520, n * 9 + 200));
  const H = 300;
  const pL = 170;
  const pT = 24;
  const pB = 28;
  const pR = 20;
  const cellW = (W - pL - pR) / Math.max(n, 1);
  const cellH = (H - pT - pB) / Math.max(top.length, 1);
  const all = shapVals.flat();
  const absMax = Math.max(Math.abs(Math.min(...all)), Math.abs(Math.max(...all)), 1e-9);

  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>SHAP Heatmap</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block", background: C.paper }} xmlns="http://www.w3.org/2000/svg">
        {top.map((fi, r) => (
          <g key={fi}>
            <text x={pL - 6} y={pT + r * cellH + cellH / 2 + 4} textAnchor="end" fontSize={10} fill={C.ink}>{features[fi]}</text>
            {Array.from({ length: n }, (_, c) => {
              const sv = shapVals[c][fi];
              const t = 0.5 + sv / absMax / 2;
              const col = `rgb(${Math.round(40 + 215 * t)},${Math.round(120 + 60 * (1 - Math.abs(t - 0.5) * 2))},${Math.round(210 - 170 * t)})`;
              return <rect key={c} x={pL + c * cellW} y={pT + r * cellH} width={cellW + 0.2} height={cellH - 0.2} fill={col} />;
            })}
          </g>
        ))}
        <text x={W / 2} y={H - 7} textAnchor="middle" fontSize={11} fill={C.gray}>Observations</text>
      </svg>
    </div>
  );
}

function ScatterPlot({ y, pred, title }) {
  const W = 560;
  const H = 330;
  const pL = 52;
  const pR = 18;
  const pT = 20;
  const pB = 42;
  const minV = Math.min(...y, ...pred);
  const maxV = Math.max(...y, ...pred);
  const sx = (v) => pL + ((v - minV) / (maxV - minV || 1)) * (W - pL - pR);
  const sy = (v) => H - pB - ((v - minV) / (maxV - minV || 1)) * (H - pT - pB);

  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block", background: C.paper }} xmlns="http://www.w3.org/2000/svg">
        <line x1={pL} y1={H - pB} x2={W - pR} y2={H - pB} stroke={C.border} />
        <line x1={pL} y1={H - pB} x2={pL} y2={pT} stroke={C.border} />
        <line x1={sx(minV)} y1={sy(minV)} x2={sx(maxV)} y2={sy(maxV)} stroke={C.gray} strokeDasharray="4,3" />
        {y.map((v, i) => <circle key={i} cx={sx(v)} cy={sy(pred[i])} r={3} fill={C.accent} opacity={0.8} />)}
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fill={C.gray}>Actual</text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={11} fill={C.gray} transform={`rotate(-90,12,${H / 2})`}>Predicted</text>
      </svg>
    </div>
  );
}

function ResidualBars({ y, pred }) {
  const residual = y.map((v, i) => pred[i] - v);
  const bins = 14;
  const mn = Math.min(...residual);
  const mx = Math.max(...residual);
  const step = (mx - mn || 1) / bins;
  const hist = Array.from({ length: bins }, () => 0);
  residual.forEach((r) => {
    const id = clamp(Math.floor((r - mn) / step), 0, bins - 1);
    hist[id] += 1;
  });
  const maxH = Math.max(...hist, 1);

  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>Residual Distribution</div>
      <div style={{ display: "flex", gap: 4, alignItems: "flex-end", height: 140, padding: "8px 6px", border: `1px solid ${C.border}`, borderRadius: 6, background: "#FBFCFE" }}>
        {hist.map((h, i) => <div key={i} style={{ flex: 1, height: `${(h / maxH) * 100}%`, background: C.mid, borderRadius: 2, opacity: 0.8 }} />)}
      </div>
    </div>
  );
}

function ClusterScatter({ projection, labels, title }) {
  const W = 640;
  const H = 360;
  const pL = 44;
  const pR = 14;
  const pT = 20;
  const pB = 34;
  const xs = projection.map((d) => d.x);
  const ys = projection.map((d) => d.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const sx = (v) => pL + ((v - xMin) / (xMax - xMin || 1)) * (W - pL - pR);
  const sy = (v) => H - pB - ((v - yMin) / (yMax - yMin || 1)) * (H - pT - pB);

  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block", background: C.paper }} xmlns="http://www.w3.org/2000/svg">
        <line x1={pL} y1={H - pB} x2={W - pR} y2={H - pB} stroke={C.border} />
        <line x1={pL} y1={H - pB} x2={pL} y2={pT} stroke={C.border} />
        {projection.map((d, i) => {
          const t = (labels[i] * 37) % 255;
          const col = `rgb(${80 + (t % 140)},${60 + ((t * 2) % 160)},${110 + ((t * 3) % 130)})`;
          return <circle key={i} cx={sx(d.x)} cy={sy(d.y)} r={3} fill={col} opacity={0.85} />;
        })}
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize={11} fill={C.gray}>PC1</text>
        <text x={12} y={H / 2} textAnchor="middle" fontSize={11} fill={C.gray} transform={`rotate(-90,12,${H / 2})`}>PC2</text>
      </svg>
    </div>
  );
}

function ClusterSizeBars({ labels }) {
  const k = Math.max(...labels) + 1;
  const counts = Array.from({ length: k }, (_, c) => labels.filter((v) => v === c).length);
  const maxC = Math.max(...counts, 1);
  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>Cluster Sizes</div>
      {counts.map((c, i) => (
        <div key={i} style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 11, color: C.ink, marginBottom: 3 }}>Cluster {i + 1} ({c})</div>
          <div style={{ height: 14, background: C.light, borderRadius: 4, overflow: "hidden" }}>
            <div style={{ width: `${(c / maxC) * 100}%`, height: "100%", background: C.mid }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function DataProfileTable({ profile }) {
  if (!profile) return null;
  const cols = ["feature", "count", "missing", "mean", "std", "min", "q1", "median", "q3", "max"];
  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>Data Profiling</div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ background: C.navy, color: "white" }}>
              {cols.map((c) => <th key={c} style={{ padding: "7px 8px", textAlign: "left" }}>{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {profile.stats.map((r, i) => (
              <tr key={r.feature} style={{ background: i % 2 === 0 ? C.paper : C.bg }}>
                {cols.map((c) => <td key={c} style={{ padding: "6px 8px", fontFamily: c === "feature" ? "inherit" : "monospace" }}>{typeof r[c] === "number" ? formatMetric(r[c]) : r[c]}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function CorrelationHeatmap({ profile }) {
  if (!profile) return null;
  const cols = profile.columns.slice(0, 14);
  const idx = cols.map((c) => profile.columns.indexOf(c));
  const mat = idx.map((i) => idx.map((j) => profile.corr[i][j]));
  const n = cols.length;
  const W = 660;
  const H = 420;
  const pL = 130;
  const pT = 28;
  const pR = 14;
  const pB = 120;
  const cellW = (W - pL - pR) / Math.max(n, 1);
  const cellH = (H - pT - pB) / Math.max(n, 1);

  return (
    <div>
      <div style={{ fontWeight: 700, color: C.navy, marginBottom: 8 }}>Correlation Heatmap</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, display: "block", background: C.paper }} xmlns="http://www.w3.org/2000/svg">
        {Array.from({ length: n }, (_, r) => (
          <g key={r}>
            <text x={pL - 6} y={pT + r * cellH + cellH / 2 + 4} textAnchor="end" fontSize={9} fill={C.ink}>{cols[r]}</text>
            {Array.from({ length: n }, (_, c) => {
              const v = mat[r][c];
              const t = (v + 1) / 2;
              const col = `rgb(${Math.round(40 + 215 * t)},${Math.round(120 + 60 * (1 - Math.abs(t - 0.5) * 2))},${Math.round(210 - 170 * t)})`;
              return <rect key={c} x={pL + c * cellW} y={pT + r * cellH} width={cellW + 0.4} height={cellH + 0.4} fill={col} />;
            })}
          </g>
        ))}
        {cols.map((c, i) => <text key={c} x={pL + i * cellW + 2} y={H - 70} transform={`rotate(55 ${pL + i * cellW + 2} ${H - 70})`} fontSize={9} fill={C.ink}>{c}</text>)}
      </svg>
    </div>
  );
}

function exportMetricsWorkbook(rows) {
  if (!window.XLSX) return;
  const cols = ["group", "model", "RMSE", "MAE", "MAPE", "sMAPE", "MBE", "R2", "VAF", "MSE", "MAD"];
  const data = rows.map((r) => ({ group: r.group, model: cleanModelLabel(r.name), ...r.metrics }));
  const ws = window.XLSX.utils.json_to_sheet(data, { header: cols });
  const wb = window.XLSX.utils.book_new();
  window.XLSX.utils.book_append_sheet(wb, ws, "metrics");
  window.XLSX.writeFile(wb, "research_metrics.xlsx");
}

function exportPredictionsWorkbook(rows, yName) {
  if (!window.XLSX) return;
  const wb = window.XLSX.utils.book_new();
  rows.forEach((r) => {
    const sheetRows = r.actual.map((v, i) => ({ index: i + 1, [yName]: v, predicted_full_fit: r.predFull[i], predicted_kfold_oof: r.predOOF[i] }));
    const ws = window.XLSX.utils.json_to_sheet(sheetRows);
    const safeName = r.name.replace(/[^A-Za-z0-9_]/g, "_").slice(0, 28) || "model";
    window.XLSX.utils.book_append_sheet(wb, ws, safeName);
  });
  window.XLSX.writeFile(wb, "actual_vs_predicted_models.xlsx");
}

function exportClusterWorkbook(clusterRows, baseRows) {
  if (!window.XLSX) return;
  const wb = window.XLSX.utils.book_new();
  clusterRows.forEach((r) => {
    const rows = baseRows.map((src, i) => ({ index: i + 1, ...src, cluster: r.labels[i] + 1 }));
    const ws = window.XLSX.utils.json_to_sheet(rows);
    const safeName = r.name.replace(/[^A-Za-z0-9_]/g, "_").slice(0, 28) || "cluster";
    window.XLSX.utils.book_append_sheet(wb, ws, safeName);
  });
  window.XLSX.writeFile(wb, "clustering_labels.xlsx");
}

function exportProfileWorkbook(profile) {
  if (!window.XLSX || !profile) return;
  const wb = window.XLSX.utils.book_new();
  const wsStats = window.XLSX.utils.json_to_sheet(profile.stats);
  window.XLSX.utils.book_append_sheet(wb, wsStats, "profile");

  const corrRows = profile.corr.map((row, i) => {
    const obj = { feature: profile.columns[i] };
    profile.columns.forEach((c, j) => { obj[c] = row[j]; });
    return obj;
  });
  const wsCorr = window.XLSX.utils.json_to_sheet(corrRows);
  window.XLSX.utils.book_append_sheet(wb, wsCorr, "correlation");
  window.XLSX.writeFile(wb, "data_profile.xlsx");
}

function parseNum(v, fallback) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export default function App() {
  const xlsxReady = useXLSX();
  const [rawData, setRawData] = useState(null);
  const [fileName, setFileName] = useState("");
  const [columns, setColumns] = useState([]);
  const [targetCol, setTargetCol] = useState("");
  const [analysisMode, setAnalysisMode] = useState("regression");

  const [results, setResults] = useState([]);
  const [clusterResults, setClusterResults] = useState([]);
  const [featureNames, setFeatureNames] = useState([]);
  const [bestModelKey, setBestModelKey] = useState("");
  const [selectedModelKey, setSelectedModelKey] = useState("");
  const [selectedClusterKey, setSelectedClusterKey] = useState("");
  const [shapData, setShapData] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(false);

  const [kFold, setKFold] = useState(5);
  const [cvSeed, setCvSeed] = useState(42);
  const [hybridAlgo, setHybridAlgo] = useState("de");
  const [hybridPop, setHybridPop] = useState(10);
  const [hybridGen, setHybridGen] = useState(8);
  const [hybridParamA, setHybridParamA] = useState(0.7);
  const [hybridParamB, setHybridParamB] = useState(0.9);

  const [clusterModelMode, setClusterModelMode] = useState("all");
  const [clusterK, setClusterK] = useState(3);
  const [clusterIter, setClusterIter] = useState(80);
  const [categoryTab, setCategoryTab] = useState("exploration");
  const [activeTab, setActiveTab] = useState("profile");
  const [shapRowIndex, setShapRowIndex] = useState(0);
  const visualRef = useRef(null);

  const handleData = useCallback((rows, fname) => {
    setRawData(rows);
    setFileName(fname);
    const cols = Object.keys(rows[0]).filter((k) => rows.every((r) => typeof r[k] === "number" && Number.isFinite(r[k])));
    setColumns(cols);
    const target = cols[cols.length - 1] || "";
    setTargetCol(target);
    
    const profile = profileNumericData(rows, cols);
    
    // Compute analytical SHAP for the target column if available
    if (target && cols.length > 1) {
      const feats = cols.filter((c) => c !== target);
      const X = rows.map((r) => feats.map((f) => r[f]));
      const y = rows.map((r) => r[target]);
      try {
        const ols = computeAnalyticalShap(X, y);
        profile.olsAnalysis = {
          intercept: ols.intercept,
          coefs: ols.coefs,
          r2: ols.r2,
          adjR2: ols.adjR2,
          rmse: ols.rmse,
          yPred: ols.yPred,
          shapVals: ols.shapVals,
          meanAbs: ols.meanAbs,
          orderImp: ols.orderImp
        };
      } catch (e) {
        console.log("Analytical SHAP computation skipped:", e.message);
      }
    }
    
    setProfile(profile);
    setResults([]);
    setClusterResults([]);
    setShapData(null);
  }, []);

  const trainSpecs = useMemo(() => ({
    "AdaBoost": {
      standalone: { nEstimators: 65, learningRate: 0.2, bins: 14 },
      spec: [
        { name: "nEstimators", min: 20, max: 180, int: true },
        { name: "learningRate", min: 0.03, max: 0.7, int: false },
        { name: "bins", min: 8, max: 30, int: true }
      ],
      train: trainAdaBoostRegressor
    },
    "XGBoost-like": {
      standalone: { nEstimators: 120, learningRate: 0.08, bins: 16, subsample: 0.82 },
      spec: [
        { name: "nEstimators", min: 40, max: 220, int: true },
        { name: "learningRate", min: 0.02, max: 0.3, int: false },
        { name: "bins", min: 8, max: 34, int: true },
        { name: "subsample", min: 0.55, max: 1, int: false }
      ],
      train: trainXGBoostLike
    },
    "CatBoost-like": {
      standalone: { nEstimators: 130, learningRate: 0.07, bins: 15, featureSubsample: 0.72 },
      spec: [
        { name: "nEstimators", min: 50, max: 240, int: true },
        { name: "learningRate", min: 0.015, max: 0.25, int: false },
        { name: "bins", min: 8, max: 30, int: true },
        { name: "featureSubsample", min: 0.35, max: 1, int: false }
      ],
      train: trainCatBoostLike
    }
  }), []);

  const hybridMeta = useMemo(() => {
    if (hybridAlgo === "pso") return { tag: "PSO", title: "Particle Swarm Optimization (PSO)", a: "Inertia (w)", b: "Acceleration (c1=c2)", dA: 0.72, dB: 1.5 };
    if (hybridAlgo === "ga") return { tag: "GA", title: "Genetic Algorithm (GA)", a: "Mutation Rate", b: "Crossover Rate", dA: 0.1, dB: 0.85 };
    return { tag: "DE", title: "Differential Evolution (DE)", a: "F", b: "CR", dA: 0.7, dB: 0.9 };
  }, [hybridAlgo]);

  const runRegressionResearch = useCallback(() => {
    if (!rawData || !targetCol) return;
    setLoading(true);

    setTimeout(() => {
      try {
        const feats = columns.filter((c) => c !== targetCol);
        const X = rawData.map((r) => feats.map((f) => r[f]));
        const y = rawData.map((r) => r[targetCol]);
        const cvOpts = {
          k: clamp(Math.round(parseNum(kFold, 5)), 2, Math.min(10, Math.max(2, y.length - 1))),
          shuffle: true,
          seed: clamp(Math.round(parseNum(cvSeed, 42)), 1, 999999)
        };

        const out = [];
        Object.entries(trainSpecs).forEach(([name, spec], idx) => {
          const trainStandalone = (Xt, yt, seed) => spec.train(Xt, yt, spec.standalone, seed);
          const cvStandalone = runKFold(trainStandalone, X, y, cvOpts);
          const fullStandalone = trainStandalone(X, y, cvOpts.seed + idx + 1000);
          const predFull = fullStandalone.predict(X);

          out.push({
            key: `${name}-standalone`,
            group: "Standalone",
            name,
            metrics: cvStandalone.metrics,
            predOOF: cvStandalone.oof,
            predFull,
            actual: y,
            model: fullStandalone,
            params: spec.standalone
          });

          const objective = (vec) => {
            const params = decodeParams(vec, spec.spec);
            const trainer = (Xt, yt, seed) => spec.train(Xt, yt, params, seed);
            return runKFold(trainer, X, y, cvOpts).metrics.RMSE;
          };

          const optimized = optimizeHyperparams({
            algorithm: hybridAlgo,
            objective,
            dims: spec.spec.length,
            popSize: clamp(Math.round(parseNum(hybridPop, 10)), 4, 24),
            generations: clamp(Math.round(parseNum(hybridGen, 8)), 2, 40),
            paramA: parseNum(hybridParamA, hybridMeta.dA),
            paramB: parseNum(hybridParamB, hybridMeta.dB),
            seed: cvOpts.seed + idx * 97 + 23
          });

          const bestParams = decodeParams(optimized.vector, spec.spec);
          const trainHybrid = (Xt, yt, seed) => spec.train(Xt, yt, bestParams, seed);
          const cvHybrid = runKFold(trainHybrid, X, y, cvOpts);
          const fullHybrid = trainHybrid(X, y, cvOpts.seed + idx + 2000);

          out.push({
            key: `${name}-${hybridAlgo}`,
            group: "Hybrid",
            name: `${name}-${hybridMeta.tag}`,
            metrics: cvHybrid.metrics,
            predOOF: cvHybrid.oof,
            predFull: fullHybrid.predict(X),
            actual: y,
            model: fullHybrid,
            params: bestParams
          });
        });

        out.sort((a, b) => a.metrics.RMSE - b.metrics.RMSE);
        const best = out[0];
        const shap = computeApproxShap(best.model, X);

        setFeatureNames(feats);
        setResults(out);
        setClusterResults([]);
        setBestModelKey(best.key);
        setSelectedModelKey(best.key);
        setShapData(shap);
        setActiveTab("metrics");
      } catch (e) {
        alert(`Research pipeline error: ${e.message}`);
      }
      setLoading(false);
    }, 40);
  }, [rawData, targetCol, columns, kFold, cvSeed, hybridAlgo, hybridPop, hybridGen, hybridParamA, hybridParamB, hybridMeta, trainSpecs]);

  const runClustering = useCallback(() => {
    if (!rawData || !columns.length) return;
    setLoading(true);

    setTimeout(() => {
      try {
        const feats = columns.filter((c) => c !== targetCol);
        const used = feats.length >= 2 ? feats : columns;
        const Xraw = rawData.map((r) => used.map((f) => r[f]));
        const { Xn } = normalizeMatrix(Xraw);
        const k = clamp(Math.round(parseNum(clusterK, 3)), 2, 12);
        const maxIter = clamp(Math.round(parseNum(clusterIter, 80)), 20, 300);
        const seed = clamp(Math.round(parseNum(cvSeed, 42)), 1, 999999);

        const out = [];
        if (clusterModelMode === "all" || clusterModelMode === "kmeans") {
          const km = runKMeans(Xn, k, maxIter, seed + 9);
          out.push({
            key: "kmeans",
            name: "KMeans",
            labels: km.labels,
            centroids: km.centroids,
            metrics: computeClusteringMetrics(Xn, km.labels, km.centroids),
            projection: pca2D(Xn),
            featureImportance: featureClusterSeparation(Xn, km.labels, used)
          });
        }

        if (clusterModelMode === "all" || clusterModelMode === "fcm") {
          const fcm = runFuzzyCMeans(Xn, k, maxIter, 2, seed + 19);
          out.push({
            key: "fcm",
            name: "Fuzzy C-Means",
            labels: fcm.labels,
            centroids: fcm.centroids,
            metrics: computeClusteringMetrics(Xn, fcm.labels, fcm.centroids),
            projection: pca2D(Xn),
            featureImportance: featureClusterSeparation(Xn, fcm.labels, used)
          });
        }

        out.sort((a, b) => b.metrics.Silhouette - a.metrics.Silhouette);
        setFeatureNames(used);
        setClusterResults(out);
        setResults([]);
        setShapData(null);
        setSelectedClusterKey(out[0]?.key || "");
        setActiveTab("cluster_metrics");
      } catch (e) {
        alert(`Clustering pipeline error: ${e.message}`);
      }
      setLoading(false);
    }, 40);
  }, [rawData, columns, targetCol, clusterK, clusterIter, clusterModelMode, cvSeed]);

  useEffect(() => {
    if (!results.length || !selectedModelKey || !rawData || !featureNames.length) return;
    const picked = results.find((r) => r.key === selectedModelKey);
    if (!picked) return;
    const X = rawData.map((r) => featureNames.map((f) => r[f]));
    setShapData(computeApproxShap(picked.model, X));
  }, [selectedModelKey, results, rawData, featureNames]);

  const selectedResult = results.find((r) => r.key === selectedModelKey) || null;
  const selectedCluster = clusterResults.find((r) => r.key === selectedClusterKey) || clusterResults[0] || null;

  // Category tabs for data exploration, SHAP analysis, and modeling
  // const [categoryTab, setCategoryTab] = useState("exploration");

  const explorationTabs = useMemo(() => [
    { id: "profile", label: "Data Profile" },
    { id: "corr", label: "Correlation Heatmap" }
  ], []);

  const shapTabs = useMemo(() => [
    { id: "shap_importance", label: "Feature Importance" },
    { id: "shap_signed", label: "Signed Direction" },
    { id: "shap_coverage", label: "Coverage Curve" },
    { id: "shap_beeswarm", label: "Beeswarm Plot" },
    { id: "shap_heatmap", label: "Heatmap" },
    { id: "shap_dependence", label: "Dependence Plot" },
    { id: "shap_distribution", label: "Impact Dist." },
    { id: "shap_sign_balance", label: "Sign Balance" },
    { id: "shap_cohort", label: "Cohort Contrast" },
    { id: "shap_table", label: "SHAP Table" },
    { id: "shap_waterfall", label: "Local Waterfall" }
    ,{ id: "shap_force", label: "Force Plot" }
  ], []);

  const modelingTabs = useMemo(() => {
    if (analysisMode === "clustering") {
      return [
        { id: "cluster_metrics", label: "Metrics" },
        { id: "cluster_scatter", label: "Cluster Scatter" },
        { id: "cluster_sizes", label: "Cluster Sizes" },
        { id: "cluster_importance", label: "Feature Separation" }
      ];
    }
    return [
      { id: "metrics", label: "Model Metrics" },
      { id: "pred", label: "Predictions vs Actual" },
      { id: "residual", label: "Residual Analysis" }
    ];
  }, [analysisMode]);

  // Determine which tabs to show based on category
  const getTabs = () => {
    if (categoryTab === "exploration") return explorationTabs;
    if (categoryTab === "shap") return shapTabs;
    return modelingTabs;
  };

  const tabs = getTabs();
  const featureCols = columns.filter((c) => c !== targetCol);
  const shapMaxRow = Math.max((profile?.olsAnalysis?.shapVals?.length || 1) - 1, 0);
  const shapMeanSigned = useMemo(() => {
    const sv = profile?.olsAnalysis?.shapVals || [];
    if (!sv.length) return [];
    return featureCols.map((_, j) => vMean(sv.map((r) => r[j])));
  }, [profile, featureCols]);
  const reportParagraph = useMemo(() => buildReportParagraph({
    activeTab,
    analysisMode,
    targetCol,
    selectedResult,
    selectedCluster,
    shapRowIndex: Math.min(shapRowIndex, shapMaxRow)
  }), [activeTab, analysisMode, targetCol, selectedResult, selectedCluster, shapRowIndex, shapMaxRow]);

  return (
    <div style={{ minHeight: "100vh", background: C.bg, fontFamily: "Georgia,serif" }}>
      <div style={{ background: C.navy, padding: "14px 28px", display: "flex", alignItems: "center", justifyContent: "space-between", boxShadow: "0 2px 8px rgba(0,0,0,0.15)" }}>
        <div>
          <div style={{ color: "white", fontSize: 18, fontWeight: 700 }}>SHAP Research Studio</div>
          <div style={{ color: "#A5C8E1", fontSize: 11, marginTop: 1 }}>Regression + clustering + data profiling + extended SHAP visuals</div>
        </div>
        {(results.length > 0 || clusterResults.length > 0) && (
          <div style={{ color: "#A5C8E1", fontSize: 12, fontFamily: "monospace" }}>{fileName} · mode={analysisMode}</div>
        )}
      </div>

      <div style={{ maxWidth: 1280, margin: "0 auto", padding: "22px 18px", display: "grid", gridTemplateColumns: (results.length || clusterResults.length || profile) ? "320px 1fr" : "1fr", gap: 18, alignItems: "start" }}>
        <div>
          <div style={cardStyle}>
            <div style={{ fontWeight: 700, color: C.navy, marginBottom: 10, fontSize: 13 }}>1 - Upload Data</div>
            <UploadPanel onData={handleData} />
            {fileName && <div style={{ fontSize: 11, color: C.gray, marginTop: 7, fontFamily: "monospace", wordBreak: "break-all" }}>{fileName}</div>}
          </div>

          {columns.length > 0 && (
            <>
              <div style={cardStyle}>
                <div style={{ fontWeight: 700, color: C.navy, marginBottom: 10, fontSize: 13 }}>2 - Analysis Mode</div>
                <select value={analysisMode} onChange={(e) => { setAnalysisMode(e.target.value); setActiveTab(e.target.value === "clustering" ? "cluster_metrics" : "metrics"); }} style={{ width: "100%", padding: "7px 8px", borderRadius: 5, border: `1px solid ${C.border}`, marginBottom: 8 }}>
                  <option value="regression">Regression (standalone + hybrid)</option>
                  <option value="clustering">Clustering models</option>
                </select>

                <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>Target variable</div>
                <select value={targetCol} onChange={(e) => setTargetCol(e.target.value)} style={{ width: "100%", padding: "7px 8px", borderRadius: 5, border: `1px solid ${C.border}`, marginBottom: 8 }}>
                  {columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  <div>
                    <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>{analysisMode === "clustering" ? "clusters (k)" : "k-fold"}</div>
                    <input value={analysisMode === "clustering" ? clusterK : kFold} onChange={(e) => analysisMode === "clustering" ? setClusterK(e.target.value) : setKFold(e.target.value)} style={{ width: "100%", padding: 6, borderRadius: 4, border: `1px solid ${C.border}` }} />
                  </div>
                  <div>
                    <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>seed</div>
                    <input value={cvSeed} onChange={(e) => setCvSeed(e.target.value)} style={{ width: "100%", padding: 6, borderRadius: 4, border: `1px solid ${C.border}` }} />
                  </div>
                </div>
              </div>

              {analysisMode === "regression" ? (
                <div style={cardStyle}>
                  <div style={{ fontWeight: 700, color: C.navy, marginBottom: 10, fontSize: 13 }}>3 - Hybrid Optimizer</div>
                  <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>Algorithm</div>
                  <select value={hybridAlgo} onChange={(e) => {
                    const next = e.target.value;
                    setHybridAlgo(next);
                    if (next === "pso") { setHybridParamA(0.72); setHybridParamB(1.5); }
                    else if (next === "ga") { setHybridParamA(0.1); setHybridParamB(0.85); }
                    else { setHybridParamA(0.7); setHybridParamB(0.9); }
                  }} style={{ width: "100%", padding: "7px 8px", borderRadius: 5, border: `1px solid ${C.border}`, marginBottom: 8 }}>
                    <option value="de">Differential Evolution (DE)</option>
                    <option value="pso">Particle Swarm Optimization (PSO)</option>
                    <option value="ga">Genetic Algorithm (GA)</option>
                  </select>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <div>
                      <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>population</div>
                      <input value={hybridPop} onChange={(e) => setHybridPop(e.target.value)} style={{ width: "100%", padding: 6, borderRadius: 4, border: `1px solid ${C.border}` }} />
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>generations</div>
                      <input value={hybridGen} onChange={(e) => setHybridGen(e.target.value)} style={{ width: "100%", padding: 6, borderRadius: 4, border: `1px solid ${C.border}` }} />
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>{hybridMeta.a}</div>
                      <input value={hybridParamA} onChange={(e) => setHybridParamA(e.target.value)} style={{ width: "100%", padding: 6, borderRadius: 4, border: `1px solid ${C.border}` }} />
                    </div>
                    <div>
                      <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>{hybridMeta.b}</div>
                      <input value={hybridParamB} onChange={(e) => setHybridParamB(e.target.value)} style={{ width: "100%", padding: 6, borderRadius: 4, border: `1px solid ${C.border}` }} />
                    </div>
                  </div>
                  <button onClick={runRegressionResearch} disabled={loading || !targetCol} style={{ marginTop: 12, width: "100%", padding: "10px 0", background: loading ? C.gray : C.blue, color: "white", border: "none", borderRadius: 6, fontSize: 13, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer" }}>
                    {loading ? "Running analytics..." : "Run Regression Analytics"}
                  </button>
                </div>
              ) : (
                <div style={cardStyle}>
                  <div style={{ fontWeight: 700, color: C.navy, marginBottom: 10, fontSize: 13 }}>3 - Clustering Options</div>
                  <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>Clustering model(s)</div>
                  <select value={clusterModelMode} onChange={(e) => setClusterModelMode(e.target.value)} style={{ width: "100%", padding: "7px 8px", borderRadius: 5, border: `1px solid ${C.border}`, marginBottom: 8 }}>
                    <option value="all">Run all clustering models</option>
                    <option value="kmeans">KMeans</option>
                    <option value="fcm">Fuzzy C-Means</option>
                  </select>
                  <div style={{ fontSize: 11, color: C.gray, marginBottom: 4 }}>max iterations</div>
                  <input value={clusterIter} onChange={(e) => setClusterIter(e.target.value)} style={{ width: "100%", padding: 6, borderRadius: 4, border: `1px solid ${C.border}` }} />
                  <button onClick={runClustering} disabled={loading || !columns.length} style={{ marginTop: 12, width: "100%", padding: "10px 0", background: loading ? C.gray : C.blue, color: "white", border: "none", borderRadius: 6, fontSize: 13, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer" }}>
                    {loading ? "Running clustering..." : "Run Clustering Models"}
                  </button>
                </div>
              )}
            </>
          )}

          {(results.length > 0 || clusterResults.length > 0 || profile) && (
            <div style={cardStyle}>
              <div style={{ fontWeight: 700, color: C.navy, marginBottom: 10, fontSize: 13 }}>4 - Exports</div>
              {analysisMode === "regression" && results.length > 0 && (
                <>
                  <button onClick={() => exportMetricsWorkbook(results)} disabled={!xlsxReady} style={{ width: "100%", padding: "8px 0", marginBottom: 8, border: "none", borderRadius: 6, background: C.navy, color: "white", cursor: "pointer" }}>Download Metrics Excel</button>
                  <button onClick={() => exportPredictionsWorkbook(results, targetCol)} disabled={!xlsxReady} style={{ width: "100%", padding: "8px 0", marginBottom: 8, border: "none", borderRadius: 6, background: C.mid, color: "white", cursor: "pointer" }}>Download Actual vs Predicted Excel</button>
                </>
              )}
              {analysisMode === "clustering" && clusterResults.length > 0 && (
                <button onClick={() => exportClusterWorkbook(clusterResults, rawData)} disabled={!xlsxReady} style={{ width: "100%", padding: "8px 0", marginBottom: 8, border: "none", borderRadius: 6, background: C.mid, color: "white", cursor: "pointer" }}>Download Cluster Labels Excel</button>
              )}
              {profile && (
                <button onClick={() => exportProfileWorkbook(profile)} disabled={!xlsxReady} style={{ width: "100%", padding: "8px 0", border: "none", borderRadius: 6, background: C.blue, color: "white", cursor: "pointer" }}>Download Data Profile Excel</button>
              )}
            </div>
          )}
        </div>

        {(results.length > 0 || clusterResults.length > 0 || profile) ? (
          <div>
            <div style={{ ...cardStyle, marginBottom: 10 }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                {analysisMode === "regression" && results.length > 0 && (
                  <>
                    <div style={{ fontSize: 12, color: C.gray }}>SHAP model:</div>
                    <select value={selectedModelKey} onChange={(e) => setSelectedModelKey(e.target.value)} style={{ padding: "6px 8px", borderRadius: 5, border: `1px solid ${C.border}` }}>
                      {results.map((r) => <option key={r.key} value={r.key}>{r.name.replace(/-like\b/gi, "")}</option>)}
                    </select>
                    {bestModelKey === selectedModelKey && <span style={{ background: "#E8F6EF", color: "#1E8449", padding: "3px 8px", borderRadius: 10, fontSize: 11, fontWeight: 700 }}>best RMSE</span>}
                  </>
                )}
                {analysisMode === "clustering" && clusterResults.length > 0 && (
                  <>
                    <div style={{ fontSize: 12, color: C.gray }}>Clustering model:</div>
                    <select value={selectedClusterKey} onChange={(e) => setSelectedClusterKey(e.target.value)} style={{ padding: "6px 8px", borderRadius: 5, border: `1px solid ${C.border}` }}>
                      {clusterResults.map((r) => <option key={r.key} value={r.key}>{r.name}</option>)}
                    </select>
                  </>
                )}
              </div>

              <div style={{ display: "flex", gap: 6, marginTop: 10, flexWrap: "wrap" }}>
                <div style={{ display: "flex", gap: 3, flexWrap: "wrap", marginBottom: 10 }}>
                  <div style={{ fontSize: 11, color: C.gray, marginRight: 8, alignSelf: "center" }}>Analysis Category:</div>
                  {(analysisMode === "regression" || analysisMode === "clustering") && (
                    <>
                      <button onClick={() => { setCategoryTab("exploration"); setActiveTab("profile"); }} style={{ padding: "6px 12px", borderRadius: 18, border: "none", cursor: "pointer", background: categoryTab === "exploration" ? C.navy : C.light, color: categoryTab === "exploration" ? "white" : C.gray, fontWeight: categoryTab === "exploration" ? 700 : 500, fontSize: 12 }}>📊 Data Exploration</button>
                      {(results.length > 0 || clusterResults.length > 0) && <button onClick={() => { setCategoryTab("shap"); setActiveTab("shap_importance"); }} style={{ padding: "6px 12px", borderRadius: 18, border: "none", cursor: "pointer", background: categoryTab === "shap" ? C.navy : C.light, color: categoryTab === "shap" ? "white" : C.gray, fontWeight: categoryTab === "shap" ? 700 : 500, fontSize: 12, display: analysisMode === "clustering" ? "none" : "block" }}>🔍 SHAP Analysis</button>}
                      <button onClick={() => { setCategoryTab("modeling"); setActiveTab(analysisMode === "clustering" ? "cluster_metrics" : "metrics"); }} style={{ padding: "6px 12px", borderRadius: 18, border: "none", cursor: "pointer", background: categoryTab === "modeling" ? C.navy : C.light, color: categoryTab === "modeling" ? "white" : C.gray, fontWeight: categoryTab === "modeling" ? 700 : 500, fontSize: 12 }}>⚙️ Modeling</button>
                    </>
                  )}
                </div>
              </div>

              <div style={{ display: "flex", gap: 6, marginTop: 8, flexWrap: "wrap" }}>
                <div style={{ fontSize: 11, color: C.gray, marginRight: 8, alignSelf: "center" }}>Details:</div>
                {tabs.map((t) => (
                  <button key={t.id} onClick={() => setActiveTab(t.id)} style={{ padding: "6px 12px", borderRadius: 18, border: "none", cursor: "pointer", background: activeTab === t.id ? C.blue : C.light, color: activeTab === t.id ? "white" : C.gray, fontWeight: activeTab === t.id ? 700 : 500, fontSize: 12 }}>{t.label}</button>
                ))}
              </div>
            </div>

            <InsightNarrative categoryTab={categoryTab} activeTab={activeTab} targetCol={targetCol} selectedResult={selectedResult} selectedCluster={selectedCluster} />

            <div style={cardStyle}>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                <button onClick={async () => {
                  const ok = await copyTextToClipboard(reportParagraph);
                  if (!ok) alert("Unable to copy paragraph to clipboard.");
                }} style={{ padding: "6px 10px", border: "none", borderRadius: 5, background: "#DDEAF6", color: C.ink, cursor: "pointer", fontSize: 11 }}>
                  Copy Report Paragraph
                </button>
                <button onClick={() => downloadTextFile(reportParagraph, `${analysisMode}_${activeTab}_narrative.txt`)} style={{ padding: "6px 10px", border: "none", borderRadius: 5, background: "#C8DCEF", color: C.ink, cursor: "pointer", fontSize: 11 }}>
                  Download Report TXT
                </button>
                <button onClick={() => downloadSvgFromContainer(visualRef.current, `${analysisMode}_${activeTab}_publication`)} style={{ padding: "6px 10px", border: "none", borderRadius: 5, background: C.light, color: C.ink, cursor: "pointer", fontSize: 11 }}>
                  Download SVG
                </button>
                <button onClick={() => downloadPngFromContainer(visualRef.current, `${analysisMode}_${activeTab}_publication`, 4)} style={{ padding: "6px 10px", border: "none", borderRadius: 5, background: C.navy, color: "white", cursor: "pointer", fontSize: 11 }}>
                  Download HiFi PNG
                </button>
                <button onClick={() => copyPngFromContainer(visualRef.current, 4)} style={{ padding: "6px 10px", border: "none", borderRadius: 5, background: C.mid, color: "white", cursor: "pointer", fontSize: 11 }}>
                  Clip to Clipboard
                </button>
              </div>
              <div ref={visualRef}>
                {/* DATA EXPLORATION & DISCOVERY TAB */}
                {categoryTab === "exploration" && (
                  <>
                    {activeTab === "profile" && profile && <DataProfileTable profile={profile} />}
                    {activeTab === "corr" && profile && <CorrelationHeatmap profile={profile} />}
                  </>
                )}

                {/* SHAP ANALYSIS TAB */}
                {categoryTab === "shap" && analysisMode === "regression" && (
                  <>
                    {profile?.olsAnalysis ? (
                      <>
                        {activeTab === "shap_importance" && <OlsFeatureImportance features={featureCols} meanAbs={profile.olsAnalysis.meanAbs} coefs={profile.olsAnalysis.coefs} orderImp={profile.olsAnalysis.orderImp} target={targetCol} />}
                        {activeTab === "shap_signed" && <OlsSignedImpactBars features={featureCols} meanAbs={profile.olsAnalysis.meanAbs} meanSigned={shapMeanSigned} orderImp={profile.olsAnalysis.orderImp} target={targetCol} />}
                        {activeTab === "shap_coverage" && <OlsShapCumulativeCoverage features={featureCols} meanAbs={profile.olsAnalysis.meanAbs} orderImp={profile.olsAnalysis.orderImp} />}
                        {activeTab === "shap_beeswarm" && <OlsShapBeeswarm features={featureCols} shapVals={profile.olsAnalysis.shapVals} Xraw={rawData.map((r) => featureCols.map((f) => r[f]))} orderImp={profile.olsAnalysis.orderImp} target={targetCol} />}
                        {activeTab === "shap_heatmap" && <ShapHeatmap features={featureCols} shapVals={profile.olsAnalysis.shapVals} orderImp={profile.olsAnalysis.orderImp} />}
                        {activeTab === "shap_dependence" && <OlsShapDependence features={featureCols} shapVals={profile.olsAnalysis.shapVals} Xraw={rawData.map((r) => featureCols.map((f) => r[f]))} orderImp={profile.olsAnalysis.orderImp} target={targetCol} />}
                        {activeTab === "shap_distribution" && <OlsShapImpactDistribution shapVals={profile.olsAnalysis.shapVals} />}
                        {activeTab === "shap_sign_balance" && <OlsShapSignBalance features={featureCols} shapVals={profile.olsAnalysis.shapVals} orderImp={profile.olsAnalysis.orderImp} />}
                        {activeTab === "shap_cohort" && <OlsShapCohortContrast features={featureCols} shapVals={profile.olsAnalysis.shapVals} orderImp={profile.olsAnalysis.orderImp} y={rawData.map((r) => r[targetCol])} />}
                        {activeTab === "shap_table" && <OlsShapTopTable features={featureCols} meanAbs={profile.olsAnalysis.meanAbs} meanSigned={shapMeanSigned} orderImp={profile.olsAnalysis.orderImp} />}
                        {(activeTab === "shap_waterfall" || activeTab === "shap_force") && (
                          <div>
                            <div style={{ marginBottom: 10, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                              <div style={{ fontSize: 12, color: C.gray }}>Observation row</div>
                              <input
                                type="range"
                                min={0}
                                max={shapMaxRow}
                                value={Math.min(shapRowIndex, shapMaxRow)}
                                onChange={(e) => setShapRowIndex(Number(e.target.value))}
                                style={{ width: 220 }}
                              />
                              <input
                                value={Math.min(shapRowIndex, shapMaxRow) + 1}
                                onChange={(e) => {
                                  const next = clamp((Number(e.target.value) || 1) - 1, 0, shapMaxRow);
                                  setShapRowIndex(next);
                                }}
                                style={{ width: 72, padding: "4px 6px", borderRadius: 4, border: `1px solid ${C.border}`, fontFamily: "monospace" }}
                              />
                              <div style={{ fontSize: 11, color: C.gray }}>/ {shapMaxRow + 1}</div>
                            </div>
                            {activeTab === "shap_waterfall" && <OlsShapWaterfall features={featureCols} shapVals={profile.olsAnalysis.shapVals} orderImp={profile.olsAnalysis.orderImp} rowIndex={Math.min(shapRowIndex, shapMaxRow)} target={targetCol} />}
                            {activeTab === "shap_force" && <OlsShapForcePlot features={featureCols} shapVals={profile.olsAnalysis.shapVals} orderImp={profile.olsAnalysis.orderImp} rowIndex={Math.min(shapRowIndex, shapMaxRow)} target={targetCol} />}
                          </div>
                        )}
                      </>
                    ) : (
                      <div style={{ padding: 20, textAlign: "center", color: C.gray }}>Analytical SHAP not available. Load data with numeric target to compute.</div>
                    )}
                  </>
                )}

                {/* MODELING TAB */}
                {categoryTab === "modeling" && analysisMode === "regression" && (
                  <>
                    {activeTab === "metrics" && results.length > 0 && <MetricsTable rows={results} />}
                    {activeTab === "pred" && selectedResult && <ScatterPlot y={selectedResult.actual} pred={selectedResult.predOOF} title={`${cleanModelLabel(selectedResult.name)} - k-fold OOF actual vs predicted`} />}
                    {activeTab === "residual" && selectedResult && <ResidualBars y={selectedResult.actual} pred={selectedResult.predOOF} />}
                  </>
                )}

                {/* CLUSTERING MODELS IN MODELING TAB */}
                {categoryTab === "modeling" && analysisMode === "clustering" && (
                  <>
                    {activeTab === "cluster_metrics" && clusterResults.length > 0 && <ClusterMetricsTable rows={clusterResults} />}
                    {activeTab === "cluster_scatter" && selectedCluster && <ClusterScatter projection={selectedCluster.projection} labels={selectedCluster.labels} title={`${selectedCluster.name} - PCA cluster map`} />}
                    {activeTab === "cluster_sizes" && selectedCluster && <ClusterSizeBars labels={selectedCluster.labels} />}
                    {activeTab === "cluster_importance" && selectedCluster && <ImportanceBar features={selectedCluster.featureImportance.map((d) => d.feature)} meanAbs={selectedCluster.featureImportance.map((d) => d.score)} meanSigned={selectedCluster.featureImportance.map((d) => d.score)} title="Cluster Feature Separation (eta^2)" />}
                  </>
                )}
              </div>
            </div>

            <div style={{ fontSize: 11, color: C.gray, textAlign: "right", marginTop: 8 }}>
              {analysisMode === "regression" ? `Hybrid optimizer: ${hybridMeta.title}` : "Clustering models include KMeans and Fuzzy C-Means"}
            </div>
          </div>
        ) : (
          <div style={{ ...cardStyle, minHeight: 320, display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 10 }}>
            <div style={{ color: C.navy, fontSize: 17, fontWeight: 700 }}>Upload data to start analytics</div>
            <div style={{ maxWidth: 500, color: C.gray, textAlign: "center", fontSize: 13 }}>
              Run regression or clustering workflows, inspect data profiles, and explore extended SHAP and model diagnostics visuals.
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
