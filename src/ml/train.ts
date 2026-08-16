/**
 * Shared training pipeline.
 *
 * Used by BOTH the offline training script (scripts/train.ts) and the live
 * POST /retrain endpoint (Convex action) so the exact same preprocessing,
 * training, evaluation and model-selection logic runs everywhere.
 */

import type { Row } from "./dataset";
import { COLUMNS } from "./dataset";
import { LoanPreprocessor } from "./preprocess";
import { LogisticRegression } from "./logistic";
import { RandomForestClassifier } from "./tree";
import { evaluate, rocAucOf, round4, type Metrics } from "./metrics";
import { mulberry32 } from "./rng";
import { riskCategoryFromScore, riskScoreFromProbability } from "./explain";
import type { FeatureImportanceItem, ModelArtifact, ModelKind } from "./types";

export interface TrainOptions {
  seed?: number;
  testSize?: number;
}

const NUMERIC_SET = new Set([
  "applicant_age",
  "applicant_income",
  "coapplicant_income",
  "loan_amount",
  "loan_term",
  "credit_history",
  "dependents",
  "existing_loans",
  "monthly_expenses",
  "savings",
  "debt_to_income_ratio",
]);

export function stratifiedSplit(rows: Row[], testSize: number, seed: number) {
  const rand = mulberry32(seed);
  const statusIdx = COLUMNS.indexOf("loan_status");
  const byClass = new Map<string, number[]>();
  rows.forEach((row, i) => {
    const cls = String(row[statusIdx]);
    const list = byClass.get(cls) ?? [];
    list.push(i);
    byClass.set(cls, list);
  });
  const train: number[] = [];
  const test: number[] = [];
  for (const [, idx] of byClass) {
    for (let k = idx.length - 1; k > 0; k--) {
      const j = Math.floor(rand() * (k + 1));
      [idx[k], idx[j]] = [idx[j], idx[k]];
    }
    const nTest = Math.round(idx.length * testSize);
    test.push(...idx.slice(0, nTest));
    train.push(...idx.slice(nTest));
  }
  return { train, test };
}

export function aggregateImportance(
  vectorImportance: number[],
  preprocessor: LoanPreprocessor,
): FeatureImportanceItem[] {
  const names = preprocessor.params.featureNames;
  const agg = new Map<string, number>();
  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    const base = name.includes("_") && !NUMERIC_SET.has(name) ? name.slice(0, name.lastIndexOf("_")) : name;
    agg.set(base, (agg.get(base) ?? 0) + vectorImportance[i]);
  }
  const total = [...agg.values()].reduce((a, b) => a + b, 0) || 1;
  const items: FeatureImportanceItem[] = [...agg.entries()].map(([feature, importance]) => ({
    feature,
    label: feature.replace(/_/g, " "),
    importance: round4(importance / total),
  }));
  return items.sort((a, b) => b.importance - a.importance);
}

/**
 * Permutation importance on a held-out set: how much ROC-AUC drops when each
 * feature's values are shuffled (averaged over repeats). The standard,
 * unbiased complement to impurity-based importance; treats binary features
 * such as credit_history fairly.
 */
export function permutationImportanceOf(
  X: number[][],
  y: number[],
  predictProba: (x: number[]) => number,
  repeats = 5,
  seed = 7,
): number[] {
  const rand = mulberry32(seed);
  const baseAuc = rocAucOf(y, X.map((x) => predictProba(x)));
  const d = X[0].length;
  const n = X.length;
  const drops = new Array<number>(d).fill(0);

  for (let rep = 0; rep < repeats; rep++) {
    for (let f = 0; f < d; f++) {
      const col = X.map((row) => row[f]);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [col[i], col[j]] = [col[j], col[i]];
      }
      const shuffled = X.map((row, i) => {
        const copy = row.slice();
        copy[f] = col[i];
        return copy;
      });
      const auc = rocAucOf(y, shuffled.map((x) => predictProba(x)));
      drops[f] += Math.max(0, baseAuc - auc);
    }
  }
  const means = drops.map((v) => v / repeats);
  const total = means.reduce((a, b) => a + b, 0) || 1;
  return means.map((v) => v / total);
}

function describeModel(name: string, m: Metrics, trainAcc: number): string {
  const [tn, fp, fn, tp] = m.confusionMatrix;
  return (
    `${name}: acc=${(m.accuracy * 100).toFixed(2)}% prec=${(m.precision * 100).toFixed(2)}% ` +
    `rec=${(m.recall * 100).toFixed(2)}% f1=${(m.f1 * 100).toFixed(2)}% auc=${(m.rocAuc * 100).toFixed(2)}% ` +
    `train_acc=${(trainAcc * 100).toFixed(2)}% cm=[[${tn},${fp}],[${fn},${tp}]]`
  );
}

/**
 * Run the full pipeline on a dataset (rows in COLUMNS order, including the
 * loan_status target). Returns the persisted artifact plus a text report.
 */
export function trainAndEvaluate(rows: Row[], options: TrainOptions = {}): { artifact: ModelArtifact; report: string } {
  const seed = options.seed ?? 42;
  const testSize = options.testSize ?? 0.2;
  const statusIdx = COLUMNS.indexOf("loan_status");
  const approved = rows.filter((r) => r[statusIdx] === "Approved").length;

  const { train, test } = stratifiedSplit(rows, testSize, seed);
  const toXY = (idx: number[]) => {
    const X = idx.map((i) => rows[i]);
    const y = idx.map((i) => (rows[i][statusIdx] === "Approved" ? 1 : 0));
    return { X, y };
  };
  const { X: XTrainRaw, y: yTrain } = toXY(train);
  const { X: XTestRaw, y: yTest } = toXY(test);

  const preprocessor = LoanPreprocessor.fit(XTrainRaw);
  const XTrain = preprocessor.transformRows(XTrainRaw);
  const XTest = preprocessor.transformRows(XTestRaw);

  const lr = LogisticRegression.train(XTrain, yTrain);
  const lrTrain = evaluate(yTrain, XTrain.map((x) => lr.predict(x)), XTrain.map((x) => lr.predictProba(x)));
  const lrScores = XTest.map((x) => lr.predictProba(x));
  const lrTest = evaluate(yTest, lrScores.map((s) => (s >= 0.5 ? 1 : 0)), lrScores);

  const rf = RandomForestClassifier.train(XTrain, yTrain);
  const rfTrain = evaluate(yTrain, XTrain.map((x) => rf.predict(x)), XTrain.map((x) => rf.predictProba(x)));
  const rfScores = XTest.map((x) => rf.predictProba(x));
  const rfTest = evaluate(yTest, rfScores.map((s) => (s >= 0.5 ? 1 : 0)), rfScores);

  // Selection criterion: ROC-AUC — this system ranks applicants by risk, and
  // ROC-AUC measures exactly that ranking quality. F1 is reported alongside.
  const selected: ModelKind = rfTest.rocAuc >= lrTest.rocAuc ? "random_forest" : "logistic_regression";
  const criterion = "roc_auc";

  let importance: FeatureImportanceItem[];
  let permutationImportance: FeatureImportanceItem[];
  if (selected === "random_forest") {
    importance = aggregateImportance(rf.featureImportance(), preprocessor);
    permutationImportance = aggregateImportance(
      permutationImportanceOf(XTest, yTest, (x) => rf.predictProba(x)),
      preprocessor,
    );
  } else {
    const weights = lr.weights.map(Math.abs);
    importance = aggregateImportance(weights, preprocessor);
    permutationImportance = aggregateImportance(
      permutationImportanceOf(XTest, yTest, (x) => lr.predictProba(x)),
      preprocessor,
    );
  }

  const testProbas = selected === "random_forest" ? rfScores : lrScores;
  const riskDist = { low: 0, medium: 0, high: 0 };
  for (const p of testProbas) {
    const cat = riskCategoryFromScore(riskScoreFromProbability(p));
    if (cat === "Low Risk") riskDist.low++;
    else if (cat === "Medium Risk") riskDist.medium++;
    else riskDist.high++;
  }

  const selectedMetrics = selected === "random_forest" ? rfTest : lrTest;
  const selectedTrain = selected === "random_forest" ? rfTrain : lrTrain;

  const artifact: ModelArtifact = {
    version: 1,
    trainedAt: new Date().toISOString(),
    dataset: {
      name: "Simulated Loan Application Dataset",
      description:
        "Deterministic, education-oriented simulated dataset. Approval labels come from an interpretable underwriting-style scoring function (credit history, income, debt-to-income ratio, loan size, savings, existing debt, employment, education) plus fixed noise — not random labels.",
      n: rows.length,
      seed,
      simulated: true,
      approvalRate: round4(approved / rows.length),
      missingness: ["credit_history (~7%)", "loan_amount (~5%)", "dependents (~3%)", "employment_status (~2%)"],
    },
    split: { train: train.length, test: test.length, testSize, randomState: seed, stratified: true },
    pipeline: preprocessor.params,
    selectedModel: selected,
    selectionCriterion: criterion,
    models: {
      logistic_regression: {
        kind: "logistic_regression",
        metrics: lrTest,
        trainAccuracy: round4(lrTrain.accuracy),
        params: { ...lr.params },
        weights: lr.weights,
        bias: lr.bias,
        featureImportance: [],
      },
      random_forest: {
        kind: "random_forest",
        metrics: rfTest,
        trainAccuracy: round4(rfTrain.accuracy),
        params: { ...rf.params },
        trees: rf.trees,
        featureImportance: [],
      },
    },
    featureImportance: importance,
    permutationImportance: permutationImportance,
    riskScore: {
      lowMax: 30,
      mediumMax: 60,
      highMin: 61,
      interpretation:
        "An interpretable 0–100 machine-learning-derived educational risk score: 0–30 Low Risk, 31–60 Medium Risk, 61–100 High Risk. Not an official banking credit score.",
    },
    riskDistribution: { ...riskDist, total: test.length },
    classes: ["Rejected", "Approved"],
    disclaimer:
      "This tool is an educational decision-support system. Predictions are machine learning estimates, not guarantees, and do not constitute financial, legal, or lending advice. A human review is always required before any real lending decision.",
  };

  const report = [
    `Dataset: ${rows.length} applications, approval rate ${((approved / rows.length) * 100).toFixed(1)}%`,
    `Split: ${train.length} train / ${test.length} test (stratified, seed ${seed})`,
    `Features: ${preprocessor.params.featureNames.length} model features`,
    describeModel("Logistic Regression", lrTest, lrTrain.accuracy),
    describeModel("Random Forest", rfTest, rfTrain.accuracy),
    `Selected: ${selected} (by ${criterion})`,
    `Final: acc=${(selectedMetrics.accuracy * 100).toFixed(2)}% prec=${(selectedMetrics.precision * 100).toFixed(2)}% ` +
      `rec=${(selectedMetrics.recall * 100).toFixed(2)}% f1=${(selectedMetrics.f1 * 100).toFixed(2)}% auc=${(selectedMetrics.rocAuc * 100).toFixed(2)}%`,
  ].join("\n");

  return { artifact, report };
}
