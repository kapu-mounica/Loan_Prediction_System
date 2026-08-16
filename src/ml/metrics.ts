/**
 * Binary classification metrics (positive class = 1 = "Approved").
 */

export type ConfusionMatrix = [tn: number, fp: number, fn: number, tp: number];

export function confusionMatrix(y: number[], yHat: number[]): ConfusionMatrix {
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let tp = 0;
  for (let i = 0; i < y.length; i++) {
    if (y[i] === 1) {
      if (yHat[i] === 1) tp++;
      else fn++;
    } else {
      if (yHat[i] === 1) fp++;
      else tn++;
    }
  }
  return [tn, fp, fn, tp];
}

export function accuracyOf(y: number[], yHat: number[]): number {
  let correct = 0;
  for (let i = 0; i < y.length; i++) if (y[i] === yHat[i]) correct++;
  return correct / Math.max(y.length, 1);
}

export function precisionOf(cm: ConfusionMatrix): number {
  const [, fp, , tp] = cm;
  return tp + fp === 0 ? 0 : tp / (tp + fp);
}

export function recallOf(cm: ConfusionMatrix): number {
  const [, , fn, tp] = cm;
  return tp + fn === 0 ? 0 : tp / (tp + fn);
}

export function f1Of(cm: ConfusionMatrix): number {
  const p = precisionOf(cm);
  const r = recallOf(cm);
  return p + r === 0 ? 0 : (2 * p * r) / (p + r);
}

/**
 * Area under the ROC curve via the trapezoidal rule.
 * y: true labels (0/1), scores: predicted probability of the positive class.
 */
export function rocAucOf(y: number[], scores: number[]): number {
  const pairs = y.map((label, i) => ({ label, score: scores[i] }));
  pairs.sort((a, b) => b.score - a.score);

  let tp = 0;
  let fp = 0;
  const nPos = y.reduce((a, b) => a + b, 0);
  const nNeg = y.length - nPos;
  if (nPos === 0 || nNeg === 0) return 0.5;

  const points: [number, number][] = [[0, 0]];
  for (const p of pairs) {
    if (p.label === 1) tp++;
    else fp++;
    points.push([fp / nNeg, tp / nPos]);
  }

  let area = 0;
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1];
    const [x1, y1] = points[i];
    area += ((y0 + y1) / 2) * (x1 - x0);
  }
  return area;
}

export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number;
  confusionMatrix: ConfusionMatrix;
  n: number;
}

export function evaluate(y: number[], yHat: number[], scores: number[]): Metrics {
  const cm = confusionMatrix(y, yHat);
  return {
    accuracy: accuracyOf(y, yHat),
    precision: precisionOf(cm),
    recall: recallOf(cm),
    f1: f1Of(cm),
    rocAuc: rocAucOf(y, scores),
    confusionMatrix: cm,
    n: y.length,
  };
}

export function round4(v: number): number {
  return Math.round(v * 10000) / 10000;
}
