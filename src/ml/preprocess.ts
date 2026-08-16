/**
 * Preprocessing pipeline mirroring scikit-learn's ColumnTransformer approach:
 *
 *   1. Impute missing values (numeric → median, categorical → mode)
 *   2. Winsorize numeric outliers at the [p1, p99] percentiles
 *   3. One-hot encode categorical features
 *   4. Standardize (z-score) numeric features
 *
 * Every parameter (medians, modes, winsorize bounds, means, standard
 * deviations, one-hot levels) is learned on the training split and persisted
 * in the model artifact. The EXACT same pipeline is applied at inference time
 * by the API, so training and production preprocessing can never diverge.
 */

import { CATEGORICAL_COLUMNS, COLUMNS, NUMERIC_COLUMNS, Row } from "./dataset";

export interface PipelineParams {
  numeric: string[];
  categorical: string[];
  medians: Record<string, number>;
  modes: Record<string, string>;
  winsorize: Record<string, [number, number]>;
  means: Record<string, number>;
  stds: Record<string, number>;
  oneHot: Record<string, string[]>;
  /** Final feature vector order (scaled numerics first, then one-hot columns). */
  featureNames: string[];
}

const colIndex = (name: string) => COLUMNS.indexOf(name as (typeof COLUMNS)[number]);

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function mode(values: string[]): string {
  const counts = new Map<string, number>();
  for (const v of values) counts.set(v, (counts.get(v) ?? 0) + 1);
  let best = values[0];
  let bestCount = -1;
  for (const [v, c] of counts) {
    if (c > bestCount) {
      best = v;
      bestCount = c;
    }
  }
  return best;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export class LoanPreprocessor {
  readonly params: PipelineParams;

  private constructor(params: PipelineParams) {
    this.params = params;
  }

  /** Learn all pipeline parameters from training rows (column order = COLUMNS). */
  static fit(rows: Row[]): LoanPreprocessor {
    const numeric = [...NUMERIC_COLUMNS];
    const categorical = [...CATEGORICAL_COLUMNS];

    const medians: Record<string, number> = {};
    const modes: Record<string, string> = {};
    const winsorize: Record<string, [number, number]> = {};
    const means: Record<string, number> = {};
    const stds: Record<string, number> = {};
    const oneHot: Record<string, string[]> = {};

    for (const name of numeric) {
      const idx = colIndex(name);
      const values = rows
        .map((r) => r[idx])
        .filter((v): v is number => typeof v === "number" && !Number.isNaN(v));
      const med = median(values);
      medians[name] = med;

      const sorted = [...values].sort((a, b) => a - b);
      winsorize[name] = [percentile(sorted, 0.01), percentile(sorted, 0.99)];

      // Impute + winsorize, then compute standardization statistics.
      const cleaned = rows.map((r) => {
        const raw = typeof r[idx] === "number" && r[idx] !== null ? (r[idx] as number) : med;
        const [lo, hi] = winsorize[name];
        return Math.min(hi, Math.max(lo, raw));
      });
      const mean = cleaned.reduce((a, b) => a + b, 0) / cleaned.length;
      const variance =
        cleaned.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / cleaned.length;
      means[name] = mean;
      stds[name] = Math.sqrt(variance) || 1;
    }

    for (const name of categorical) {
      const idx = colIndex(name);
      const values = rows
        .map((r) => r[idx])
        .filter((v): v is string => typeof v === "string" && v.length > 0);
      modes[name] = mode(values);
      const levels = [...new Set(values)].sort();
      oneHot[name] = levels;
    }

    const featureNames: string[] = [];
    for (const name of numeric) featureNames.push(name);
    for (const name of categorical) {
      for (const level of oneHot[name]) featureNames.push(`${name}_${level}`);
    }

    return new LoanPreprocessor({
      numeric,
      categorical,
      medians,
      modes,
      winsorize,
      means,
      stds,
      oneHot,
      featureNames,
    });
  }

  static fromParams(params: PipelineParams): LoanPreprocessor {
    return new LoanPreprocessor(params);
  }

  /** Transform one raw row (COLUMNS order, missing = null) into a feature vector. */
  transformRow(row: Row): number[] {
    const p = this.params;
    const out: number[] = [];

    for (const name of p.numeric) {
      const idx = colIndex(name);
      const raw = typeof row[idx] === "number" && row[idx] !== null ? (row[idx] as number) : p.medians[name];
      const [lo, hi] = p.winsorize[name];
      const clipped = Math.min(hi, Math.max(lo, raw));
      out.push((clipped - p.means[name]) / p.stds[name]);
    }

    for (const name of p.categorical) {
      const idx = colIndex(name);
      const raw = typeof row[idx] === "string" && row[idx] !== null ? (row[idx] as string) : p.modes[name];
      const levels = p.oneHot[name];
      for (const level of levels) {
        out.push(raw === level ? 1 : 0);
      }
    }

    return out;
  }

  /** Transform many rows (used by training). */
  transformRows(rows: Row[]): number[][] {
    return rows.map((r) => this.transformRow(r));
  }

  /** Rebuild a raw row for a specific original feature value (used by local explanations). */
  rowWithValue(base: Row, columnName: string, value: number | string | null): Row {
    const copy = [...base];
    copy[colIndex(columnName)] = value;
    return copy;
  }

  /**
   * Build a raw row (COLUMNS order) from a flat object of model inputs.
   * Missing keys become null so imputation applies.
   */
  static rowFromInput(input: Record<string, number | string | null>): Row {
    return COLUMNS.map((c) => {
      if (c === "loan_status" || c === "risk_category") return null;
      const v = input[c];
      return v === undefined ? null : v;
    });
  }
}
