/**
 * Frontend client helpers for the Loan Risk Assessment API.
 *
 * All data flows through the Convex backend (src/convex/loan/api.ts), which
 * runs the identical preprocessing + model pipeline used during training.
 * No applicant data is stored anywhere — results are kept only in the
 * browser's sessionStorage and are cleared when the tab closes.
 */
import type { ApplicantInput, PredictionResult } from "@/ml/types";

export type { ApplicantInput, PredictionResult };

// ---------------------------------------------------------------------------
// API response shapes (mirror the Convex `summarize` payload)
// ---------------------------------------------------------------------------

export interface MetricEntry {
  algorithm: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number;
  confusionMatrix: [number, number, number, number];
  trainAccuracy: number | null;
  n: number;
  kind?: "logistic_regression" | "random_forest";
}

export interface FeatureImportanceEntry {
  feature: string;
  label: string;
  importance: number;
}

export interface ModelInfo {
  version: number;
  trainedAt: string;
  dataset: {
    name: string;
    description: string;
    n: number;
    seed: number;
    simulated: boolean;
    approvalRate: number;
    missingness: string[];
  };
  split: {
    train: number;
    test: number;
    testSize: number;
    randomState: number;
    stratified: boolean;
  };
  selectedModel: "random_forest" | "logistic_regression";
  selectionCriterion: string;
  selectedAlgorithm: string;
  metrics: MetricEntry;
  comparison: MetricEntry[];
  featureImportance: FeatureImportanceEntry[];
  permutationImportance: FeatureImportanceEntry[];
  riskScore: {
    lowMax: number;
    mediumMax: number;
    highMin: number;
    interpretation: string;
  };
  riskDistribution: { low: number; medium: number; high: number; total: number };
  classes: string[];
  disclaimer: string;
  pipeline: {
    nFeatures: number;
    numeric: string[];
    categorical: string[];
    featureNames: string[];
    missingValueHandling: string;
  };
}

export interface FeatureDef {
  name: string;
  label: string;
  type: "number" | "select" | "boolean";
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: string[];
  required: boolean;
}

export interface HealthInfo {
  status: string;
  service: string;
  model: {
    algorithm: string;
    version: number;
    trainedAt: string;
    source: string;
  };
  timestamp: number;
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

export function formatCurrency(value: number): string {
  return inr.format(Math.round(value));
}

export function formatCompactCurrency(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
  return String(Math.round(value));
}

export function formatPercent(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value: number, digits = 1): string {
  return value.toFixed(digits);
}

export function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch {
    return iso;
  }
}

export function titleCase(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// ---------------------------------------------------------------------------
// Risk helpers
// ---------------------------------------------------------------------------

export type RiskTone = "low" | "medium" | "high";

export function riskTone(category: string): RiskTone {
  const c = category.toLowerCase();
  if (c.includes("high")) return "high";
  if (c.includes("medium")) return "medium";
  return "low";
}

export const RISK_COLORS: Record<RiskTone, string> = {
  low: "#10b981",
  medium: "#f59e0b",
  high: "#ef4444",
};

/** Tailwind badge classes per risk tone. */
export const RISK_BADGE: Record<RiskTone, string> = {
  low: "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-300",
  medium:
    "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300",
  high: "border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-300",
};

// ---------------------------------------------------------------------------
// Session-scoped assessment state (never persisted server-side)
// ---------------------------------------------------------------------------

const RESULT_KEY = "creditlens:assessment:result";
const INPUT_KEY = "creditlens:assessment:input";

export interface StoredAssessment {
  input: ApplicantInput;
  result: PredictionResult;
  createdAt: number;
}

export function saveAssessment(input: ApplicantInput, result: PredictionResult) {
  const payload: StoredAssessment = { input, result, createdAt: Date.now() };
  try {
    sessionStorage.setItem(RESULT_KEY, JSON.stringify(payload));
    sessionStorage.setItem(INPUT_KEY, JSON.stringify(input));
  } catch {
    /* storage unavailable — non-fatal */
  }
}

export function loadAssessment(): StoredAssessment | null {
  try {
    const raw = sessionStorage.getItem(RESULT_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as StoredAssessment;
  } catch {
    return null;
  }
}

export function clearAssessment() {
  try {
    sessionStorage.removeItem(RESULT_KEY);
    sessionStorage.removeItem(INPUT_KEY);
  } catch {
    /* no-op */
  }
}
