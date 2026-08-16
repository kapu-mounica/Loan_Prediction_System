/**
 * Shared types for the loan risk ML engine and API.
 */

import type { PipelineParams } from "./preprocess";
import type { LogisticRegression } from "./logistic";
import type { RandomForestClassifier } from "./tree";

/** Applicant input accepted by the prediction API. All values may be missing
 *  (null) — the preprocessing pipeline imputes them, exactly like training. */
export interface ApplicantInput {
  applicant_age: number | null;
  applicant_income: number | null;
  coapplicant_income: number | null;
  loan_amount: number | null;
  loan_term: number | null;
  credit_history: number | null;
  employment_status: string | null;
  education: string | null;
  marital_status: string | null;
  dependents: number | null;
  property_area: string | null;
  self_employed: string | null;
  existing_loans: number | null;
  monthly_expenses: number | null;
  savings: number | null;
  debt_to_income_ratio: number | null;
}

export type ModelKind = "random_forest" | "logistic_regression";

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number;
  confusionMatrix: [number, number, number, number];
  n: number;
  trainAccuracy?: number;
}

export interface FeatureImportanceItem {
  feature: string;
  label: string;
  importance: number;
}

export interface TrainedModel {
  kind: ModelKind;
  metrics: ModelMetrics;
  trainAccuracy?: number;
  params: Record<string, number | string>;
  weights?: number[];
  bias?: number;
  trees?: unknown[];
  featureImportance: FeatureImportanceItem[];
}

export interface ModelArtifact {
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
  pipeline: PipelineParams;
  selectedModel: ModelKind;
  selectionCriterion: string;
  models: Record<ModelKind, TrainedModel>;
  /** Impurity-based feature importance of the selected model. */
  featureImportance: FeatureImportanceItem[];
  /** Permutation-based feature importance computed on the held-out test set. */
  permutationImportance: FeatureImportanceItem[];
  riskScore: {
    lowMax: number;
    mediumMax: number;
    highMin: number;
    interpretation: string;
  };
  riskDistribution: { low: number; medium: number; high: number; total: number };
  classes: ["Rejected", "Approved"];
  disclaimer: string;
}

export interface Factor {
  feature: string;
  label: string;
  delta: number;
  direction: "positive" | "negative";
  description: string;
}

export interface FinancialIndicators {
  debtToIncomeRatio: number;
  loanToAnnualIncome: number;
  savingsToLoan: number;
  expenseRatio: number;
  monthlyLoanBurdenEstimate: number;
  loanTermYears: number;
}

export interface PredictionResult {
  prediction: "Approved" | "Rejected";
  probabilityApproved: number;
  /** 0–100 interpretable, ML-derived educational risk score. */
  riskScore: number;
  riskCategory: "Low Risk" | "Medium Risk" | "High Risk";
  factors: Factor[];
  financialIndicators: FinancialIndicators;
  model: {
    algorithm: string;
    version: number;
    trainedAt: string;
    metrics: ModelMetrics;
  };
  disclaimer: string;
  /** Raw model output for advanced mode. */
  raw: {
    selectedModel: ModelKind;
    riskDistribution: { low: number; medium: number; high: number; total: number };
  };
}

export type LoadedModel =
  | { kind: "random_forest"; model: RandomForestClassifier }
  | { kind: "logistic_regression"; model: LogisticRegression };
