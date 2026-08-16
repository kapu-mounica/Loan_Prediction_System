/**
 * Loan Risk Assessment API (Convex backend).
 *
 * Mapping to the FastAPI endpoints the product spec defines:
 *   GET  /            → health
 *   GET  /health      → health
 *   GET  /model-info  → modelInfo
 *   GET  /features    → features
 *   POST /predict     → predict
 *   POST /risk-score  → riskScore
 *   POST /retrain     → retrain
 *
 * The identical API is provided as a Python FastAPI service in backend/
 * (see README.md) for local execution with scikit-learn.
 */

import { action, query } from "../_generated/server";
import { v } from "convex/values";
import { internal } from "../_generated/api";
import type { QueryCtx } from "../_generated/server";
import { bundledArtifact, deserializeModel, predictFromArtifact } from "./engine";
import { validateApplicant } from "./validate";
import { trainAndEvaluate } from "../../ml/train";
import { LOAN_DATASET } from "../loanData";
import { OPTIONS } from "../../ml/dataset";
import type { ModelArtifact } from "../../ml/types";

// ---------------------------------------------------------------------------
// Feature schema (used by the API and documented in the frontend)
// ---------------------------------------------------------------------------

const FEATURES = [
  { name: "applicant_age", label: "Applicant age", type: "number", min: 18, max: 90, step: 1, unit: "years", required: true },
  { name: "marital_status", label: "Marital status", type: "select", options: [...OPTIONS.marital_status], required: true },
  { name: "dependents", label: "Dependents", type: "number", min: 0, max: 10, step: 1, unit: "people", required: true },
  { name: "education", label: "Education", type: "select", options: [...OPTIONS.education], required: true },
  { name: "employment_status", label: "Employment status", type: "select", options: [...OPTIONS.employment_status], required: true },
  { name: "self_employed", label: "Self employed", type: "select", options: [...OPTIONS.self_employed], required: true },
  { name: "applicant_income", label: "Applicant income", type: "number", min: 0, step: 1000, unit: "₹/yr", required: true },
  { name: "coapplicant_income", label: "Co-applicant income", type: "number", min: 0, step: 1000, unit: "₹/yr", required: true },
  { name: "monthly_expenses", label: "Monthly expenses", type: "number", min: 0, step: 500, unit: "₹/mo", required: true },
  { name: "savings", label: "Savings", type: "number", min: 0, step: 1000, unit: "₹", required: true },
  { name: "existing_loans", label: "Existing loans", type: "number", min: 0, max: 20, step: 1, unit: "loans", required: true },
  { name: "debt_to_income_ratio", label: "Debt-to-income ratio", type: "number", min: 0, max: 5, step: 0.01, unit: "ratio", required: false },
  { name: "loan_amount", label: "Loan amount", type: "number", min: 1, step: 1000, unit: "₹", required: true },
  { name: "loan_term", label: "Loan term", type: "number", min: 6, max: 480, step: 1, unit: "months", required: true },
  { name: "credit_history", label: "Credit history", type: "boolean", options: ["1", "0"], required: true },
  { name: "property_area", label: "Property area", type: "select", options: [...OPTIONS.property_area], required: true },
];

const INPUT_VALIDATOR = v.object({
  applicant_age: v.optional(v.union(v.number(), v.null())),
  applicant_income: v.optional(v.union(v.number(), v.null())),
  coapplicant_income: v.optional(v.union(v.number(), v.null())),
  loan_amount: v.optional(v.union(v.number(), v.null())),
  loan_term: v.optional(v.union(v.number(), v.null())),
  credit_history: v.optional(v.union(v.number(), v.null())),
  employment_status: v.optional(v.union(v.string(), v.null())),
  education: v.optional(v.union(v.string(), v.null())),
  marital_status: v.optional(v.union(v.string(), v.null())),
  dependents: v.optional(v.union(v.number(), v.null())),
  property_area: v.optional(v.union(v.string(), v.null())),
  self_employed: v.optional(v.union(v.string(), v.null())),
  existing_loans: v.optional(v.union(v.number(), v.null())),
  monthly_expenses: v.optional(v.union(v.number(), v.null())),
  savings: v.optional(v.union(v.number(), v.null())),
  debt_to_income_ratio: v.optional(v.union(v.number(), v.null())),
});

async function latestModelFromDb(ctx: QueryCtx): Promise<ModelArtifact | null> {
  const doc = await ctx.db
    .query("models")
    .withIndex("by_name", (q) => q.eq("name", "production"))
    .order("desc")
    .first();
  return doc ? (doc.payload as ModelArtifact) : null;
}

/** Summary shape returned to the frontend (no trees/weights). */
function summarize(artifact: ModelArtifact) {
  const metricsOf = (kind: "logistic_regression" | "random_forest") => {
    const m = artifact.models[kind];
    return {
      algorithm: kind === "random_forest" ? "Random Forest Classifier" : "Logistic Regression",
      accuracy: m.metrics.accuracy,
      precision: m.metrics.precision,
      recall: m.metrics.recall,
      f1: m.metrics.f1,
      rocAuc: m.metrics.rocAuc,
      confusionMatrix: m.metrics.confusionMatrix,
      trainAccuracy: m.trainAccuracy ?? null,
      n: m.metrics.n,
    };
  };
  return {
    version: artifact.version,
    trainedAt: artifact.trainedAt,
    dataset: artifact.dataset,
    split: artifact.split,
    selectedModel: artifact.selectedModel,
    selectionCriterion: artifact.selectionCriterion,
    selectedAlgorithm:
      artifact.selectedModel === "random_forest" ? "Random Forest Classifier" : "Logistic Regression",
    metrics: metricsOf(artifact.selectedModel),
    comparison: [
      { ...metricsOf("logistic_regression"), kind: "logistic_regression" },
      { ...metricsOf("random_forest"), kind: "random_forest" },
    ],
    featureImportance: artifact.featureImportance,
    permutationImportance: artifact.permutationImportance,
    riskScore: artifact.riskScore,
    riskDistribution: artifact.riskDistribution,
    classes: artifact.classes,
    disclaimer: artifact.disclaimer,
    pipeline: {
      nFeatures: artifact.pipeline.featureNames.length,
      numeric: artifact.pipeline.numeric,
      categorical: artifact.pipeline.categorical,
      featureNames: artifact.pipeline.featureNames,
      missingValueHandling:
        "Missing values are imputed (numeric → median, categorical → mode), numeric outliers are winsorized at the 1st/99th percentiles, categorical features are one-hot encoded, and numeric features are standardized — identical to training.",
    },
  };
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export const health = query({
  args: {},
  handler: async (ctx) => {
    const dbModel = await latestModelFromDb(ctx);
    const artifact = dbModel ?? bundledArtifact();
    return {
      status: "ok",
      service: "Loan Risk Assessment & Approval Prediction API",
      model: {
        algorithm: artifact.selectedModel,
        version: artifact.version,
        trainedAt: artifact.trainedAt,
        source: dbModel ? "retrained (database)" : "bundled artifact",
      },
      timestamp: Date.now(),
    };
  },
});

export const modelInfo = query({
  args: {},
  handler: async (ctx) => {
    const dbModel = await latestModelFromDb(ctx);
    const artifact = dbModel ?? bundledArtifact();
    return summarize(artifact);
  },
});

export const features = query({
  args: {},
  handler: async () => FEATURES,
});

export const predict = action({
  args: { input: INPUT_VALIDATOR },
  handler: async (ctx, args) => {
    const input = validateApplicant(args.input as unknown as Record<string, unknown>);
    const dbModel = (await ctx.runQuery(internal.loan.models.getLatestModel)) as ModelArtifact | null;
    const artifact = dbModel ?? bundledArtifact();
    const loaded = deserializeModel(artifact);
    return predictFromArtifact(input, artifact, loaded);
  },
});

export const riskScore = action({
  args: { input: INPUT_VALIDATOR },
  handler: async (ctx, args) => {
    const input = validateApplicant(args.input as unknown as Record<string, unknown>);
    const dbModel = (await ctx.runQuery(internal.loan.models.getLatestModel)) as ModelArtifact | null;
    const artifact = dbModel ?? bundledArtifact();
    const loaded = deserializeModel(artifact);
    const result = predictFromArtifact(input, artifact, loaded);
    return {
      riskScore: result.riskScore,
      riskCategory: result.riskCategory,
      probabilityApproved: result.probabilityApproved,
      prediction: result.prediction,
    };
  },
});

export const retrain = action({
  args: {},
  handler: async (ctx) => {
    // Runs the identical training pipeline used offline (scripts/train.ts).
    const { artifact, report } = trainAndEvaluate(LOAN_DATASET as (number | string | null)[][]);
    await ctx.runMutation(internal.loan.models.storeModel, { artifact });
    return { artifact: summarize(artifact), report };
  },
});
