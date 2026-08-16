/**
 * ML engine used by the loan API.
 *
 * The exact same preprocessing pipeline and model code used during training
 * (src/ml/*) runs here, so training and production inference can never diverge.
 */

import { MODEL_JSON } from "../loanModel";
import { COLUMNS } from "../../ml/dataset";
import { LoanPreprocessor } from "../../ml/preprocess";
import { LogisticRegression } from "../../ml/logistic";
import { RandomForestClassifier, type TreeNode } from "../../ml/tree";
import {
  financialIndicators,
  localFactors,
  riskCategoryFromScore,
  riskScoreFromProbability,
} from "../../ml/explain";
import type { ApplicantInput, LoadedModel, ModelArtifact, PredictionResult } from "../../ml/types";

let cachedArtifact: ModelArtifact | null = null;

/** The model artifact bundled with the deployment (trained by scripts/train.ts). */
export function bundledArtifact(): ModelArtifact {
  if (!cachedArtifact) {
    cachedArtifact = JSON.parse(MODEL_JSON) as ModelArtifact;
  }
  return cachedArtifact;
}

/** Rebuild a runnable model instance from a serialized artifact. */
export function deserializeModel(artifact: ModelArtifact): LoadedModel {
  if (artifact.selectedModel === "random_forest") {
    const m = artifact.models.random_forest;
    const params = m.params;
    const rf = new RandomForestClassifier(
      m.trees as unknown as TreeNode[][],
      {
        nEstimators: Number(params.nEstimators ?? 80),
        maxDepth: Number(params.maxDepth ?? 12),
        minSamplesSplit: Number(params.minSamplesSplit ?? 6),
        minSamplesLeaf: Number(params.minSamplesLeaf ?? 6),
        maxFeatures: params.maxFeatures === "sqrt" ? "sqrt" : Number(params.maxFeatures ?? 4),
        randomState: Number(params.randomState ?? 42),
      },
      artifact.pipeline.featureNames.length,
    );
    return { kind: "random_forest", model: rf };
  }

  const m = artifact.models.logistic_regression;
  return {
    kind: "logistic_regression",
    model: LogisticRegression.fromSerialized(m.weights ?? [], m.bias ?? 0),
  };
}

/** Run the full prediction pipeline for one applicant. */
export function predictFromArtifact(
  input: ApplicantInput,
  artifact: ModelArtifact,
  loaded: LoadedModel,
): PredictionResult {
  const preprocessor = LoanPreprocessor.fromParams(artifact.pipeline);
  const rawRow = LoanPreprocessor.rowFromInput(input as unknown as Record<string, number | string | null>);
  const features = preprocessor.transformRow(rawRow);

  const probaOf = (row: (number | string | null)[]) => {
    const f = preprocessor.transformRow(row);
    return loaded.kind === "random_forest" ? loaded.model.predictProba(f) : loaded.model.predictProba(f);
  };

  const probabilityApproved = Math.round(probaOf(rawRow) * 10000) / 10000;
  const prediction = probabilityApproved >= 0.5 ? "Approved" : "Rejected";
  const riskScore = riskScoreFromProbability(probabilityApproved);
  const riskCategory = riskCategoryFromScore(riskScore);
  const factors = localFactors(rawRow, probaOf, preprocessor).slice(0, 5);
  const indicators = financialIndicators(input as unknown as Record<string, number | string | null>);
  const selectedMetrics = artifact.models[artifact.selectedModel].metrics;

  return {
    prediction,
    probabilityApproved,
    riskScore,
    riskCategory,
    factors,
    financialIndicators: indicators,
    model: {
      algorithm: artifact.selectedModel === "random_forest" ? "Random Forest Classifier" : "Logistic Regression",
      version: artifact.version,
      trainedAt: artifact.trainedAt,
      metrics: selectedMetrics,
    },
    disclaimer: artifact.disclaimer,
    raw: {
      selectedModel: artifact.selectedModel,
      riskDistribution: artifact.riskDistribution,
    },
  };
}

/** Keep COLUMNS referenced for bundlers that tree-shake unused imports. */
export const inputColumnOrder = COLUMNS;
