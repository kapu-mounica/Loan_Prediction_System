/**
 * Explainability helpers.
 *
 * Global explainability  — Random Forest impurity-based feature importance
 *   (computed from the trained forest, see tree.ts).
 *
 * Local explainability  — ablation-based feature attribution: for each input
 *   feature, the feature's value is replaced by its training median/mode and
 *   the resulting change in predicted approval probability is measured. A
 *   feature with a large |Δp| strongly influenced this prediction. This is a
 *   standard local-attribution technique (a "what-if" analysis); it is an
 *   approximation of SHAP-style attributions and is documented as such.
 *
 * Feature importance describes model behavior — it is not proof of causal
 * financial relationships.
 */

import { CATEGORICAL_COLUMNS, COLUMNS, NUMERIC_COLUMNS } from "./dataset";
import { LoanPreprocessor } from "./preprocess";
import type { Factor, FinancialIndicators } from "./types";

const FEATURE_LABELS: Record<string, string> = {
  applicant_age: "Applicant age",
  applicant_income: "Applicant income",
  coapplicant_income: "Co-applicant income",
  loan_amount: "Loan amount",
  loan_term: "Loan term",
  credit_history: "Credit history",
  employment_status: "Employment status",
  education: "Education",
  marital_status: "Marital status",
  dependents: "Dependents",
  property_area: "Property area",
  self_employed: "Self-employed",
  existing_loans: "Existing loans",
  monthly_expenses: "Monthly expenses",
  savings: "Savings",
  debt_to_income_ratio: "Debt-to-income ratio",
};

export function featureLabel(name: string): string {
  return FEATURE_LABELS[name] ?? name.replace(/_/g, " ");
}

function describeValue(name: string, value: number | string): string {
  switch (name) {
    case "credit_history":
      return value === 1 ? "Positive credit history" : "No/negative credit history";
    case "employment_status":
      return `Employment status: ${value}`;
    case "education":
      return `Education: ${value}`;
    case "marital_status":
      return `Marital status: ${value}`;
    case "property_area":
      return `Property area: ${value}`;
    case "self_employed":
      return `Self-employed: ${value}`;
    case "applicant_age":
      return `Age ${value}`;
    case "applicant_income":
      return `Applicant income ₹${Number(value).toLocaleString("en-IN")}`;
    case "coapplicant_income":
      return `Co-applicant income ₹${Number(value).toLocaleString("en-IN")}`;
    case "loan_amount":
      return `Loan amount ₹${Number(value).toLocaleString("en-IN")}`;
    case "loan_term":
      return `Loan term ${value} months`;
    case "dependents":
      return `${value} dependent${Number(value) === 1 ? "" : "s"}`;
    case "existing_loans":
      return `${value} existing loan${Number(value) === 1 ? "" : "s"}`;
    case "monthly_expenses":
      return `Monthly expenses ₹${Number(value).toLocaleString("en-IN")}`;
    case "savings":
      return `Savings ₹${Number(value).toLocaleString("en-IN")}`;
    case "debt_to_income_ratio":
      return `Debt-to-income ratio ${(Number(value) * 100).toFixed(1)}%`;
    default:
      return `${name}: ${value}`;
  }
}

/**
 * Local ablation attribution. `proba` must return the approval probability for
 * a raw row; `preprocessor` supplies the training medians/modes used to
 * neutralize each feature.
 */
export function localFactors(
  rawRow: (number | string | null)[],
  proba: (row: (number | string | null)[]) => number,
  preprocessor: LoanPreprocessor,
): Factor[] {
  const p = preprocessor.params;
  const base = proba(rawRow);
  const factors: Factor[] = [];

  for (const name of [...NUMERIC_COLUMNS, ...CATEGORICAL_COLUMNS]) {
    const current = rawRow[rawRowColumn(name)];
    const neutral: number | string = p.medians[name] ?? p.modes[name];
    const modified = preprocessor.rowWithValue(rawRow, name, neutral);
    const delta = base - proba(modified);

    factors.push({
      feature: name,
      label: featureLabel(name),
      delta,
      direction: delta > 0 ? "positive" : "negative",
      description: `${describeValue(name, current ?? neutral)} — ${Math.abs(delta) < 0.005 ? "minimal effect on this prediction" : `pulls the prediction ${delta > 0 ? "toward approval" : "toward rejection"}`}`,
    });
  }

  return factors.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
}

function rawRowColumn(name: string): number {
  return COLUMNS.indexOf(name as (typeof COLUMNS)[number]);
}

/** Interpretable financial indicators derived from the raw inputs (real math). */
export function financialIndicators(input: Record<string, number | string | null>): FinancialIndicators {
  const num = (k: string, fallback = 0) => {
    const v = input[k];
    return typeof v === "number" && Number.isFinite(v) ? v : fallback;
  };
  const income = Math.max(num("applicant_income") + num("coapplicant_income"), 1);
  const monthlyExpenses = num("monthly_expenses");
  const existingLoans = num("existing_loans");
  const loanAmount = num("loan_amount");
  const termMonths = Math.max(num("loan_term"), 1);
  const savings = num("savings");

  const dti = (monthlyExpenses + existingLoans * 0.075 * income) / income;
  const loanToIncome = loanAmount / income;
  const savingsToLoan = loanAmount > 0 ? savings / loanAmount : 0;
  const expenseRatio = monthlyExpenses / Math.max(income, 1);

  // Standard amortized monthly payment at an illustrative 9% p.a. — an
  // educational estimate, not a quote.
  const annualRate = 0.09;
  const monthlyRate = annualRate / 12;
  const monthlyBurden =
    monthlyRate === 0
      ? loanAmount / termMonths
      : (loanAmount * monthlyRate) / (1 - Math.pow(1 + monthlyRate, -termMonths));

  return {
    debtToIncomeRatio: Math.round(dti * 10000) / 10000,
    loanToAnnualIncome: Math.round(loanToIncome * 10000) / 10000,
    savingsToLoan: Math.round(savingsToLoan * 10000) / 10000,
    expenseRatio: Math.round(expenseRatio * 10000) / 10000,
    monthlyLoanBurdenEstimate: Math.round(monthlyBurden),
    loanTermYears: Math.round((termMonths / 12) * 10) / 10,
  };
}

/**
 * Map an approval probability to the 0–100 interpretable risk score.
 * This is an ML-derived educational score, not an official credit score.
 */
export function riskScoreFromProbability(probabilityApproved: number): number {
  return Math.round((1 - probabilityApproved) * 100);
}

export function riskCategoryFromScore(score: number): "Low Risk" | "Medium Risk" | "High Risk" {
  if (score <= 30) return "Low Risk";
  if (score <= 60) return "Medium Risk";
  return "High Risk";
}
