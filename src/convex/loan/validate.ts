/**
 * Applicant input validation.
 *
 * Rejects impossible values (negative income, non-positive loan amount,
 * zero/invalid loan term, unknown categorical values, ...) with helpful
 * messages. Missing values (null/undefined) are allowed and imputed by the
 * preprocessing pipeline — the same way the training data was handled.
 */

import { ConvexError } from "convex/values";
import { OPTIONS } from "../../ml/dataset";
import type { ApplicantInput } from "../../ml/types";

interface NumRule {
  label: string;
  min?: number;
  max?: number;
  integer?: boolean;
}

interface StrRule {
  label: string;
  options: readonly string[];
}

const NUM_RULES: Record<string, NumRule> = {
  applicant_age: { label: "Applicant age", min: 18, max: 90, integer: true },
  applicant_income: { label: "Applicant income", min: 0, max: 1e8 },
  coapplicant_income: { label: "Co-applicant income", min: 0, max: 1e8 },
  loan_amount: { label: "Loan amount", min: 1, max: 1e8 },
  loan_term: { label: "Loan term", min: 6, max: 480, integer: true },
  credit_history: { label: "Credit history", min: 0, max: 1, integer: true },
  dependents: { label: "Dependents", min: 0, max: 10, integer: true },
  existing_loans: { label: "Existing loans", min: 0, max: 20, integer: true },
  monthly_expenses: { label: "Monthly expenses", min: 0, max: 1e8 },
  savings: { label: "Savings", min: 0, max: 1e8 },
  debt_to_income_ratio: { label: "Debt-to-income ratio", min: 0, max: 5 },
};

const STR_RULES: Record<string, StrRule> = {
  employment_status: { label: "Employment status", options: OPTIONS.employment_status },
  education: { label: "Education", options: OPTIONS.education },
  marital_status: { label: "Marital status", options: OPTIONS.marital_status },
  property_area: { label: "Property area", options: OPTIONS.property_area },
  self_employed: { label: "Self employed", options: OPTIONS.self_employed },
};

function fail(message: string): never {
  throw new ConvexError(message);
}

function readNum(input: Record<string, unknown>, key: string, rule: NumRule): number | null {
  const raw = input[key];
  if (raw === undefined || raw === null || raw === "") return null;
  if (typeof raw !== "number" || !Number.isFinite(raw)) {
    fail(`${rule.label} must be a number.`);
  }
  const value = raw as number;
  if (rule.integer && !Number.isInteger(value)) {
    fail(`${rule.label} must be a whole number.`);
  }
  if (rule.min !== undefined && value < rule.min) {
    if (rule.min === 0) fail(`${rule.label} cannot be negative.`);
    if (rule.min === 1) fail(`${rule.label} must be greater than zero.`);
    fail(`${rule.label} must be at least ${rule.min}.`);
  }
  if (rule.max !== undefined && value > rule.max) {
    fail(`${rule.label} must be at most ${rule.max}.`);
  }
  return value;
}

function readStr(input: Record<string, unknown>, key: string, rule: StrRule): string | null {
  const raw = input[key];
  if (raw === undefined || raw === null || raw === "") return null;
  if (typeof raw !== "string") {
    fail(`${rule.label} must be one of: ${rule.options.join(", ")}.`);
  }
  if (!rule.options.includes(raw)) {
    fail(`${rule.label} must be one of: ${rule.options.join(", ")}.`);
  }
  return raw;
}

export function validateApplicant(input: Record<string, unknown>): ApplicantInput {
  return {
    applicant_age: readNum(input, "applicant_age", NUM_RULES.applicant_age),
    applicant_income: readNum(input, "applicant_income", NUM_RULES.applicant_income),
    coapplicant_income: readNum(input, "coapplicant_income", NUM_RULES.coapplicant_income),
    loan_amount: readNum(input, "loan_amount", NUM_RULES.loan_amount),
    loan_term: readNum(input, "loan_term", NUM_RULES.loan_term),
    credit_history: readNum(input, "credit_history", NUM_RULES.credit_history),
    employment_status: readStr(input, "employment_status", STR_RULES.employment_status),
    education: readStr(input, "education", STR_RULES.education),
    marital_status: readStr(input, "marital_status", STR_RULES.marital_status),
    dependents: readNum(input, "dependents", NUM_RULES.dependents),
    property_area: readStr(input, "property_area", STR_RULES.property_area),
    self_employed: readStr(input, "self_employed", STR_RULES.self_employed),
    existing_loans: readNum(input, "existing_loans", NUM_RULES.existing_loans),
    monthly_expenses: readNum(input, "monthly_expenses", NUM_RULES.monthly_expenses),
    savings: readNum(input, "savings", NUM_RULES.savings),
    debt_to_income_ratio: readNum(input, "debt_to_income_ratio", NUM_RULES.debt_to_income_ratio),
  };
}
