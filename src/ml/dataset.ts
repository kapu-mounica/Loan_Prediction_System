/**
 * Education-oriented loan dataset generator.
 *
 * IMPORTANT: this is a SIMULATED dataset, generated deterministically for
 * educational use. The labels are NOT random — every application's approval
 * probability is computed from a real, interpretable underwriting-style
 * scoring function (credit history, income, debt-to-income ratio, loan size,
 * savings, existing debt, employment, education, age) plus a fixed amount of
 * noise. A model trained on this data therefore learns meaningful, consistent
 * financial patterns rather than memorizing random labels. The simulated
 * source is documented in README.md; a real dataset can be dropped in by
 * matching the same column schema.
 */

import { makeGaussian, mulberry32, pickWeighted } from "./rng";

export const COLUMNS = [
  "applicant_age",
  "applicant_income",
  "coapplicant_income",
  "loan_amount",
  "loan_term",
  "credit_history",
  "employment_status",
  "education",
  "marital_status",
  "dependents",
  "property_area",
  "self_employed",
  "existing_loans",
  "monthly_expenses",
  "savings",
  "debt_to_income_ratio",
  "loan_status",
  "risk_category",
] as const;

export type Column = (typeof COLUMNS)[number];

/** A dataset row in COLUMNS order. `null` represents a missing value. */
export type Row = (number | string | null)[];

/** Numeric (or ordinal numeric) feature columns used by the model. */
export const NUMERIC_COLUMNS = [
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
] as const;

/** Categorical feature columns used by the model. */
export const CATEGORICAL_COLUMNS = [
  "employment_status",
  "education",
  "marital_status",
  "property_area",
  "self_employed",
] as const;

export const OPTIONS: Record<string, readonly string[]> = {
  employment_status: ["Employed", "Unemployed"],
  education: ["Graduate", "Not Graduate"],
  marital_status: ["Married", "Single"],
  property_area: ["Urban", "Semiurban", "Rural"],
  self_employed: ["Yes", "No"],
};

const TERM_OPTIONS = [12, 24, 36, 48, 60, 84, 120, 180, 240, 360];
const TERM_WEIGHTS = [0.06, 0.08, 0.16, 0.12, 0.18, 0.08, 0.1, 0.1, 0.07, 0.05];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const roundTo = (v: number, step: number) => Math.round(v / step) * step;

/** Latent score parameters (tuned so the dataset is realistic and learnable). */
export interface LatentParams {
  /** Multiplier on the deterministic feature contribution. */
  scale: number;
  /** Std dev of the noise added to the latent score. */
  noise: number;
  /** Constant intercept (controls the overall approval rate). */
  intercept: number;
}

export const DEFAULT_LATENT: LatentParams = { scale: 2.0, noise: 0.3, intercept: -1.97 };

/**
 * Generate `n` loan applications with a fixed seed.
 *
 * @param latent optional overrides for the latent scoring parameters (used by
 *        training experiments; defaults to DEFAULT_LATENT).
 * @returns rows in COLUMNS order, including the derived `loan_status` target
 *          and `risk_category` columns.
 */
export function generateLoanDataset(
  n = 1500,
  seed = 42,
  latent: Partial<LatentParams> = {},
): Row[] {
  const { scale, noise, intercept } = { ...DEFAULT_LATENT, ...latent };
  const rand = mulberry32(seed);
  const gauss = makeGaussian(rand);
  const rows: Row[] = [];

  for (let i = 0; i < n; i++) {
    // ---- Applicant profile -------------------------------------------------
    const age = clamp(Math.round(38 + gauss() * 9), 21, 65);
    const employed = rand() < 0.78;
    const selfEmployed = employed && rand() < 0.2;
    const education = rand() < 0.56 ? "Graduate" : "Not Graduate";
    const married = rand() < 0.63;
    const dependents = pickWeighted(rand, [0, 1, 2, 3, 4], [0.27, 0.24, 0.2, 0.16, 0.13]);
    const area = pickWeighted(
      rand,
      ["Urban", "Semiurban", "Rural"] as const,
      [0.4, 0.32, 0.28],
    );
    const existingLoans = pickWeighted(rand, [0, 1, 2, 3, 4], [0.55, 0.25, 0.12, 0.06, 0.02]);

    // ---- Finances ----------------------------------------------------------
    const income = clamp(Math.round(Math.exp(gauss() * 0.45 + Math.log(48000))), 15000, 250000);
    const coIncome =
      married && rand() < 0.62
        ? clamp(Math.round(Math.exp(gauss() * 0.5 + Math.log(30000))), 5000, 150000)
        : 0;
    const combinedIncome = income + coIncome;

    // ---- Credit history (correlated with stable applicant traits) ----------
    let pCredit =
      0.8 +
      (employed ? 0.06 : 0) +
      (age > 40 ? 0.05 : 0) +
      (education === "Graduate" ? 0.03 : 0) -
      (dependents >= 3 ? 0.07 : 0) -
      (existingLoans >= 3 ? 0.09 : 0);
    pCredit = clamp(pCredit, 0.35, 0.97);
    const creditHistory = rand() < pCredit ? 1 : 0;

    // ---- Loan request ------------------------------------------------------
    const loanRatio = 0.35 + Math.pow(rand(), 1.35) * 1.4; // 0.35x – 1.75x combined income
    const loanAmount = clamp(
      roundTo(combinedIncome * loanRatio * (0.85 + gauss() * 0.2), 1000),
      10000,
      600000,
    );
    const term = pickWeighted(rand, TERM_OPTIONS, TERM_WEIGHTS);

    // ---- Expenses, savings, debt-to-income ----------------------------------
    const expenseRatio = 0.3 + rand() * 0.25 + dependents * 0.025;
    const monthlyExpenses = clamp(
      Math.round(income * expenseRatio * (0.9 + gauss() * 0.12)),
      3000,
      Math.round(income * 0.9),
    );
    let savings =
      creditHistory === 1
        ? Math.round(income * (0.6 + rand() * 1.6) * (0.85 + gauss() * 0.25))
        : Math.round(income * (0.15 + rand() * 0.5));
    savings = rand() < 0.06 ? 0 : Math.max(0, Math.round(savings));
    const dti = clamp(
      monthlyExpenses / income + existingLoans * 0.075 + Math.max(0, gauss() * 0.03),
      0.08,
      0.92,
    );
    const dtiRounded = Math.round(dti * 100) / 100;

    // ---- Latent creditworthiness score --------------------------------------
    // All contributions are monotonic and interpretable; noise is fixed.
    const logIncome = Math.log(combinedIncome + 1);
    const logSavings = Math.log(savings + 1);
    const loanToIncome = loanAmount / Math.max(combinedIncome, 1);
    const expenseRatioF = monthlyExpenses / Math.max(income, 1);

    // Realistic underwriting logic: strong main effects, plus threshold and
    // interaction effects (e.g. a large loan hurts most when credit history is
    // already weak, savings amplify creditworthiness) that a tree ensemble can
    // learn better than a purely linear model.
    const dtiC = Math.min(dti, 1);
    const ltiC = Math.min(loanToIncome, 2.2);
    const featurePart =
      creditHistory * 1.8 +
      (logIncome - 10.9) * 1.05 +
      (logSavings - 9.2) * 0.16 * (creditHistory === 1 ? 1.6 : 0.6) +
      (0.42 - dtiC) * 2.6 -
      (dtiC > 0.5 ? 0.45 : 0) +
      (0.85 - ltiC) * 0.9 -
      (ltiC > 0.9 ? 0.5 : 0) -
      (creditHistory === 0 ? ltiC * 0.5 : 0) +
      (0.4 - expenseRatioF) * 1.2 +
      (1 - existingLoans) * 0.38 +
      (employed ? 0.34 : -0.34) +
      (education === "Graduate" ? 0.2 : -0.12) +
      (married ? 0.06 : -0.06) +
      (area === "Urban" ? 0.18 : area === "Semiurban" ? 0.1 : -0.14) +
      (selfEmployed ? -0.12 : 0.05) +
      (12 - Math.abs(age - 38)) * 0.02;
    const latent = featurePart * scale + gauss() * noise + intercept;

    const pApprove = 1 / (1 + Math.exp(-latent));
    const approved = rand() < pApprove;
    const rejectP = 1 - pApprove;
    // Risk category uses the same 0–30 / 31–60 / 61–100 boundaries as the
    // model's risk score so the dataset and the API stay consistent.
    const riskScore = Math.round(rejectP * 100);
    const riskCategory =
      riskScore <= 30 ? "Low Risk" : riskScore <= 60 ? "Medium Risk" : "High Risk";

    // ---- Assemble row --------------------------------------------------------
    const row: Row = [
      age,
      income,
      coIncome,
      loanAmount,
      term,
      creditHistory,
      employed ? "Employed" : "Unemployed",
      education,
      married ? "Married" : "Single",
      dependents,
      area,
      selfEmployed ? "Yes" : "No",
      existingLoans,
      monthlyExpenses,
      savings,
      dtiRounded,
    ];

    // Inject realistic missing values so the preprocessing pipeline's
    // imputation is exercised exactly like a real-world dataset.
    if (rand() < 0.07) row[5] = null; // credit_history
    if (rand() < 0.05) row[3] = null; // loan_amount
    if (rand() < 0.03) row[9] = null; // dependents
    if (rand() < 0.02) row[6] = null; // employment_status

    row.push(approved ? "Approved" : "Rejected", riskCategory);
    rows.push(row);
  }

  return rows;
}

/** Render rows as CSV text (missing values as empty cells). */
export function rowsToCsv(rows: Row[]): string {
  const esc = (v: number | string | null) => {
    if (v === null) return "";
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const header = COLUMNS.join(",");
  const body = rows.map((r) => r.map(esc).join(",")).join("\n");
  return `${header}\n${body}\n`;
}
