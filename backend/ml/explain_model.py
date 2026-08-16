"""Explainability helpers (mirrors src/ml/explain.ts).

- risk_score_from_probability / risk_category_from_score: map the approval
  probability to the interpretable 0-100 score and Low/Medium/High bands.
- local_factors: ablation ("what-if") attribution. For each feature, replace
  its value with the training median/mode and measure the change in predicted
  approval probability. Large |delta| => the feature strongly influenced this
  prediction; sign => direction (positive pulls toward approval).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from backend.ml.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    LoanPreprocessor,
)

FEATURE_LABELS = {
    "applicant_age": "Applicant age",
    "applicant_income": "Applicant income",
    "coapplicant_income": "Co-applicant income",
    "loan_amount": "Loan amount",
    "loan_term": "Loan term",
    "credit_history": "Credit history",
    "employment_status": "Employment status",
    "education": "Education",
    "marital_status": "Marital status",
    "dependents": "Dependents",
    "property_area": "Property area",
    "self_employed": "Self-employed",
    "existing_loans": "Existing loans",
    "monthly_expenses": "Monthly expenses",
    "savings": "Savings",
    "debt_to_income_ratio": "Debt-to-income ratio",
}


def risk_score_from_probability(probability_approved: float) -> int:
    return int(round((1.0 - probability_approved) * 100))


def risk_category_from_score(score: float) -> str:
    if score <= 30:
        return "Low Risk"
    if score <= 60:
        return "Medium Risk"
    return "High Risk"


def local_factors(row: Dict, predict_proba, preprocessor: LoanPreprocessor) -> List[Dict]:
    """Ablation attribution for a single applicant row (dict of raw values)."""
    base_df = pd.DataFrame([row])
    base = float(predict_proba(base_df)[0])
    factors = []

    for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
        modified = base_df.copy()
        neutral = preprocessor.medians.get(col, preprocessor.modes.get(col))
        if neutral is None:
            continue
        modified[col] = neutral
        delta = base - float(predict_proba(modified)[0])
        factors.append({
            "feature": col,
            "label": FEATURE_LABELS.get(col, col.replace("_", " ")),
            "delta": round(float(delta), 4),
            "direction": "positive" if delta > 0 else "negative",
            "description": (
                f"Value: {row.get(col)} — "
                f"{'pulls toward approval' if delta > 0 else 'pulls toward rejection'}"
                if abs(delta) >= 0.005
                else "minimal effect on this prediction"
            ),
        })

    return sorted(factors, key=lambda f: -abs(f["delta"]))[:5]


def financial_indicators(row: Dict) -> Dict:
    """Real, interpretable financial ratios derived from the raw inputs."""
    num = lambda k, fallback=0.0: (
        float(row[k]) if isinstance(row.get(k), (int, float)) and np.isfinite(row[k]) else fallback
    )
    income = max(num("applicant_income") + num("coapplicant_income"), 1.0)
    monthly_expenses = num("monthly_expenses")
    existing_loans = num("existing_loans")
    loan_amount = num("loan_amount")
    term_months = max(num("loan_term"), 1.0)
    savings = num("savings")

    dti = (monthly_expenses + existing_loans * 0.075 * income) / income
    monthly_rate = 0.09 / 12  # illustrative 9% p.a. — educational estimate
    monthly_burden = (
        loan_amount / term_months
        if monthly_rate == 0
        else (loan_amount * monthly_rate) / (1 - (1 + monthly_rate) ** -term_months)
    )

    return {
        "debtToIncomeRatio": round(dti, 4),
        "loanToAnnualIncome": round(loan_amount / income, 4),
        "savingsToLoan": round(savings / loan_amount, 4) if loan_amount > 0 else 0.0,
        "expenseRatio": round(monthly_expenses / income, 4),
        "monthlyLoanBurdenEstimate": round(monthly_burden),
        "loanTermYears": round(term_months / 12, 1),
    }
