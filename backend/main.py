"""Loan Risk Assessment & Approval Prediction — FastAPI service.

Self-hosted reference implementation of the API exposed by the deployed web
app (see src/convex/loan/api.ts). The endpoint contract is identical:

    GET  /            -> health
    GET  /health      -> health
    GET  /model-info  -> evaluation summary (metrics, comparison, importance)
    GET  /features    -> feature schema for the forms
    POST /predict     -> prediction + risk score + explanation
    POST /risk-score  -> compact risk-only response
    POST /retrain     -> re-run training and promote the new winner

Run locally from the project root:

    pip install -r backend/requirements.txt
    python -m backend.ml.train_model     # train once, writes backend/model/*.pkl
    uvicorn backend.main:app --reload    # serve on http://localhost:8000

The same preprocessing pipeline used during training is loaded from disk and
reused verbatim for every prediction — training and inference cannot diverge.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.ml.explain_model import (
    financial_indicators,
    local_factors,
    risk_category_from_score,
    risk_score_from_probability,
)
from backend.ml.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    POSITIVE_CLASS,
    TARGET,
    LoanPreprocessor,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "backend" / "model"
MODEL_PATH = MODEL_DIR / "loan_risk_model.pkl"

app = FastAPI(
    title="Loan Risk Assessment & Approval Prediction API",
    description="Machine-learning based loan risk assessment (educational).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading (lazy, with a clear error if training was not run yet)
# ---------------------------------------------------------------------------

_model = None  # {"model": estimator, "preprocessor": LoanPreprocessor, "artifact": dict}


def load_model() -> dict:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No trained model found. Run `python -m backend.ml.train_model` "
                    "from the project root to train and persist the artifacts."
                ),
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def artifact() -> dict:
    return load_model()["artifact"]


# ---------------------------------------------------------------------------
# Feature schema (mirrors FEATURES in src/convex/loan/api.ts)
# ---------------------------------------------------------------------------

FEATURES: List[Dict] = [
    {"name": "applicant_age", "label": "Applicant age", "type": "number", "min": 18, "max": 90, "step": 1, "unit": "years", "required": True},
    {"name": "marital_status", "label": "Marital status", "type": "select", "options": ["Married", "Single"], "required": True},
    {"name": "dependents", "label": "Dependents", "type": "number", "min": 0, "max": 10, "step": 1, "unit": "people", "required": True},
    {"name": "education", "label": "Education", "type": "select", "options": ["Graduate", "Not Graduate"], "required": True},
    {"name": "employment_status", "label": "Employment status", "type": "select", "options": ["Employed", "Unemployed"], "required": True},
    {"name": "self_employed", "label": "Self employed", "type": "select", "options": ["Yes", "No"], "required": True},
    {"name": "applicant_income", "label": "Applicant income", "type": "number", "min": 0, "step": 1000, "unit": "₹/yr", "required": True},
    {"name": "coapplicant_income", "label": "Co-applicant income", "type": "number", "min": 0, "step": 1000, "unit": "₹/yr", "required": True},
    {"name": "monthly_expenses", "label": "Monthly expenses", "type": "number", "min": 0, "step": 500, "unit": "₹/mo", "required": True},
    {"name": "savings", "label": "Savings", "type": "number", "min": 0, "step": 1000, "unit": "₹", "required": True},
    {"name": "existing_loans", "label": "Existing loans", "type": "number", "min": 0, "max": 20, "step": 1, "unit": "loans", "required": True},
    {"name": "debt_to_income_ratio", "label": "Debt-to-income ratio", "type": "number", "min": 0, "max": 5, "step": 0.01, "unit": "ratio", "required": False},
    {"name": "loan_amount", "label": "Loan amount", "type": "number", "min": 1, "step": 1000, "unit": "₹", "required": True},
    {"name": "loan_term", "label": "Loan term", "type": "number", "min": 6, "max": 480, "step": 1, "unit": "months", "required": True},
    {"name": "credit_history", "label": "Credit history", "type": "boolean", "options": ["1", "0"], "required": True},
    {"name": "property_area", "label": "Property area", "type": "select", "options": ["Urban", "Semiurban", "Rural"], "required": True},
]

NUM_RULES: Dict[str, Dict] = {
    "applicant_age": {"label": "Applicant age", "min": 18, "max": 90, "integer": True},
    "applicant_income": {"label": "Applicant income", "min": 0, "max": 1e8},
    "coapplicant_income": {"label": "Co-applicant income", "min": 0, "max": 1e8},
    "loan_amount": {"label": "Loan amount", "min": 1, "max": 1e8},
    "loan_term": {"label": "Loan term", "min": 6, "max": 480, "integer": True},
    "credit_history": {"label": "Credit history", "min": 0, "max": 1, "integer": True},
    "dependents": {"label": "Dependents", "min": 0, "max": 10, "integer": True},
    "existing_loans": {"label": "Existing loans", "min": 0, "max": 20, "integer": True},
    "monthly_expenses": {"label": "Monthly expenses", "min": 0, "max": 1e8},
    "savings": {"label": "Savings", "min": 0, "max": 1e8},
    "debt_to_income_ratio": {"label": "Debt-to-income ratio", "min": 0, "max": 5},
}

STR_OPTIONS: Dict[str, List[str]] = {
    "employment_status": ["Employed", "Unemployed"],
    "education": ["Graduate", "Not Graduate"],
    "marital_status": ["Married", "Single"],
    "property_area": ["Urban", "Semiurban", "Rural"],
    "self_employed": ["Yes", "No"],
}


class ApplicantInput(BaseModel):
    applicant_age: Optional[float] = None
    applicant_income: Optional[float] = None
    coapplicant_income: Optional[float] = None
    loan_amount: Optional[float] = None
    loan_term: Optional[float] = None
    credit_history: Optional[float] = None
    employment_status: Optional[str] = None
    education: Optional[str] = None
    marital_status: Optional[str] = None
    dependents: Optional[float] = None
    property_area: Optional[str] = None
    self_employed: Optional[str] = None
    existing_loans: Optional[float] = None
    monthly_expenses: Optional[float] = None
    savings: Optional[float] = None
    debt_to_income_ratio: Optional[float] = None


def validate_input(data: ApplicantInput) -> Dict:
    """Same rules as src/convex/loan/validate.ts — rejects impossible values,
    allows missing values (imputed by the pipeline)."""
    out: Dict = {}
    payload = data.model_dump()

    for key, rule in NUM_RULES.items():
        value = payload.get(key)
        if value is None:
            out[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise HTTPException(422, f"{rule['label']} must be a number.")
        if rule.get("integer") and float(value) != int(value):
            raise HTTPException(422, f"{rule['label']} must be a whole number.")
        if rule.get("min") is not None and value < rule["min"]:
            raise HTTPException(422, f"{rule['label']} must be at least {rule['min']}.")
        if rule.get("max") is not None and value > rule["max"]:
            raise HTTPException(422, f"{rule['label']} must be at most {rule['max']}.")
        out[key] = float(value)

    for key, options in STR_OPTIONS.items():
        value = payload.get(key)
        if value is None:
            out[key] = None
        elif value not in options:
            raise HTTPException(422, f"{key} must be one of: {', '.join(options)}.")
        else:
            out[key] = value

    return out


def run_prediction(input_data: Dict) -> Dict:
    loaded = load_model()
    model = loaded["model"]
    preprocessor: LoanPreprocessor = loaded["preprocessor"]
    art = loaded["artifact"]

    row = pd.DataFrame([input_data])
    features = preprocessor.transform(row)
    probability = float(model.predict_proba(features)[0, 1])
    probability = round(probability, 4)
    prediction = POSITIVE_CLASS if probability >= 0.5 else "Rejected"
    risk_score = risk_score_from_probability(probability)
    risk_category = risk_category_from_score(risk_score)

    return {
        "prediction": prediction,
        "probabilityApproved": probability,
        "riskScore": risk_score,
        "riskCategory": risk_category,
        "factors": local_factors(input_data, model.predict_proba, preprocessor),
        "financialIndicators": financial_indicators(input_data),
        "model": {
            "algorithm": (
                "Random Forest Classifier" if art["selected_model"] == "random_forest"
                else "Logistic Regression"
            ),
            "version": art.get("version", 1),
            "trainedAt": art.get("trained_at", ""),
            "metrics": art["metrics"],
        },
        "disclaimer": art["disclaimer"],
        "raw": {
            "selectedModel": art["selected_model"],
            "riskDistribution": art["risk_distribution"],
        },
    }


def summarize() -> Dict:
    art = artifact()
    comparison = art["comparison"]
    return {
        "version": art.get("version", 1),
        "trainedAt": art.get("trained_at", ""),
        "dataset": art["dataset"],
        "split": art["split"],
        "selectedModel": art["selected_model"],
        "selectionCriterion": art["selection_criterion"],
        "selectedAlgorithm": (
            "Random Forest Classifier" if art["selected_model"] == "random_forest"
            else "Logistic Regression"
        ),
        "metrics": art["metrics"],
        "comparison": [
            {"algorithm": "Logistic Regression", **comparison["logistic_regression"]},
            {"algorithm": "Random Forest Classifier", **comparison["random_forest"]},
        ],
        "featureImportance": art["feature_importance"],
        "permutationImportance": art["permutation_importance"],
        "riskScore": {
            "lowMax": 30,
            "mediumMax": 60,
            "highMin": 61,
            "interpretation": (
                "An interpretable 0-100 machine-learning-derived educational risk score: "
                "0-30 Low Risk, 31-60 Medium Risk, 61-100 High Risk. Not an official "
                "banking credit score."
            ),
        },
        "riskDistribution": art["risk_distribution"],
        "classes": ["Rejected", "Approved"],
        "disclaimer": art["disclaimer"],
        "pipeline": {
            "nFeatures": art["pipeline"]["n_features"],
            "numeric": art["pipeline"]["numeric"],
            "categorical": art["pipeline"]["categorical"],
            "missingValueHandling": (
                "Missing values are imputed (numeric -> median, categorical -> mode), "
                "numeric outliers are winsorized at the 1st/99th percentiles, "
                "categorical features are one-hot encoded, and numeric features are "
                "standardized - identical to training."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
@app.get("/health")
def health() -> Dict:
    art = artifact()
    return {
        "status": "ok",
        "service": "Loan Risk Assessment & Approval Prediction API",
        "model": {
            "algorithm": art["selected_model"],
            "version": art.get("version", 1),
            "trainedAt": art.get("trained_at", ""),
            "source": "trained artifacts (backend/model/)",
        },
        "timestamp": int(pd.Timestamp.now().timestamp() * 1000),
    }


@app.get("/model-info")
def model_info() -> Dict:
    return summarize()


@app.get("/features")
def features() -> List[Dict]:
    return FEATURES


@app.post("/predict")
def predict(input: ApplicantInput) -> Dict:
    return run_prediction(validate_input(input))


@app.post("/risk-score")
def risk_score(input: ApplicantInput) -> Dict:
    result = run_prediction(validate_input(input))
    return {
        "riskScore": result["riskScore"],
        "riskCategory": result["riskCategory"],
        "probabilityApproved": result["probabilityApproved"],
        "prediction": result["prediction"],
    }


@app.post("/retrain")
def retrain() -> Dict:
    """Re-run the full training pipeline (train_model.py) and reload artifacts."""
    global _model
    proc = subprocess.run(
        [sys.executable, "-m", "backend.ml.train_model"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise HTTPException(500, f"Retraining failed:\n{proc.stderr}")
    _model = None  # force reload on the next request
    return {"artifact": summarize(), "report": proc.stdout}
