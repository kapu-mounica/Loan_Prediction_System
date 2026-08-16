"""Preprocessing pipeline for the loan risk model.

Mirrors the TypeScript pipeline in ``src/ml/preprocess.ts`` so the Python
backend and the deployed web app use the same methodology:

1. Impute missing values:  numeric -> training median, categorical -> training mode
2. Winsorize outliers:     numeric features clipped to the 1st/99th percentiles
3. Encode categoricals:    one-hot encoding with the categories seen in training
4. Scale numeric features: standardized with the training mean / standard deviation

The exact same fitted pipeline used during training is persisted and reused at
prediction time (``save`` / ``load``).
"""

from __future__ import annotations

from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature columns (must match the CSV header and the TypeScript engine).
NUMERIC_COLUMNS = [
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
]

CATEGORICAL_COLUMNS = [
    "employment_status",
    "education",
    "marital_status",
    "property_area",
    "self_employed",
]

TARGET = "loan_status"
POSITIVE_CLASS = "Approved"


class LoanPreprocessor:
    """Fit imputation values, winsorization bounds, encoder and scaler on the
    training set, then reuse them to transform training and live rows."""

    def __init__(self, numeric_columns: Optional[List[str]] = None,
                 categorical_columns: Optional[List[str]] = None):
        self.numeric_columns = list(numeric_columns or NUMERIC_COLUMNS)
        self.categorical_columns = list(categorical_columns or CATEGORICAL_COLUMNS)
        self.medians: dict = {}
        self.modes: dict = {}
        self.winsor_bounds: dict = {}
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.scaler = StandardScaler()
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LoanPreprocessor":
        X = X.copy()

        for col in self.numeric_columns:
            series = pd.to_numeric(X[col], errors="coerce")
            self.medians[col] = float(series.median())
            lo, hi = series.quantile(0.01), series.quantile(0.99)
            self.winsor_bounds[col] = (float(lo), float(hi))

        for col in self.categorical_columns:
            series = X[col].dropna()
            self.modes[col] = series.mode().iloc[0] if len(series) else "Unknown"

        # One-hot encode categoricals: impute with the training mode FIRST so
        # the encoder never learns a spurious "nan" category, then encode the
        # (string) categories present in training.
        cat_df = self._imputed_categoricals(X)
        self.encoder.fit(cat_df)

        # Standardize the winsorized numeric matrix.
        self.scaler.fit(self._winsorized_numeric(X))

        # Final feature-name order: standardized numeric block + one-hot block.
        self._feature_names = (
            list(self.numeric_columns)
            + [f"{col}_{cat}" for col, cats in zip(
                self.categorical_columns,
                self.encoder.categories_,
            ) for cat in cats]
        )
        return self

    # ------------------------------------------------------------------
    def _imputed_categoricals(self, X: pd.DataFrame) -> pd.DataFrame:
        cat_df = X[self.categorical_columns].copy()
        for col in self.categorical_columns:
            mask = cat_df[col].isna()
            cat_df.loc[mask, col] = self.modes.get(col, "Unknown")
        return cat_df.astype(str)

    def _winsorized_numeric(self, X: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(X), len(self.numeric_columns)), dtype=float)
        for j, col in enumerate(self.numeric_columns):
            series = pd.to_numeric(X[col], errors="coerce")
            lo, hi = self.winsor_bounds[col]
            out[:, j] = np.clip(series.fillna(self.medians[col]).to_numpy(), lo, hi)
        return out

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._feature_names:
            raise RuntimeError("LoanPreprocessor.transform called before fit()")

        num = self.scaler.transform(self._winsorized_numeric(X))
        cat = self.encoder.transform(self._imputed_categoricals(X))
        return np.hstack([num, cat])

    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "LoanPreprocessor":
        return joblib.load(path)
