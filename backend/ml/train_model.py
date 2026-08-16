"""Training pipeline for the Python backend (mirrors scripts/train.ts).

Run from the project root:

    python -m backend.ml.train_model

Steps:
1. Load backend/dataset/loan_dataset.csv (the shared, deterministic dataset)
2. Stratified 80/20 train/test split with a fixed random state (42)
3. Fit the preprocessing pipeline on the training set only
4. Train Logistic Regression and a Random Forest on the identical split
5. Evaluate both on the identical held-out test set
   (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
6. Select the better model by ROC-AUC (ranking quality for a risk system)
7. Compute feature importance (impurity-based for RF, |weights| for LR) and
   permutation importance on the test set
8. Persist: loan_risk_model.pkl, scaler.pkl, encoder.pkl,
   feature_columns.pkl, report.json — all under backend/model/
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "backend" / "dataset" / "loan_dataset.csv"
MODEL_DIR = ROOT / "backend" / "model"

SEED = 42
TEST_SIZE = 0.2

RF_PARAMS = {
    "n_estimators": 80,
    "max_depth": 12,
    "min_samples_split": 6,
    "min_samples_leaf": 6,
    "max_features": "sqrt",
    "random_state": SEED,
    "n_jobs": -1,
}

LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": SEED,
}


def aggregate_importance(importances: np.ndarray, feature_names: list) -> list:
    """Sum one-hot columns back to their base feature and normalize (mirrors
    the TypeScript aggregateImportance)."""
    from backend.ml.preprocessing import NUMERIC_COLUMNS

    numeric_set = set(NUMERIC_COLUMNS)
    agg: dict = {}
    for name, imp in zip(feature_names, importances):
        base = name
        if "_" in name and name not in numeric_set:
            base = name.rsplit("_", 1)[0]
        agg[base] = agg.get(base, 0.0) + float(imp)
    total = sum(agg.values()) or 1.0
    return sorted(
        ({"feature": k, "label": k.replace("_", " "), "importance": round(v / total, 4)}
         for k, v in agg.items()),
        key=lambda item: -item["importance"],
    )


def main() -> None:
    print("Loan Risk Assessment — Python training pipeline\n")

    # 1. Dataset ----------------------------------------------------------
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}. Generate it with: bun scripts/train.ts")
        sys.exit(1)
    df = pd.read_csv(DATASET_PATH)
    from backend.ml.preprocessing import POSITIVE_CLASS, TARGET, LoanPreprocessor

    approved = int((df[TARGET] == POSITIVE_CLASS).sum())
    print(f"1. Dataset: {len(df)} applications, approval rate {approved / len(df):.1%}")

    # 2. Split (stratified, fixed seed) -----------------------------------
    X = df.drop(columns=[TARGET, "risk_category"])
    y = (df[TARGET] == POSITIVE_CLASS).astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"2. Split: {len(X_train)} train / {len(X_test)} test (stratified, seed {SEED})")

    # 3. Preprocessing ----------------------------------------------------
    preprocessor = LoanPreprocessor().fit(X_train, y_train)
    Xt_train = preprocessor.transform(X_train)
    Xt_test = preprocessor.transform(X_test)
    feature_names = preprocessor.feature_names()
    print(f"3. Preprocessing fitted: {len(feature_names)} model features")

    # 4-5. Train + evaluate both models -----------------------------------
    from backend.ml.evaluate_model import describe_model, evaluate_model

    lr = LogisticRegression(**LR_PARAMS).fit(Xt_train, y_train)
    lr_proba = lr.predict_proba(Xt_test)[:, 1]
    lr_train_acc = lr.score(Xt_train, y_train)
    lr_metrics = evaluate_model(lr, Xt_test, y_test, lr_proba)

    rf = RandomForestClassifier(**RF_PARAMS).fit(Xt_train, y_train)
    rf_proba = rf.predict_proba(Xt_test)[:, 1]
    rf_train_acc = rf.score(Xt_train, y_train)
    rf_metrics = evaluate_model(rf, Xt_test, y_test, rf_proba)

    print("4-5. Evaluation on held-out test set:")
    print("     " + describe_model("Logistic Regression", lr_metrics, lr_train_acc))
    print("     " + describe_model("Random Forest", rf_metrics, rf_train_acc))

    # 6. Select by ROC-AUC ------------------------------------------------
    selected = "random_forest" if rf_metrics["rocAuc"] >= lr_metrics["rocAuc"] else "logistic_regression"
    model = rf if selected == "random_forest" else lr
    selected_proba = rf_proba if selected == "random_forest" else lr_proba
    selected_metrics = rf_metrics if selected == "random_forest" else lr_metrics
    selected_train_acc = rf_train_acc if selected == "random_forest" else lr_train_acc
    print(f"6. Selected: {selected} (by roc_auc)")

    # 7. Feature importance -----------------------------------------------
    if selected == "random_forest":
        impurity = rf.feature_importances_
        perm = permutation_importance(
            rf, Xt_test, y_test, n_repeats=5, random_state=7, scoring="roc_auc", n_jobs=-1
        ).importances_mean
    else:
        impurity = np.abs(lr.coef_[0])
        perm = permutation_importance(
            lr, Xt_test, y_test, n_repeats=5, random_state=7, scoring="roc_auc", n_jobs=-1
        ).importances_mean

    feature_importance = aggregate_importance(impurity, feature_names)
    permutation_importance_list = aggregate_importance(np.maximum(perm, 0.0), feature_names)
    print("7. Feature importance computed (impurity + permutation)")

    # Risk distribution of the selected model on the test set
    risk_dist = {"low": 0, "medium": 0, "high": 0}
    for p in selected_proba:
        score = round((1.0 - p) * 100)
        if score <= 30:
            risk_dist["low"] += 1
        elif score <= 60:
            risk_dist["medium"] += 1
        else:
            risk_dist["high"] += 1

    # 8. Persist artifacts -------------------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "selected_model": selected,
        "selection_criterion": "roc_auc",
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Simulated Loan Application Dataset",
            "n": int(len(df)),
            "approval_rate": round(approved / len(df), 4),
            "seed": SEED,
            "source": "backend/dataset/loan_dataset.csv (shared with scripts/train.ts)",
        },
        "split": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": SEED,
            "stratified": True,
        },
        "pipeline": {
            "n_features": len(feature_names),
            "numeric": preprocessor.numeric_columns,
            "categorical": preprocessor.categorical_columns,
        },
        "comparison": {
            "logistic_regression": {**lr_metrics, "train_accuracy": round(lr_train_acc, 4)},
            "random_forest": {**rf_metrics, "train_accuracy": round(rf_train_acc, 4)},
        },
        "metrics": selected_metrics,
        "train_accuracy": round(selected_train_acc, 4),
        "feature_importance": feature_importance,
        "permutation_importance": permutation_importance_list,
        "risk_distribution": {**risk_dist, "total": int(len(y_test))},
        "disclaimer": (
            "This tool is an educational decision-support system. Predictions are "
            "machine learning estimates, not guarantees, and do not constitute "
            "financial, legal, or lending advice. A human review is always required "
            "before any real lending decision."
        ),
    }

    joblib.dump({"model": model, "preprocessor": preprocessor, "artifact": artifact},
                MODEL_DIR / "loan_risk_model.pkl")
    joblib.dump(preprocessor.scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(preprocessor.encoder, MODEL_DIR / "encoder.pkl")
    joblib.dump(feature_names, MODEL_DIR / "feature_columns.pkl")
    (MODEL_DIR / "report.json").write_text(json.dumps(artifact, indent=2), "utf-8")

    print("\n8. Wrote artifacts under backend/model/")
    print("   - loan_risk_model.pkl (fitted pipeline + model + artifact)")
    print("   - scaler.pkl, encoder.pkl, feature_columns.pkl")
    print("   - report.json")

    final = artifact["metrics"]
    print(f"\nFinal: acc={final['accuracy'] * 100:.2f}% prec={final['precision'] * 100:.2f}% "
          f"rec={final['recall'] * 100:.2f}% f1={final['f1'] * 100:.2f}% "
          f"auc={final['rocAuc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
