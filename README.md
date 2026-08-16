# Loan Risk Assessment & Approval Prediction System

**AI-Powered Machine Learning Based Loan Risk Assessment**

A complete, production-shaped loan risk assessment and approval prediction
system: a trained machine-learning model evaluates loan applications and
returns an approval prediction, an interpretable 0–100 risk score, and an
explanation of the factors that drove the decision.

Built for **B.Tech CSE projects, resumes/portfolios, and placement
demonstrations** — every layer is real: dataset generation, preprocessing,
training, evaluation, model selection, a live prediction API, an analytics
dashboard, and explainability.

---

## Live deployment

The app is deployed on the Freebuff/Vly platform:

- **Frontend** — React 19 + TypeScript + Vite + Tailwind CSS v4 + shadcn/ui + Recharts + Framer Motion
- **Backend / API** — Convex (queries, mutations, actions). No applicant data is stored.
- **Machine learning** — a pure TypeScript ML engine (`src/ml/`) implementing
  real algorithms (gradient-descent logistic regression, bagged decision-tree
  random forest) that runs in the same process as the API, so training and
  inference share the exact same preprocessing and model code.

A **Python FastAPI + scikit-learn** reference implementation of the same
system lives in [`backend/`](#python-fastapi-backend) for local/self-hosted
execution (e.g. to demonstrate a classic Python ML stack in a presentation).

---

## Project structure

```
loan-risk-assessment-system/
│
├── frontend/  (this repo's src/)          # React + TypeScript app
│   ├── src/
│   │   ├── pages/                         # Landing, Assess, Results, Analytics,
│   │   │                                  #   Explainability, ResponsibleAI, About
│   │   ├── components/                    # AppLayout, RiskGauge, ConfusionMatrix, shadcn/ui
│   │   ├── ml/                            # TypeScript ML engine
│   │   │   ├── dataset.ts                 # deterministic dataset generator (seed 42)
│   │   │   ├── preprocess.ts              # impute → winsorize → one-hot → standardize
│   │   │   ├── logistic.ts                # Logistic Regression (gradient descent)
│   │   │   ├── tree.ts                    # Random Forest (bagged decision trees)
│   │   │   ├── metrics.ts                 # accuracy, precision, recall, F1, ROC-AUC, CM
│   │   │   ├── explain.ts                 # risk score mapping + local ablation factors
│   │   │   ├── train.ts                   # shared training + evaluation + selection
│   │   │   └── artifacts/model.json       # the trained, deployed model artifact
│   │   └── convex/                        # Convex backend
│   │       ├── schema.ts                  # users + models tables
│   │       ├── loan/api.ts                # health, model-info, features, predict, risk-score, retrain
│   │       ├── loan/engine.ts             # inference (deserialize + predict)
│   │       ├── loan/validate.ts           # input validation rules
│   │       └── loanData.ts                # generated dataset (retrain endpoint)
│   └── public/
│
├── backend/                               # Python FastAPI reference implementation
│   ├── main.py                            # FastAPI app (same endpoints as the Convex API)
│   ├── requirements.txt
│   ├── dataset/
│   │   └── loan_dataset.csv               # the training dataset (shared with the TS pipeline)
│   ├── model/                             # artifacts written by train_model.py
│   │   ├── loan_risk_model.pkl
│   │   ├── scaler.pkl
│   │   ├── encoder.pkl
│   │   └── feature_columns.pkl
│   └── ml/
│       ├── train_model.py                 # train + evaluate + select + persist
│       ├── preprocessing.py               # mirrors src/ml/preprocess.ts
│       ├── evaluate_model.py              # metrics (sklearn)
│       └── explain_model.py               # risk score + local factors
│
├── scripts/train.ts                       # offline training pipeline
├── README.md
└── .gitignore
```

---

## Dataset

`backend/dataset/loan_dataset.csv` (1,500 applications) is the single source
of truth for training.

**Source:** the dataset is **simulated, not scraped** — it is generated
deterministically by `src/ml/dataset.ts` (fixed seed 42) so the whole project
is reproducible offline. It follows the structure of classic educational
"loan approval prediction" datasets (credit history, income, loan amount,
property area, education, …). The full generation logic, including the
labeling function, is documented in the source file.

**Labels are not random.** Each application's approval probability comes from
an interpretable, underwriting-style scoring function — credit history,
income, debt-to-income ratio, loan size, savings, existing debt, employment,
education, and age — plus a small fixed amount of noise. Models trained on
this data learn consistent, meaningful financial patterns.

**Features (16) + target:**

`applicant_age, applicant_income, coapplicant_income, loan_amount, loan_term,
credit_history, employment_status, education, marital_status, dependents,
property_area, self_employed, existing_loans, monthly_expenses, savings,
debt_to_income_ratio` → **`loan_status`** (`Approved` / `Rejected`) plus a
derived **`risk_category`** (`Low Risk` / `Medium Risk` / `High Risk`).

**Missing values** are injected realistically (~7% credit history, ~5% loan
amount, ~3% dependents, ~2% employment status) so the imputation pipeline is
exercised exactly like a real dataset.

> **Swap in a real dataset:** replace `backend/dataset/loan_dataset.csv` with
> any dataset matching the same column schema, then run
> `bun scripts/train.ts` (and/or `python -m backend.ml.train_model`) to
> retrain, re-evaluate, and redeploy. No other change is required.

---

## Machine learning methodology

1. **Preprocessing** — missing values imputed (numeric → training median,
   categorical → training mode), numeric outliers winsorized at the 1st/99th
   percentiles, categorical features one-hot encoded, numeric features
   standardized. The fitted pipeline is persisted and reused verbatim at
   prediction time (both in `src/ml/preprocess.ts` and
   `backend/ml/preprocessing.py`).
2. **Split** — stratified 80/20 train/test split with a fixed random state
   (seed 42).
3. **Models** — both **Logistic Regression** (gradient descent, 1,000
   iterations) and **Random Forest Classifier** (80 trees, max depth 12, √n
   features per split) are trained on the identical split.
4. **Evaluation** — accuracy, precision, recall, F1-score, ROC-AUC and the
   confusion matrix are computed on the identical held-out test set for both
   models.
5. **Selection** — the better model is chosen **by ROC-AUC** (the metric that
   best measures how well a model ranks risk across thresholds), not by
   assumption. The analytics page shows the full head-to-head comparison.
6. **Explainability** — global impurity-based feature importance plus an
   unbiased permutation importance on the test set; per-prediction local
   attributions via ablation ("what-if") analysis.
7. **Retraining** — the `POST /retrain` endpoint (or `bun scripts/train.ts`)
   re-runs the entire pipeline and promotes the new winner to production
   without any code change.

### Live metrics

The currently deployed model's metrics are always visible on the landing
page, the dashboard, and `/analytics` (accuracy, ROC-AUC, F1, precision,
recall, confusion matrix, feature importance, risk distribution).

---

## API

The Convex backend exposes the following contract (the Python FastAPI service
in `backend/` exposes the identical HTTP contract):

| Method | Endpoint      | Description |
| ------ | ------------- | ----------- |
| GET    | `/`           | Health + active model summary |
| GET    | `/health`     | Same, with a timestamp |
| GET    | `/model-info` | Full evaluation summary (metrics, comparison, importance, pipeline) |
| GET    | `/features`   | Feature schema used by the forms |
| POST   | `/predict`    | Run an application → prediction, probability, risk score, category, factors, indicators, model info, disclaimer |
| POST   | `/risk-score` | Compact risk-only response |
| POST   | `/retrain`    | Retrain both models, select the winner, promote to production |

Example `POST /predict` body:

```json
{
  "input": {
    "applicant_age": 32,
    "applicant_income": 55000,
    "coapplicant_income": 25000,
    "loan_amount": 150000,
    "loan_term": 360,
    "credit_history": 1,
    "employment_status": "Employed",
    "education": "Graduate",
    "marital_status": "Married",
    "dependents": 1,
    "property_area": "Urban",
    "self_employed": "No",
    "existing_loans": 1,
    "monthly_expenses": 22000,
    "savings": 120000,
    "debt_to_income_ratio": 0.4
  }
}
```

No endpoint stores applicant data; results are computed and returned only.

---

## Frontend

- **`/`** — landing page with a live model preview (a real prediction, not a
  mockup), stats band, feature grid, how-it-works, and the LR vs RF comparison
  chart.
- **`/assess`** — a guided three-section application form (Applicant →
  Financial → Loan) with the same validation rules as the API.
- **`/results`** — the explained verdict: prediction, probability, risk gauge,
  risk category, top factors with direction, financial indicators, model
  metrics, and the disclaimer.
- **`/analytics`** — full evaluation dashboard: head-to-head metric table and
  chart, confusion matrix, impurity + permutation feature importance, risk
  distribution, and pipeline/dataset details.
- **`/explain`** — how the pipeline, both algorithms, the risk score, and the
  per-prediction explanations work.
- **`/dashboard`** — authenticated model overview with a **Retrain** action.
- **`/responsible-ai`**, **`/about`** — principles and documentation.

All product pages are protected by `RequireAuth`; signing in returns you to
the page you requested.

---

## Python FastAPI backend

The deployed web app runs the ML engine in TypeScript on Convex (so it is
fully self-contained and serverless). The `backend/` directory additionally
provides the equivalent **Python/scikit-learn implementation** — the same
pipeline, models, metrics, selection rule and endpoints — for local execution
or a classic Python-ML presentation:

```bash
cd loan-risk-assessment-system
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt

python -m backend.ml.train_model                    # trains + evaluates + writes backend/model/*.pkl
uvicorn backend.main:app --reload                   # API at http://localhost:8000/docs
```

---

## Development

```bash
bun install
bun convex dev --once     # Convex codegen
bun tsc -b --noEmit       # typecheck
bun scripts/train.ts      # (re)train the model and regenerate artifacts
bun run dev               # start the Vite dev server
```

The frontend talks to Convex via `src/convex/loan/api.ts`; there is no local
server to run for the deployed app.

---

## Responsible AI

This is an **educational decision-support system**, not a lending decision
maker:

- No protected characteristics (gender, caste, religion, …) are used as features.
- Applicant data is **never stored** — results live only in the browser tab.
- Every prediction ships with a disclaimer and requires human review before
  any real-world use.
- Models are trained on a simulated dataset; real-world performance requires
  retraining on real, representative data.

---

## License & disclaimer

Educational project. Predictions are machine learning estimates, not
guarantees, and do not constitute financial, legal, or lending advice.
