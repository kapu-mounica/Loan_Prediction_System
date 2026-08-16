import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/convex/_generated/api";
import { formatPercent, type ModelInfo } from "@/lib/loan-api";
import { useQuery } from "convex/react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BookOpen,
  Database,
  GitBranch,
  Layers,
  Server,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router";

const API_ENDPOINTS = [
  { method: "GET", path: "/", desc: "Service health + active model summary" },
  { method: "GET", path: "/health", desc: "Same as /, with a timestamp" },
  { method: "GET", path: "/model-info", desc: "Full evaluation summary (metrics, comparison, importance, pipeline)" },
  { method: "GET", path: "/features", desc: "Feature schema used by the prediction forms" },
  { method: "POST", path: "/predict", desc: "Run an application through the trained model → prediction + risk + explanation" },
  { method: "POST", path: "/risk-score", desc: "Compact risk-only response (score, category, probability, verdict)" },
  { method: "POST", path: "/retrain", desc: "Retrain both models on the dataset, select the winner, promote it to production" },
];

const STRUCTURE = `loan-risk-assessment-system/
├── frontend/               → this React + TypeScript app (src/)
│   ├── src/
│   │   ├── pages/          → Landing, Assess, Results, Analytics, Explainability, …
│   │   ├── components/     → layout, risk gauge, confusion matrix, shadcn/ui
│   │   ├── ml/             → TypeScript ML engine (dataset, preprocessing,
│   │   │                     logistic regression, random forest, metrics, explain)
│   │   └── convex/         → Convex backend (schema, loan API, engine, auth)
│   └── public/
├── backend/                → Python FastAPI reference implementation (self-hosted)
│   ├── main.py             → FastAPI app with the same endpoints as the Convex API
│   ├── requirements.txt
│   ├── dataset/
│   │   └── loan_dataset.csv  → the training dataset (shared with the TS pipeline)
│   ├── model/                → artifacts written by train_model.py
│   │   ├── loan_risk_model.pkl
│   │   ├── scaler.pkl
│   │   ├── encoder.pkl
│   │   └── feature_columns.pkl
│   └── ml/
│       ├── train_model.py    → train + evaluate + select (mirrors scripts/train.ts)
│       ├── preprocessing.py  → impute → winsorize → one-hot → standardize
│       ├── evaluate_model.py → accuracy, precision, recall, F1, ROC-AUC, CM
│       └── explain_model.py  → risk score mapping + local ablation factors
├── scripts/train.ts        → offline training pipeline (bun scripts/train.ts)
├── README.md
└── .gitignore`;

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
};

export default function About() {
  const modelInfo = useQuery(api.loan.api.modelInfo) as ModelInfo | undefined;

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6">
      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <Badge variant="secondary" className="gap-1.5">
          <BookOpen className="size-3.5 text-primary" />
          About the project
        </Badge>
        <h1 className="mt-4 text-balance text-3xl font-bold tracking-tight sm:text-4xl">
          Loan Risk Assessment & Approval Prediction System
        </h1>
        <p className="mt-3 max-w-2xl text-pretty text-lg leading-8 text-muted-foreground">
          AI-powered, machine-learning based loan risk assessment. Built for
          B.Tech CSE projects, resumes and portfolios, and placement
          demonstrations — with real training, real evaluation, and a real
          production API.
        </p>
      </motion.div>

      <Tabs defaultValue="methodology" className="mt-8">
        <TabsList className="flex-wrap">
          <TabsTrigger value="methodology" className="gap-1.5">
            <GitBranch className="size-4" />
            Methodology
          </TabsTrigger>
          <TabsTrigger value="dataset" className="gap-1.5">
            <Database className="size-4" />
            Dataset
          </TabsTrigger>
          <TabsTrigger value="api" className="gap-1.5">
            <Server className="size-4" />
            API
          </TabsTrigger>
          <TabsTrigger value="structure" className="gap-1.5">
            <Layers className="size-4" />
            Structure
          </TabsTrigger>
        </TabsList>

        {/* Methodology */}
        <TabsContent value="methodology">
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">Machine learning methodology</CardTitle>
              <CardDescription>
                A complete, reproducible pipeline — the same one the live API runs.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4 text-sm leading-6 text-muted-foreground">
                <div>
                  <p className="font-semibold tracking-tight text-foreground">1 · Data preparation</p>
                  <p className="mt-1">
                    Missing values are imputed (numeric → training median,
                    categorical → training mode), numeric outliers are
                    winsorized at the 1st/99th percentiles, categorical
                    features are one-hot encoded, and numeric features are
                    standardized. The fitted pipeline is persisted and reused
                    verbatim for predictions.
                  </p>
                </div>
                <div>
                  <p className="font-semibold tracking-tight text-foreground">2 · Train / test split</p>
                  <p className="mt-1">
                    A stratified 80/20 split with a fixed random state (
                    {modelInfo ? `seed ${modelInfo.split.randomState}` : "seed 42"})
                    keeps class balance in both sets.
                  </p>
                </div>
                <div>
                  <p className="font-semibold tracking-tight text-foreground">3 · Two models, one winner</p>
                  <p className="mt-1">
                    Logistic Regression and a Random Forest are both trained on
                    the identical split and evaluated on the identical held-out
                    test set. The winner is selected by ROC-AUC — the metric
                    that best captures ranking quality for a risk-scoring
                    system — with accuracy, precision, recall and F1 reported
                    alongside.
                  </p>
                </div>
              </div>
              <div className="space-y-4 text-sm leading-6 text-muted-foreground">
                <div>
                  <p className="font-semibold tracking-tight text-foreground">4 · Evaluation</p>
                  <p className="mt-1">
                    Accuracy, precision, recall, F1-score, ROC-AUC and the
                    confusion matrix are computed on the held-out test set for
                    both models and shown side by side on the analytics page.
                  </p>
                </div>
                <div>
                  <p className="font-semibold tracking-tight text-foreground">5 · Explainability</p>
                  <p className="mt-1">
                    Global importance comes from the selected model
                    (impurity-based) plus an unbiased permutation importance on
                    the test set. Local per-prediction attributions use an
                    ablation (“what-if”) analysis against the training
                    median/mode.
                  </p>
                </div>
                <div>
                  <p className="font-semibold tracking-tight text-foreground">6 · Deployment</p>
                  <p className="mt-1">
                    The winning model is serialized into a single artifact that
                    the Convex API serves in production. Retraining through the
                    API re-runs this entire pipeline and promotes the new
                    winner without any code change.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Dataset */}
        <TabsContent value="dataset">
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">Dataset</CardTitle>
              <CardDescription>
                {modelInfo?.dataset.name} · {modelInfo?.dataset.n.toLocaleString()} applications ·{" "}
                {modelInfo ? formatPercent(modelInfo.dataset.approvalRate, 0) : ""} approval rate
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm leading-6 text-muted-foreground">
              <p>
                The dataset is a <span className="font-medium text-foreground">deterministic, simulated
                loan-application dataset</span> generated by{" "}
                <code className="rounded bg-muted px-1.5 py-0.5 text-xs">src/ml/dataset.ts</code> with a
                fixed seed. It follows the same structure as classic educational
                loan-approval datasets (e.g. the well-known “Loan Approval
                Prediction” datasets with credit history, income, loan amount,
                property area, and education columns), but is generated locally
                so the project is fully reproducible offline.
              </p>
              <p>
                Crucially, the labels are <span className="font-medium text-foreground">not random</span>.
                Each application&apos;s approval probability is computed from an
                interpretable, underwriting-style scoring function — credit
                history, income, debt-to-income ratio, loan size, savings,
                existing debt, employment, education and age — plus a small,
                fixed amount of noise. A model trained on this data learns
                meaningful, consistent financial patterns. Realistic missing
                values are injected (~7% credit history, ~5% loan amount, ~3%
                dependents, ~2% employment status) so the imputation pipeline is
                exercised exactly like a real dataset.
              </p>
              <div>
                <p className="font-semibold tracking-tight text-foreground">16 applicant features + target:</p>
                <p className="mt-1">
                  applicant_age, applicant_income, coapplicant_income,
                  loan_amount, loan_term, credit_history, employment_status,
                  education, marital_status, dependents, property_area,
                  self_employed, existing_loans, monthly_expenses, savings,
                  debt_to_income_ratio →{" "}
                  <span className="font-medium text-foreground">loan_status</span>{" "}
                  (Approved / Rejected) with a derived{" "}
                  <span className="font-medium text-foreground">risk_category</span>{" "}
                  (Low / Medium / High Risk).
                </p>
              </div>
              <div>
                <p className="font-semibold tracking-tight text-foreground">Swapping in a real dataset</p>
                <p className="mt-1">
                  The CSV at{" "}
                  <code className="rounded bg-muted px-1.5 py-0.5 text-xs">backend/dataset/loan_dataset.csv</code>{" "}
                  is the single source of truth. Replace it with any real
                  dataset matching the same column schema, then run{" "}
                  <code className="rounded bg-muted px-1.5 py-0.5 text-xs">bun scripts/train.ts</code>{" "}
                  to retrain, re-evaluate and redeploy — no other change needed.
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                {modelInfo?.dataset.missingness.map((m) => (
                  <Badge key={m} variant="secondary">
                    missing: {m}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* API */}
        <TabsContent value="api">
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg tracking-tight">
                <Server className="size-4 text-primary" />
                API endpoints
              </CardTitle>
              <CardDescription>
                Served by the Convex backend in this deployment; the Python
                FastAPI service in backend/ exposes the identical interface for
                local use.
              </CardDescription>
            </CardHeader>
            <CardContent className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wider text-muted-foreground">
                    <th className="py-2 pr-3 font-medium">Method</th>
                    <th className="py-2 pr-3 font-medium">Path</th>
                    <th className="py-2 font-medium">Description</th>
                  </tr>
                </thead>
                <tbody>
                  {API_ENDPOINTS.map((e) => (
                    <tr key={e.path} className="border-b border-border/50">
                      <td className="py-2.5 pr-3">
                        <Badge variant="secondary" className="font-mono">
                          {e.method}
                        </Badge>
                      </td>
                      <td className="py-2.5 pr-3 font-mono text-xs">{e.path}</td>
                      <td className="py-2.5 text-muted-foreground">{e.desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-4 text-xs leading-5 text-muted-foreground">
                <span className="font-medium text-foreground">Note:</span> in
                this deployment the endpoints are Convex functions —{" "}
                <code className="rounded bg-muted px-1.5 py-0.5 text-xs">
                  api.loan.api.predict
                </code>{" "}
                etc. — called directly from the frontend, and the same contract
                is available as HTTP from the FastAPI service. Applicant data
                is never persisted by any endpoint.
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Structure */}
        <TabsContent value="structure">
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg tracking-tight">
                <Layers className="size-4 text-primary" />
                Project structure
              </CardTitle>
              <CardDescription>
                The deployed app keeps the ML engine in TypeScript (so it runs
                in the same process as the API, with training and inference
                guaranteed identical) while backend/ provides the equivalent
                Python/scikit-learn implementation for local execution.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="overflow-x-auto rounded-xl border border-border/70 bg-muted/40 p-5 text-xs leading-5">
                {STRUCTURE}
              </pre>
              <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="size-4 text-primary" />
                  <p className="text-sm text-muted-foreground">
                    {modelInfo
                      ? `Currently serving: ${modelInfo.selectedAlgorithm} · ${formatPercent(modelInfo.metrics.accuracy)} accuracy · ${formatPercent(modelInfo.metrics.rocAuc)} ROC-AUC`
                      : "Loading model summary…"}
                  </p>
                </div>
                <Button asChild className="shrink-0 gap-2">
                  <Link to="/assess">
                    Try it now
                    <ArrowRight className="size-4" />
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
