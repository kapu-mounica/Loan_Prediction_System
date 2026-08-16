import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/convex/_generated/api";
import { formatPercent, type ModelInfo } from "@/lib/loan-api";
import { useQuery } from "convex/react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Binary,
  Boxes,
  GitBranch,
  Info,
  Layers,
  Scaling,
  Sigma,
  Sparkles,
  TreePine,
  Wand2,
} from "lucide-react";
import { Link } from "react-router";

const PIPELINE_STEPS = [
  {
    icon: Layers,
    title: "1 · Missing values are imputed",
    body: "Numeric features use the training median; categorical features use the training mode. The imputation values are learned during training and reused verbatim at prediction time, so the live API fills gaps exactly like the training set was filled.",
  },
  {
    icon: Boxes,
    title: "2 · Outliers are winsorized",
    body: "Numeric features are clipped at the 1st and 99th percentiles of the training distribution. Extreme but plausible values (a very large loan, very high savings) are capped so a single outlier cannot dominate a prediction.",
  },
  {
    icon: Binary,
    title: "3 · Categorical features are encoded",
    body: "Employment status, education, marital status, property area and self-employment are one-hot encoded into binary indicator columns using the categories observed in training. The same fixed columns are produced for every prediction.",
  },
  {
    icon: Scaling,
    title: "4 · Numeric features are standardized",
    body: "Each numeric feature is rescaled to zero mean and unit variance using the training mean and standard deviation. This keeps large-currency features (income in ₹) on the same scale as small ones (dependents) for the models.",
  },
];

const MODEL_CARDS = [
  {
    icon: Sigma,
    title: "Logistic Regression",
    body: "A linear model that combines the features into a single weighted sum, then passes it through a sigmoid to produce an approval probability between 0 and 1. The learned weights make the model compact and directly inspectable — each feature gets one interpretable coefficient. It is fast and robust on small data, but cannot capture interactions between features.",
    bullet: "Trained with gradient descent, L2 regularization, 1,000 iterations (mirrors scikit-learn defaults).",
  },
  {
    icon: TreePine,
    title: "Random Forest Classifier",
    body: "An ensemble of 80 decision trees, each trained on a bootstrap sample of the training data, with a random subset of features considered at every split (√n). Each tree votes; the forest averages the votes into an approval probability. Bagging and feature randomness decorrelate the trees, which reduces variance and lets the model capture interactions — e.g. a large loan hurting most when credit history is already weak.",
    bullet: "80 trees, max depth 12, min samples split/leaf 6, max features √n, seed 42.",
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
};

export default function Explainability() {
  const modelInfo = useQuery(api.loan.api.modelInfo) as ModelInfo | undefined;

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6">
      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <p className="text-sm font-semibold uppercase tracking-wider text-primary">
          Explainability
        </p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">
          How the system decides
        </h1>
        <p className="mt-2 max-w-3xl text-pretty leading-7 text-muted-foreground">
          Every part of the system is designed to be inspectable: the
          preprocessing pipeline, the trained algorithms, the risk score, and
          the per-prediction factor explanations. None of it is a black box.
        </p>
      </motion.div>

      {/* Pipeline */}
      <motion.div initial="hidden" animate="show" variants={fadeUp} className="mt-8">
        <Card className="border-border/80">
          <CardHeader>
            <CardTitle className="text-lg tracking-tight">
              The preprocessing pipeline (identical at training and prediction)
            </CardTitle>
            <CardDescription>
              {modelInfo
                ? `${modelInfo.pipeline.numeric.length} numeric + ${modelInfo.pipeline.categorical.length} categorical inputs → ${modelInfo.pipeline.nFeatures} model features.`
                : "Loading pipeline…"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              {PIPELINE_STEPS.map((s) => (
                <div key={s.title} className="flex items-start gap-3 rounded-xl border border-border/70 p-4">
                  <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <s.icon className="size-4" />
                  </span>
                  <div>
                    <p className="text-sm font-semibold tracking-tight">{s.title}</p>
                    <p className="mt-1 text-xs leading-5 text-muted-foreground">{s.body}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Models */}
      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          {MODEL_CARDS.map((m) => (
            <Card key={m.title} className="border-border/80">
              <CardHeader>
                <div className="flex items-center gap-3">
                  <span className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                    <m.icon className="size-5" />
                  </span>
                  <CardTitle className="text-lg tracking-tight">{m.title}</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-sm leading-6 text-muted-foreground">{m.body}</p>
                <p className="mt-3 rounded-lg border border-border/70 bg-muted/30 p-3 text-xs leading-5">
                  {m.bullet}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </motion.div>

      {/* Risk score + local explanation */}
      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg tracking-tight">
                <Wand2 className="size-4 text-primary" />
                The 0–100 risk score
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-6 text-muted-foreground">
              <p>
                The selected model outputs an approval probability{" "}
                <span className="font-mono text-xs text-foreground">p</span> in
                [0, 1]. That probability is converted into an interpretable
                risk score:
              </p>
              <div className="rounded-xl border border-border/70 bg-muted/30 p-4 text-center font-mono text-sm">
                risk score = round((1 − p) × 100)
              </div>
              <p>
                The score is mapped to the same three bands used in the dataset
                and the API:
              </p>
              <div className="grid grid-cols-3 gap-3 text-center">
                <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/8 p-3">
                  <p className="text-lg font-bold tabular-nums text-emerald-600 dark:text-emerald-400">0–30</p>
                  <p className="mt-1 text-xs">Low Risk</p>
                </div>
                <div className="rounded-xl border border-amber-500/25 bg-amber-500/8 p-3">
                  <p className="text-lg font-bold tabular-nums text-amber-600 dark:text-amber-400">31–60</p>
                  <p className="mt-1 text-xs">Medium Risk</p>
                </div>
                <div className="rounded-xl border border-rose-500/25 bg-rose-500/8 p-3">
                  <p className="text-lg font-bold tabular-nums text-rose-600 dark:text-rose-400">61–100</p>
                  <p className="mt-1 text-xs">High Risk</p>
                </div>
              </div>
              <p>
                {modelInfo?.riskScore.interpretation}
              </p>
            </CardContent>
          </Card>

          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg tracking-tight">
                <GitBranch className="size-4 text-primary" />
                Per-prediction factor explanations
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm leading-6 text-muted-foreground">
              <p>
                For every prediction, the result page lists the factors that
                moved the verdict. Each one is computed with an{" "}
                <span className="font-medium text-foreground">ablation
                (“what-if”) analysis</span>:
              </p>
              <div className="rounded-xl border border-border/70 bg-muted/30 p-4 text-xs leading-5">
                <ol className="list-inside list-decimal space-y-1.5">
                  <li>Predict with the applicant&apos;s real values → baseline probability p.</li>
                  <li>
                    For each feature, replace its value with the training
                    median/mode and re-run the model → probability p′.
                  </li>
                  <li>
                    The attribution is the change Δp = p − p′. A large |Δp|
                    means the feature strongly influenced this prediction;
                    its sign shows the direction (toward approval or rejection).
                  </li>
                </ol>
              </div>
              <p>
                This is a standard local-attribution technique (an
                approximation of SHAP-style values) and is documented as such:
                feature importance describes model behavior, not proof of a
                causal financial relationship.
              </p>
              {modelInfo && (
                <div className="flex flex-wrap gap-2 pt-1">
                  <Badge className="gap-1.5 bg-primary/10 text-primary hover:bg-primary/15">
                    <Sparkles className="size-3" />
                    Top global factor: {modelInfo.featureImportance[0]?.label} (
                    {formatPercent(modelInfo.featureImportance[0]?.importance ?? 0)})
                  </Badge>
                  <Badge variant="secondary" className="gap-1.5">
                    <Info className="size-3" />
                    {modelInfo.pipeline.nFeatures} model features
                  </Badge>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </motion.div>

      {/* CTA */}
      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <div className="mt-6 flex flex-col items-start gap-4 rounded-2xl border border-border/80 bg-card p-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="font-semibold tracking-tight">See it on a real application</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Run an assessment and inspect the live factor attributions on the result page.
            </p>
          </div>
          <Button asChild className="shrink-0 gap-2">
            <Link to="/assess">
              Start an assessment
              <ArrowRight className="size-4" />
            </Link>
          </Button>
        </div>
      </motion.div>
    </div>
  );
}
