import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { RiskGauge } from "@/components/risk/RiskGauge";
import {
  formatCurrency,
  formatPercent,
  RISK_BADGE,
  riskTone,
  type ModelInfo,
} from "@/lib/loan-api";
import { api } from "@/convex/_generated/api";
import { useAction, useQuery } from "convex/react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  ClipboardList,
  Eye,
  Gauge,
  LineChart,
  Loader2,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  UserCheck,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const DEMO_INPUT = {
  applicant_age: 32,
  applicant_income: 55000,
  coapplicant_income: 25000,
  loan_amount: 150000,
  loan_term: 360,
  credit_history: 1,
  employment_status: "Employed",
  education: "Graduate",
  marital_status: "Married",
  dependents: 1,
  property_area: "Urban",
  self_employed: "No",
  existing_loans: 1,
  monthly_expenses: 22000,
  savings: 120000,
  debt_to_income_ratio: 0.4,
};

const FEATURES = [
  {
    icon: ClipboardList,
    title: "Loan Approval Prediction",
    body: "A trained Random Forest classifier — benchmarked against Logistic Regression — predicts approval from 16 financial and application features.",
  },
  {
    icon: Gauge,
    title: "Interpretable Risk Score",
    body: "Every prediction comes with a 0–100 risk score and a Low / Medium / High category, so the output means something concrete.",
  },
  {
    icon: Eye,
    title: "Explainable AI",
    body: "See which factors drove the decision — credit history, income, loan size, savings, debt — with honest global and local explanations.",
  },
  {
    icon: LineChart,
    title: "Model Analytics",
    body: "Full evaluation dashboard: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, feature importance, and model comparison.",
  },
  {
    icon: ShieldCheck,
    title: "Responsible AI",
    body: "No protected characteristics as features, no stored applicant data, human review required, and predictions clearly framed as non-guarantees.",
  },
];

const STEPS = [
  {
    step: "01",
    title: "Fill the application",
    body: "Enter applicant, financial, and loan details in a guided, validated three-section form.",
  },
  {
    step: "02",
    title: "The model evaluates",
    body: "The same preprocessing pipeline used in training runs the exact production model — Random Forest vs Logistic Regression, picked on ROC-AUC.",
  },
  {
    step: "03",
    title: "Get an explained result",
    body: "Approval prediction, probability, risk score, and the factors behind the decision — at Beginner, Intermediate, or Advanced depth.",
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5, ease: "easeOut" as const } },
};

function LiveDemo() {
  const predict = useAction(api.loan.api.predict);
  const [state, setState] = useState<{
    status: "loading" | "done" | "error";
    prediction?: "Approved" | "Rejected";
    probability?: number;
    riskScore?: number;
    riskCategory?: string;
    topFactor?: string;
  }>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    predict({ input: DEMO_INPUT })
      .then((r) => {
        if (cancelled) return;
        setState({
          status: "done",
          prediction: r.prediction,
          probability: r.probabilityApproved,
          riskScore: r.riskScore,
          riskCategory: r.riskCategory,
          topFactor: r.factors[0]?.label,
        });
      })
      .catch(() => {
        if (!cancelled) setState({ status: "error" });
      });
    return () => {
      cancelled = true;
    };
  }, [predict]);

  const tone = state.riskCategory ? riskTone(state.riskCategory) : "low";

  return (
    <Card className="relative overflow-hidden border-border/80 shadow-xl shadow-primary/5">
      <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary/80 via-chart-2 to-chart-3" />
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
            <Sparkles className="size-4 text-primary" />
            Live model preview
          </div>
          <Badge variant="secondary" className="gap-1">
            <TrendingUp className="size-3" />
            Sample applicant
          </Badge>
        </div>

        {state.status === "loading" && (
          <div className="flex h-64 flex-col items-center justify-center gap-3 text-muted-foreground">
            <Loader2 className="size-6 animate-spin" />
            <p className="text-sm">Running the production model…</p>
          </div>
        )}

        {state.status === "error" && (
          <div className="flex h-64 flex-col items-center justify-center gap-2 text-center">
            <p className="text-sm font-medium">Model preview unavailable</p>
            <p className="max-w-xs text-xs text-muted-foreground">
              The prediction API could not be reached. The assessment form
              will still work once the service is up.
            </p>
          </div>
        )}

        {state.status === "done" && state.riskScore !== undefined && (
          <div className="mt-4 flex flex-col items-center gap-4">
            <div className="flex w-full items-center justify-between gap-3">
              <div className="text-left">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Prediction
                </p>
                <p
                  className={`mt-1 text-2xl font-bold tracking-tight ${
                    state.prediction === "Approved" ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"
                  }`}
                >
                  {state.prediction}
                </p>
              </div>
              <div className="text-right">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Probability
                </p>
                <p className="tabular-nums mt-1 text-2xl font-bold tracking-tight">
                  {formatPercent(state.probability ?? 0, 0)}
                </p>
              </div>
            </div>

            <RiskGauge score={state.riskScore} category={state.riskCategory ?? ""} size="sm" />

            <div className="flex w-full flex-col gap-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Risk level</span>
                <Badge className={RISK_BADGE[tone]}>{state.riskCategory}</Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Top factor</span>
                <span className="font-medium">{state.topFactor}</span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function Landing() {
  const modelInfo = useQuery(api.loan.api.modelInfo) as ModelInfo | undefined;

  const comparisonData = modelInfo
    ? (["accuracy", "precision", "recall", "f1", "rocAuc"] as const).map((m) => ({
        metric: m.toUpperCase(),
        "Logistic Regression": modelInfo.comparison[0][m],
        "Random Forest": modelInfo.comparison[1][m],
      }))
    : [];

  return (
    <div className="overflow-hidden">
      {/* ------------------------------------------------------------------ */}
      {/* Hero                                                                 */}
      {/* ------------------------------------------------------------------ */}
      <section className="relative">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(60rem 30rem at 15% -10%, color-mix(in oklab, var(--primary) 10%, transparent), transparent 60%), radial-gradient(50rem 26rem at 90% 0%, color-mix(in oklab, var(--chart-2) 9%, transparent), transparent 60%)",
          }}
        />
        <div className="relative mx-auto grid w-full max-w-7xl gap-14 px-4 pb-20 pt-16 sm:px-6 lg:grid-cols-[1.1fr_0.9fr] lg:items-center lg:pt-24">
          <motion.div initial="hidden" animate="show" variants={fadeUp}>
            <Badge variant="secondary" className="gap-1.5 px-3 py-1">
              <Sparkles className="size-3.5 text-primary" />
              Machine Learning Based Loan Approval &amp; Risk Analysis
            </Badge>
            <h1 className="mt-6 text-balance text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              AI-Powered{" "}
              <span className="text-primary">Loan Risk</span> Assessment
            </h1>
            <p className="mt-5 max-w-xl text-pretty text-lg leading-8 text-muted-foreground">
              Submit a loan application and get a real prediction — trained
              Random Forest and Logistic Regression models, an interpretable
              0–100 risk score, and a clear explanation of every factor that
              mattered.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Button asChild size="lg" className="gap-2">
                <Link to="/assess">
                  Start Assessment
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="gap-2">
                <Link to="/analytics">
                  <BarChart3 className="size-4" />
                  View model analytics
                </Link>
              </Button>
            </div>
            <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-muted-foreground">
              <span className="flex items-center gap-1.5">
                <UserCheck className="size-4 text-primary" />
                No applicant data stored
              </span>
              <span className="flex items-center gap-1.5">
                <ShieldCheck className="size-4 text-primary" />
                Responsible AI by design
              </span>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15, ease: "easeOut" }}
          >
            <LiveDemo />
          </motion.div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Stats band                                                           */}
      {/* ------------------------------------------------------------------ */}
      {modelInfo && (
        <section className="border-y border-border/70 bg-card/60">
          <div className="mx-auto grid w-full max-w-7xl grid-cols-2 gap-px px-4 py-8 sm:px-6 md:grid-cols-4">
            {[
              { label: "Test accuracy", value: formatPercent(modelInfo.metrics.accuracy) },
              { label: "ROC-AUC", value: formatPercent(modelInfo.metrics.rocAuc) },
              { label: "Models compared", value: "2" },
              { label: "Risk classes", value: "3" },
            ].map((s) => (
              <div key={s.label} className="px-4 py-3 text-center">
                <p className="tabular-nums text-2xl font-bold tracking-tight text-foreground">
                  {s.value}
                </p>
                <p className="mt-1 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  {s.label}
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Features                                                             */}
      {/* ------------------------------------------------------------------ */}
      <section className="mx-auto w-full max-w-7xl px-4 py-20 sm:px-6">
        <div className="max-w-2xl">
          <p className="text-sm font-semibold uppercase tracking-wider text-primary">
            What you get
          </p>
          <h2 className="mt-2 text-balance text-3xl font-bold tracking-tight sm:text-4xl">
            A complete ML risk pipeline, not a mockup
          </h2>
          <p className="mt-4 text-pretty leading-7 text-muted-foreground">
            From real training and evaluation to production prediction with
            explainability — every component below is live and connected.
          </p>
        </div>
        <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.4, delay: i * 0.05 }}
            >
              <Card className="h-full border-border/80 transition-shadow hover:shadow-lg hover:shadow-primary/5">
                <CardContent className="flex h-full flex-col gap-4 p-6">
                  <span className="flex size-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
                    <f.icon className="size-5" />
                  </span>
                  <h3 className="text-lg font-semibold tracking-tight">{f.title}</h3>
                  <p className="text-sm leading-6 text-muted-foreground">{f.body}</p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
          <Card className="flex h-full flex-col justify-between border-dashed border-border">
            <CardContent className="flex h-full flex-col gap-4 p-6">
              <span className="flex size-11 items-center justify-center rounded-xl bg-muted text-muted-foreground">
                <Gauge className="size-5" />
              </span>
              <div>
                <h3 className="text-lg font-semibold tracking-tight">See it in action</h3>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  Run a complete assessment with a sample application in under a minute.
                </p>
              </div>
              <Button asChild variant="secondary" className="mt-auto w-fit gap-2">
                <Link to="/assess">
                  Start Assessment
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* How it works                                                         */}
      {/* ------------------------------------------------------------------ */}
      <section className="border-y border-border/70 bg-card/50">
        <div className="mx-auto w-full max-w-7xl px-4 py-20 sm:px-6">
          <div className="max-w-2xl">
            <p className="text-sm font-semibold uppercase tracking-wider text-primary">
              How it works
            </p>
            <h2 className="mt-2 text-balance text-3xl font-bold tracking-tight sm:text-4xl">
              From application to explained prediction
            </h2>
          </div>
          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {STEPS.map((s, i) => (
              <motion.div
                key={s.step}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ duration: 0.4, delay: i * 0.08 }}
                className="relative rounded-2xl border border-border/80 bg-background p-6 shadow-sm"
              >
                <span className="tabular-nums text-4xl font-bold tracking-tight text-primary/25">
                  {s.step}
                </span>
                <h3 className="mt-3 text-lg font-semibold tracking-tight">{s.title}</h3>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">{s.body}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Model comparison                                                     */}
      {/* ------------------------------------------------------------------ */}
      {modelInfo && (
        <section className="mx-auto w-full max-w-7xl px-4 py-20 sm:px-6">
          <div className="grid items-center gap-10 lg:grid-cols-2">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wider text-primary">
                The models
              </p>
              <h2 className="mt-2 text-balance text-3xl font-bold tracking-tight sm:text-4xl">
                Two algorithms trained, the better one selected
              </h2>
              <p className="mt-4 text-pretty leading-7 text-muted-foreground">
                Logistic Regression and a Random Forest were trained on the same
                dataset, split with a fixed random state, and evaluated on the
                same held-out test set. The winner is chosen by ROC-AUC — the
                metric that best captures how well a model ranks risk — not by
                assumption.
              </p>
              <div className="mt-6 flex items-center gap-3 rounded-xl border border-border/80 bg-card p-4">
                <Badge className="gap-1.5 bg-primary/10 text-primary hover:bg-primary/15">
                  <Sparkles className="size-3" />
                  Selected: {modelInfo.selectedAlgorithm}
                </Badge>
                <p className="text-sm text-muted-foreground">
                  {modelInfo.selectionCriterion}
                </p>
              </div>
              <Button asChild variant="outline" className="mt-6 gap-2">
                <Link to="/analytics">
                  <LineChart className="size-4" />
                  Full analytics dashboard
                </Link>
              </Button>
            </div>
            <Card className="border-border/80 shadow-lg shadow-primary/5">
              <CardContent className="p-6">
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={comparisonData} barGap={4}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                      <XAxis dataKey="metric" tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" />
                      <YAxis
                        domain={[0, 1]}
                        tickFormatter={(v) => `${Math.round(v * 100)}%`}
                        tick={{ fontSize: 12 }}
                        stroke="var(--muted-foreground)"
                        width={40}
                      />
                      <Tooltip
                        formatter={(value) => formatPercent(Number(value))}
                        contentStyle={{
                          background: "var(--card)",
                          border: "1px solid var(--border)",
                          borderRadius: 12,
                          fontSize: 13,
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      <Bar dataKey="Logistic Regression" fill="var(--chart-2)" radius={[6, 6, 0, 0]} maxBarSize={26} />
                      <Bar dataKey="Random Forest" fill="var(--chart-1)" radius={[6, 6, 0, 0]} maxBarSize={26} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Responsible AI banner                                                */}
      {/* ------------------------------------------------------------------ */}
      <section className="mx-auto w-full max-w-7xl px-4 pb-20 sm:px-6">
        <Card className="border-primary/25 bg-gradient-to-br from-primary/8 via-card to-card shadow-lg shadow-primary/5">
          <CardContent className="flex flex-col gap-6 p-8 md:flex-row md:items-center md:justify-between">
            <div className="max-w-2xl">
              <div className="flex items-center gap-2">
                <ShieldCheck className="size-5 text-primary" />
                <h3 className="text-lg font-semibold tracking-tight">
                  Built responsibly, by design
                </h3>
              </div>
              <p className="mt-3 text-sm leading-6 text-muted-foreground">
                No protected characteristics are used as features, applicant
                data is never stored, and every result is framed as an
                educational model prediction — never a lending decision.
                Human review is required before any real-world use.
              </p>
            </div>
            <Button asChild variant="outline" className="shrink-0 gap-2">
              <Link to="/responsible-ai">
                Read the Responsible AI principles
                <ArrowRight className="size-4" />
              </Link>
            </Button>
          </CardContent>
        </Card>
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Final CTA                                                             */}
      {/* ------------------------------------------------------------------ */}
      <section className="border-t border-border/70 bg-card/60">
        <div className="mx-auto flex w-full max-w-7xl flex-col items-center px-4 py-16 text-center sm:px-6">
          <h2 className="max-w-2xl text-balance text-3xl font-bold tracking-tight sm:text-4xl">
            Ready to see the model think?
          </h2>
          <p className="mt-4 max-w-xl text-pretty leading-7 text-muted-foreground">
            Enter an application and get a real, explained prediction with an
            interpretable risk score — in about a minute.
          </p>
          <Button asChild size="lg" className="mt-8 gap-2">
            <Link to="/assess">
              Start Assessment
              <ArrowRight className="size-4" />
            </Link>
          </Button>
        </div>
      </section>
    </div>
  );
}
