import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { RiskGauge } from "@/components/risk/RiskGauge";
import {
  formatCurrency,
  formatDate,
  formatPercent,
  RISK_BADGE,
  riskTone,
  titleCase,
} from "@/lib/loan-api";
import { loadAssessment } from "@/lib/loan-api";
import { motion } from "framer-motion";
import {
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BrainCircuit,
  ClipboardList,
  Info,
  LineChart,
  Lock,
  ShieldCheck,
  Wallet,
} from "lucide-react";
import { Link } from "react-router";

const INDICATOR_LABELS: Record<string, { label: string; format: (v: number) => string }> = {
  debtToIncomeRatio: { label: "Debt-to-income ratio", format: (v) => `${(v * 100).toFixed(1)}%` },
  loanToAnnualIncome: { label: "Loan ÷ annual income", format: (v) => `${v.toFixed(2)}×` },
  savingsToLoan: { label: "Savings ÷ loan", format: (v) => `${v.toFixed(2)}×` },
  expenseRatio: { label: "Expense ÷ income", format: (v) => `${(v * 100).toFixed(1)}%` },
  monthlyLoanBurdenEstimate: { label: "Est. monthly payment (9% p.a.)", format: (v) => formatCurrency(v) },
  loanTermYears: { label: "Loan term", format: (v) => `${v} years` },
};

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
};

export default function Results() {
  const stored = loadAssessment();

  if (!stored) {
    return (
      <div className="mx-auto flex w-full max-w-2xl flex-col items-center px-4 py-24 text-center sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="flex flex-col items-center"
        >
          <span className="flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
            <ClipboardList className="size-7" />
          </span>
          <h1 className="mt-6 text-3xl font-bold tracking-tight">No assessment yet</h1>
          <p className="mt-3 max-w-md text-pretty leading-7 text-muted-foreground">
            Run a loan application through the trained model and the explained
            verdict — prediction, probability, risk score and the factors
            behind it — will appear here.
          </p>
          <Button asChild size="lg" className="mt-8 gap-2">
            <Link to="/assess">
              Start an assessment
              <ArrowRight className="size-4" />
            </Link>
          </Button>
        </motion.div>
      </div>
    );
  }

  const { input, result, createdAt } = stored;
  const tone = riskTone(result.riskCategory);
  const approved = result.prediction === "Approved";
  const topFactors = result.factors.slice(0, 5);

  return (
    <div className="mx-auto w-full max-w-6xl px-4 py-10 sm:px-6">
      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <p className="text-sm font-semibold uppercase tracking-wider text-primary">
          Assessment Result
        </p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">
          Model verdict
        </h1>
        <p className="mt-2 text-muted-foreground">
          Run {new Date(createdAt).toLocaleString()} ·{" "}
          <span className="inline-flex items-center gap-1">
            <Lock className="size-3.5" /> result kept in this browser tab only
          </span>
        </p>
      </motion.div>

      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_1.1fr]">
        {/* Verdict */}
        <motion.div initial="hidden" animate="show" variants={fadeUp}>
          <Card
            className={`relative overflow-hidden border-border/80 ${
              approved
                ? "bg-gradient-to-br from-emerald-500/8 via-card to-card"
                : "bg-gradient-to-br from-rose-500/8 via-card to-card"
            }`}
          >
            <div
              className={`absolute inset-x-0 top-0 h-1 ${
                approved ? "bg-emerald-500" : "bg-rose-500"
              }`}
            />
            <CardContent className="flex flex-col items-center gap-5 p-6">
              <div className="flex w-full flex-col items-center gap-3">
                <Badge
                  className={`px-3 py-1 text-sm ${
                    approved
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-300"
                      : "border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-300"
                  }`}
                >
                  {approved ? "APPROVED" : "REJECTED"}
                </Badge>
                <p className="text-sm text-muted-foreground">
                  Approval probability{" "}
                  <span className="font-semibold tabular-nums text-foreground">
                    {formatPercent(result.probabilityApproved)}
                  </span>
                </p>
              </div>

              <RiskGauge score={result.riskScore} category={result.riskCategory} size="lg" />

              <div className="grid w-full grid-cols-2 gap-3">
                <div className="rounded-xl border border-border/70 p-3 text-center">
                  <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Risk category
                  </p>
                  <Badge className={`mt-1.5 ${RISK_BADGE[tone]}`}>{result.riskCategory}</Badge>
                </div>
                <div className="rounded-xl border border-border/70 p-3 text-center">
                  <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Risk score
                  </p>
                  <p className="mt-1.5 text-lg font-bold tabular-nums tracking-tight">
                    {result.riskScore}
                    <span className="text-xs font-normal text-muted-foreground"> / 100</span>
                  </p>
                </div>
              </div>

              <div className="flex w-full flex-wrap justify-center gap-2">
                <Button asChild size="sm" className="gap-2">
                  <Link to="/assess">
                    <ClipboardList className="size-4" />
                    New assessment
                  </Link>
                </Button>
                <Button asChild size="sm" variant="outline" className="gap-2">
                  <Link to="/analytics">
                    <LineChart className="size-4" />
                    Model analytics
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial="hidden"
          animate="show"
          variants={fadeUp}
          transition={{ delay: 0.08 }}
          className="flex flex-col gap-6"
        >
          {/* Factors */}
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">What drove this decision</CardTitle>
              <CardDescription>
                Local ablation attribution: how strongly each feature pulled the
                prediction toward approval or rejection.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {topFactors.map((f) => (
                <div
                  key={f.feature}
                  className="flex items-start gap-3 rounded-xl border border-border/70 p-3"
                >
                  <span
                    className={`mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg ${
                      f.direction === "positive"
                        ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                        : "bg-rose-500/10 text-rose-600 dark:text-rose-400"
                    }`}
                  >
                    {f.direction === "positive" ? (
                      <ArrowUpRight className="size-4" />
                    ) : (
                      <ArrowDownRight className="size-4" />
                    )}
                  </span>
                  <div className="min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-semibold tracking-tight">{f.label}</p>
                      <span
                        className={`text-xs font-medium tabular-nums ${
                          f.direction === "positive"
                            ? "text-emerald-600 dark:text-emerald-400"
                            : "text-rose-600 dark:text-rose-400"
                        }`}
                      >
                        {f.direction === "positive" ? "+" : "−"}
                        {formatPercent(Math.abs(f.delta))}
                      </span>
                    </div>
                    <p className="mt-0.5 text-xs leading-5 text-muted-foreground">
                      {f.description}
                    </p>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Indicators */}
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg tracking-tight">
                <Wallet className="size-4 text-primary" />
                Financial indicators
              </CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {Object.entries(INDICATOR_LABELS).map(([key, def]) => {
                const value = result.financialIndicators[key as keyof typeof result.financialIndicators];
                if (typeof value !== "number") return null;
                return (
                  <div key={key} className="rounded-xl border border-border/70 p-3">
                    <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                      {def.label}
                    </p>
                    <p className="mt-1.5 text-base font-bold tabular-nums tracking-tight">
                      {def.format(value)}
                    </p>
                  </div>
                );
              })}
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Model + disclaimer */}
      <motion.div initial="hidden" animate="show" variants={fadeUp} transition={{ delay: 0.12 }}>
        <Card className="mt-6 border-border/80">
          <CardContent className="flex flex-col gap-5 p-6 md:flex-row md:items-start md:justify-between">
            <div className="flex items-start gap-3">
              <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <BrainCircuit className="size-5" />
              </span>
              <div>
                <p className="text-sm font-semibold tracking-tight">
                  {result.model.algorithm} · v{result.model.version}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Trained {formatDate(result.model.trainedAt)} · test accuracy{" "}
                  {formatPercent(result.model.metrics.accuracy)} · ROC-AUC{" "}
                  {formatPercent(result.model.metrics.rocAuc)}
                </p>
                <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                  <span>Precision {formatPercent(result.model.metrics.precision)}</span>
                  <span>Recall {formatPercent(result.model.metrics.recall)}</span>
                  <span>F1 {formatPercent(result.model.metrics.f1)}</span>
                </div>
              </div>
            </div>
            <Link to="/explain" className="shrink-0 text-sm font-medium text-primary hover:underline">
              How the model works →
            </Link>
          </CardContent>
        </Card>

        <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-border/80 bg-card p-4 text-sm text-muted-foreground">
          <Info className="mt-0.5 size-4 shrink-0 text-primary" />
          <div>
            <p className="font-medium text-foreground">
              Inputs used for this prediction
            </p>
            <p className="mt-1 text-xs leading-5">
              {Object.entries(input)
                .filter(([, v]) => v !== null && v !== undefined && v !== "")
                .map(([k, v]) => `${titleCase(k)}: ${v}`)
                .join(" · ")}
            </p>
            <p className="mt-2 flex items-center gap-1.5 text-xs">
              <ShieldCheck className="size-3.5" />
              {result.disclaimer}
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
