import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ConfusionMatrix } from "@/components/risk/ConfusionMatrix";
import { api } from "@/convex/_generated/api";
import {
  formatDate,
  formatPercent,
  type ModelInfo,
} from "@/lib/loan-api";
import { useQuery } from "convex/react";
import { motion } from "framer-motion";
import {
  ArrowRight,
  BarChart3,
  Database,
  GitCompareArrows,
  Loader2,
  PieChart,
  RefreshCw,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart as RePieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const METRICS = ["accuracy", "precision", "recall", "f1", "rocAuc"] as const;
const METRIC_LABELS: Record<(typeof METRICS)[number], string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1 score",
  rocAuc: "ROC-AUC",
};

const RISK_COLORS = ["#10b981", "#f59e0b", "#ef4444"];

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
};

function MetricCard({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div className="rounded-xl border border-border/70 p-4">
      <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p className="mt-1.5 text-xl font-bold tabular-nums tracking-tight">
        {value === undefined ? "—" : formatPercent(value)}
      </p>
    </div>
  );
}

export default function Analytics() {
  const modelInfo = useQuery(api.loan.api.modelInfo) as ModelInfo | undefined;

  if (!modelInfo) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="size-5 animate-spin" />
          Loading model analytics…
        </div>
      </div>
    );
  }

  const [lr, rf] = modelInfo.comparison;
  const selectedIsRf = modelInfo.selectedModel === "random_forest";
  const selected = selectedIsRf ? rf : lr;

  const comparisonData = METRICS.map((m) => ({
    metric: METRIC_LABELS[m],
    "Logistic Regression": lr[m],
    "Random Forest": rf[m],
  }));

  const importanceData = modelInfo.featureImportance
    .slice(0, 10)
    .map((f) => ({ name: f.label, importance: f.importance }))
    .reverse();

  const permData = modelInfo.permutationImportance
    .slice(0, 10)
    .map((f) => ({ name: f.label, importance: f.importance }))
    .reverse();

  const riskData = [
    { name: "Low Risk", value: modelInfo.riskDistribution.low },
    { name: "Medium Risk", value: modelInfo.riskDistribution.medium },
    { name: "High Risk", value: modelInfo.riskDistribution.high },
  ];

  const tooltipStyle = {
    background: "var(--card)",
    border: "1px solid var(--border)",
    borderRadius: 12,
    fontSize: 13,
  };

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6">
      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <p className="text-sm font-semibold uppercase tracking-wider text-primary">
          Model Analytics
        </p>
        <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">
          Evaluation & comparison
        </h1>
        <p className="mt-2 max-w-3xl text-pretty leading-7 text-muted-foreground">
          Both algorithms were trained on the same 80/20 stratified split
          (seed {modelInfo.split.randomState}) and evaluated on the same
          held-out test set of {modelInfo.split.test} applications. The winner
          is selected by {modelInfo.selectionCriterion}, not by assumption.
        </p>
      </motion.div>

      {/* Selected model strip */}
      <motion.div initial="hidden" animate="show" variants={fadeUp} className="mt-6">
        <Card className="border-primary/25 bg-gradient-to-r from-primary/8 via-card to-card">
          <CardContent className="flex flex-col gap-4 p-5 md:flex-row md:items-center md:justify-between">
            <div className="flex items-center gap-3">
              <span className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <Sparkles className="size-5" />
              </span>
              <div>
                <p className="text-sm font-semibold tracking-tight">
                  Selected model: {modelInfo.selectedAlgorithm}
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  v{modelInfo.version} · trained {formatDate(modelInfo.trainedAt)} ·{" "}
                  {modelInfo.dataset.n.toLocaleString()} applications
                </p>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-5">
              <MetricCard label="Accuracy" value={selected.accuracy} />
              <MetricCard label="ROC-AUC" value={selected.rocAuc} />
              <MetricCard label="Precision" value={selected.precision} />
              <MetricCard label="Recall" value={selected.recall} />
              <MetricCard label="F1" value={selected.f1} />
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        {/* Comparison chart */}
        <motion.div initial="hidden" animate="show" variants={fadeUp}>
          <Card className="h-full border-border/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg tracking-tight">
                <GitCompareArrows className="size-4 text-primary" />
                Head-to-head comparison
              </CardTitle>
              <CardDescription>
                Test-set metrics for both models. Higher is better; the selected
                model is highlighted in the table below.
              </CardDescription>
            </CardHeader>
            <CardContent>
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
                      contentStyle={tooltipStyle}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="Logistic Regression" fill="var(--chart-2)" radius={[6, 6, 0, 0]} maxBarSize={28} />
                    <Bar dataKey="Random Forest" fill="var(--chart-1)" radius={[6, 6, 0, 0]} maxBarSize={28} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Comparison table */}
        <motion.div initial="hidden" animate="show" variants={fadeUp} transition={{ delay: 0.06 }}>
          <Card className="h-full border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">Metric table</CardTitle>
              <CardDescription>
                Winner per metric is tinted; the production model is marked.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border/70 text-left text-xs uppercase tracking-wider text-muted-foreground">
                      <th className="py-2 pr-3 font-medium">Metric</th>
                      <th className="py-2 pr-3 font-medium">Logistic Regression</th>
                      <th className="py-2 font-medium">Random Forest</th>
                    </tr>
                  </thead>
                  <tbody>
                    {METRICS.map((m) => {
                      const a = lr[m];
                      const b = rf[m];
                      const winner = a === b ? null : a > b ? "lr" : "rf";
                      return (
                        <tr key={m} className="border-b border-border/50">
                          <td className="py-2.5 pr-3 font-medium">{METRIC_LABELS[m]}</td>
                          <td
                            className={`py-2.5 pr-3 tabular-nums ${
                              winner === "lr" ? "text-emerald-600 dark:text-emerald-400" : ""
                            }`}
                          >
                            {formatPercent(a)}
                            {winner === "lr" && " ✓"}
                          </td>
                          <td
                            className={`py-2.5 tabular-nums ${
                              winner === "rf" ? "text-emerald-600 dark:text-emerald-400" : ""
                            }`}
                          >
                            {formatPercent(b)}
                            {winner === "rf" && " ✓"}
                            {selectedIsRf && m === "rocAuc" && (
                              <Badge variant="secondary" className="ml-2">
                                production
                              </Badge>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                    <tr className="border-b border-border/50">
                      <td className="py-2.5 pr-3 font-medium">Train accuracy</td>
                      <td className="py-2.5 pr-3 tabular-nums">
                        {lr.trainAccuracy !== null ? formatPercent(lr.trainAccuracy) : "—"}
                      </td>
                      <td className="py-2.5 tabular-nums">
                        {rf.trainAccuracy !== null ? formatPercent(rf.trainAccuracy) : "—"}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2.5 pr-3 font-medium">Test samples</td>
                      <td className="py-2.5 pr-3 tabular-nums text-muted-foreground">{lr.n}</td>
                      <td className="py-2.5 tabular-nums text-muted-foreground">{rf.n}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Badge className="gap-1.5 bg-primary/10 text-primary hover:bg-primary/15">
                  <Sparkles className="size-3" />
                  Selected by ROC-AUC
                </Badge>
                <span className="text-xs text-muted-foreground">
                  ROC-AUC measures how well the model ranks risk across all
                  thresholds — the right criterion for a risk-scoring system.
                </span>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Confusion matrix */}
        <motion.div initial="hidden" animate="show" variants={fadeUp}>
          <Card className="h-full border-border/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg tracking-tight">
                <BarChart3 className="size-4 text-primary" />
                Confusion matrix — {modelInfo.selectedAlgorithm}
              </CardTitle>
              <CardDescription>
                Held-out test predictions. Rows are actual, columns are
                predicted, classes [Rejected, Approved].
              </CardDescription>
            </CardHeader>
            <CardContent className="flex items-center justify-center">
              <div className="w-full max-w-sm">
                <ConfusionMatrix matrix={selected.confusionMatrix} />
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Risk distribution */}
        <motion.div initial="hidden" animate="show" variants={fadeUp} transition={{ delay: 0.06 }}>
          <Card className="h-full border-border/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg tracking-tight">
                <PieChart className="size-4 text-primary" />
                Test risk distribution
              </CardTitle>
              <CardDescription>
                Where the held-out test applicants fall across the three risk
                bands used by the API.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <RePieChart>
                    <Pie
                      data={riskData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius="55%"
                      outerRadius="85%"
                      paddingAngle={3}
                      stroke="var(--card)"
                    >
                      {riskData.map((entry, i) => (
                        <Cell key={entry.name} fill={RISK_COLORS[i]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                  </RePieChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-2 text-center text-xs text-muted-foreground">
                {modelInfo.riskScore.interpretation}
              </p>
            </CardContent>
          </Card>
        </motion.div>

        {/* Feature importance */}
        <motion.div initial="hidden" animate="show" variants={fadeUp}>
          <Card className="h-full border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">Feature importance</CardTitle>
              <CardDescription>
                Impurity-based importance of the selected model (top 10).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={importanceData} layout="vertical" margin={{ left: 8, right: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" domain={[0, "dataMax"]} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={140}
                      tick={{ fontSize: 12 }}
                      stroke="var(--muted-foreground)"
                    />
                    <Tooltip
                      formatter={(value) => formatPercent(Number(value))}
                      contentStyle={tooltipStyle}
                    />
                    <Bar dataKey="importance" fill="var(--chart-1)" radius={[0, 6, 6, 0]} maxBarSize={18} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Permutation importance */}
        <motion.div initial="hidden" animate="show" variants={fadeUp} transition={{ delay: 0.06 }}>
          <Card className="h-full border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">Permutation importance</CardTitle>
              <CardDescription>
                Drop in ROC-AUC when each feature&apos;s test values are
                shuffled — an unbiased, complementary view.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={permData} layout="vertical" margin={{ left: 8, right: 16 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                    <XAxis type="number" tick={{ fontSize: 12 }} stroke="var(--muted-foreground)" domain={[0, "dataMax"]} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={140}
                      tick={{ fontSize: 12 }}
                      stroke="var(--muted-foreground)"
                    />
                    <Tooltip
                      formatter={(value) => formatPercent(Number(value))}
                      contentStyle={tooltipStyle}
                    />
                    <Bar dataKey="importance" fill="var(--chart-2)" radius={[0, 6, 6, 0]} maxBarSize={18} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Pipeline + dataset */}
      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">Preprocessing pipeline</CardTitle>
              <CardDescription>
                Identical preprocessing runs during training and at prediction
                time — the pipeline used to make live predictions is exactly
                the one the model was trained on.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <p className="leading-6 text-muted-foreground">
                {modelInfo.pipeline.missingValueHandling}
              </p>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Numeric ({modelInfo.pipeline.numeric.length})
                  </p>
                  <p className="mt-1 text-xs leading-5">
                    {modelInfo.pipeline.numeric.join(", ")}
                  </p>
                </div>
                <div>
                  <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Categorical ({modelInfo.pipeline.categorical.length})
                  </p>
                  <p className="mt-1 text-xs leading-5">
                    {modelInfo.pipeline.categorical.join(", ")}
                  </p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                {modelInfo.pipeline.nFeatures} encoded model features in total.
              </p>
            </CardContent>
          </Card>

          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg tracking-tight">
                <Database className="size-4 text-primary" />
                Dataset
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div>
                <p className="font-semibold tracking-tight">{modelInfo.dataset.name}</p>
                <p className="mt-1 text-xs leading-5 text-muted-foreground">
                  {modelInfo.dataset.description}
                </p>
              </div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {[
                  { label: "Applications", value: modelInfo.dataset.n.toLocaleString() },
                  { label: "Approval rate", value: formatPercent(modelInfo.dataset.approvalRate, 0) },
                  { label: "Train / test", value: `${modelInfo.split.train}/${modelInfo.split.test}` },
                  { label: "Random state", value: String(modelInfo.split.randomState) },
                ].map((s) => (
                  <div key={s.label} className="rounded-lg border border-border/70 p-3">
                    <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                      {s.label}
                    </p>
                    <p className="mt-1 text-base font-bold tabular-nums tracking-tight">{s.value}</p>
                  </div>
                ))}
              </div>
              <div>
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Missing values in training data
                </p>
                <ul className="mt-1 list-inside list-disc space-y-0.5 text-xs text-muted-foreground">
                  {modelInfo.dataset.missingness.map((m) => (
                    <li key={m}>{m}</li>
                  ))}
                </ul>
              </div>
              <div className="flex flex-wrap gap-2 pt-1">
                <Button asChild variant="outline" size="sm" className="gap-2">
                  <Link to="/explain">
                    How the models work
                    <ArrowRight className="size-3.5" />
                  </Link>
                </Button>
                <Button asChild variant="outline" size="sm" className="gap-2">
                  <Link to="/dashboard">
                    <RefreshCw className="size-3.5" />
                    Retrain from dashboard
                  </Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </motion.div>
    </div>
  );
}
