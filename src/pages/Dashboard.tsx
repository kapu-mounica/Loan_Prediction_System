import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { api } from "@/convex/_generated/api";
import { useAuth } from "@/hooks/use-auth";
import {
  formatDate,
  formatPercent,
  RISK_BADGE,
  riskTone,
  type HealthInfo,
  type ModelInfo,
} from "@/lib/loan-api";
import { useAction, useQuery } from "convex/react";
import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  ClipboardList,
  Cpu,
  Database,
  Gauge,
  LineChart,
  Loader2,
  RefreshCw,
  Scale,
  ShieldCheck,
  Sparkles,
  SplitSquareVertical,
} from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";
import { toast } from "sonner";

const METRIC_CARDS = [
  { key: "accuracy", label: "Test accuracy", icon: Activity },
  { key: "rocAuc", label: "ROC-AUC", icon: Gauge },
  { key: "f1", label: "F1 score", icon: SplitSquareVertical },
  { key: "precision", label: "Precision", icon: ShieldCheck },
  { key: "recall", label: "Recall", icon: Cpu },
] as const;

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
};

export default function Dashboard() {
  const { user } = useAuth();
  const modelInfo = useQuery(api.loan.api.modelInfo) as ModelInfo | undefined;
  const health = useQuery(api.loan.api.health) as HealthInfo | undefined;
  const retrain = useAction(api.loan.api.retrain);

  const [retraining, setRetraining] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [reportOpen, setReportOpen] = useState(false);

  const handleRetrain = async () => {
    setRetraining(true);
    try {
      const res = await retrain();
      setReport(res.report);
      setReportOpen(true);
      toast.success("Model retrained", {
        description: "The production model was re-trained and re-evaluated.",
      });
    } catch (err) {
      toast.error("Retraining failed", {
        description: err instanceof Error ? err.message : "Unknown error",
      });
    } finally {
      setRetraining(false);
    }
  };

  const topFeatures = modelInfo?.featureImportance.slice(0, 5) ?? [];
  const dist = modelInfo?.riskDistribution;
  const distTotal = dist?.total ?? 0;
  const distItems = dist
    ? [
        { name: "Low Risk", value: dist.low, tone: "low" as const },
        { name: "Medium Risk", value: dist.medium, tone: "medium" as const },
        { name: "High Risk", value: dist.high, tone: "high" as const },
      ]
    : [];

  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wider text-primary">
            Model Overview
          </p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">
            Welcome{user?.name ? `, ${user.name}` : ""}
          </h1>
          <p className="mt-2 max-w-2xl text-pretty leading-7 text-muted-foreground">
            This is the live production model behind every assessment — the
            same trained artifact the prediction API runs. Everything below is
            real evaluation output, not a mockup.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline" className="gap-2">
            <Link to="/analytics">
              <LineChart className="size-4" />
              Full analytics
            </Link>
          </Button>
          <Button
            type="button"
            onClick={handleRetrain}
            disabled={retraining}
            className="gap-2"
          >
            {retraining ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <RefreshCw className="size-4" />
            )}
            {retraining ? "Retraining…" : "Retrain models"}
          </Button>
        </div>
      </div>

      {/* Status strip */}
      {health && modelInfo && (
        <motion.div initial="hidden" animate="show" variants={fadeUp} className="mt-8">
          <Card className="border-border/80">
            <CardContent className="flex flex-col gap-4 p-5 md:flex-row md:items-center md:justify-between">
              <div className="flex items-center gap-3">
                <span className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <BrainCircuit className="size-5" />
                </span>
                <div>
                  <p className="text-sm font-semibold tracking-tight">
                    {modelInfo.selectedAlgorithm}
                    <Badge variant="secondary" className="ml-2">
                      {health.model.source}
                    </Badge>
                  </p>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    v{modelInfo.version} · trained {formatDate(modelInfo.trainedAt)} ·{" "}
                    {modelInfo.dataset.n.toLocaleString()} applications ·{" "}
                    {modelInfo.split.train}/{modelInfo.split.test} split · selected by{" "}
                    {modelInfo.selectionCriterion}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge className="gap-1.5 bg-primary/10 text-primary hover:bg-primary/15">
                  <Sparkles className="size-3" />
                  {formatPercent(modelInfo.metrics.accuracy)} accuracy
                </Badge>
                <Badge className="gap-1.5 bg-primary/10 text-primary hover:bg-primary/15">
                  <Gauge className="size-3" />
                  {formatPercent(modelInfo.metrics.rocAuc)} ROC-AUC
                </Badge>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      <div className="mt-6 grid gap-6 lg:grid-cols-3">
        {/* Metrics */}
        <motion.div initial="hidden" animate="show" variants={fadeUp} className="lg:col-span-2">
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {METRIC_CARDS.map((m) => (
              <Card key={m.key} className="border-border/80">
                <CardContent className="p-5">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                      {m.label}
                    </p>
                    <m.icon className="size-4 text-primary" />
                  </div>
                  <p className="mt-2 tabular-nums text-3xl font-bold tracking-tight">
                    {modelInfo ? formatPercent(modelInfo.metrics[m.key]) : "—"}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {modelInfo
                      ? `train ${formatPercent(modelInfo.metrics.trainAccuracy ?? 0)} · n=${modelInfo.metrics.n}`
                      : "loading…"}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Top features */}
          <Card className="mt-6 border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">Top predictive features</CardTitle>
              <CardDescription>
                Impurity-based feature importance of the selected model —
                what drives approval decisions most.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {topFeatures.map((f, i) => (
                <div key={f.feature}>
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium capitalize">{f.label}</span>
                    <span className="tabular-nums text-muted-foreground">
                      {formatPercent(f.importance)}
                    </span>
                  </div>
                  <div className="mt-1.5 h-2 w-full overflow-hidden rounded-full bg-primary/20">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${f.importance * 100}%`,
                        background: i === 0 ? "var(--chart-1)" : i === 1 ? "var(--chart-2)" : "var(--chart-3)",
                      }}
                    />
                  </div>
                </div>
              ))}
              {modelInfo && (
                <Button asChild variant="outline" size="sm" className="mt-2 gap-2">
                  <Link to="/explain">
                    How these are computed
                    <ArrowRight className="size-3.5" />
                  </Link>
                </Button>
              )}
            </CardContent>
          </Card>
        </motion.div>

        {/* Right column */}
        <motion.div
          initial="hidden"
          animate="show"
          variants={fadeUp}
          transition={{ delay: 0.08 }}
          className="flex flex-col gap-6"
        >
          {/* Risk distribution */}
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">Test risk distribution</CardTitle>
              <CardDescription>
                Where the held-out applicants fall across risk bands.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {distItems.map((d) => (
                <div key={d.name} className="flex items-center gap-3">
                  <span
                    className={`w-2.5 h-2.5 shrink-0 rounded-full ${
                      d.tone === "low"
                        ? "bg-emerald-500"
                        : d.tone === "medium"
                          ? "bg-amber-500"
                          : "bg-rose-500"
                    }`}
                  />
                  <span className="flex-1 text-sm">{d.name}</span>
                  <span className="tabular-nums text-sm font-medium">{d.value}</span>
                  <span className="w-12 text-right text-xs tabular-nums text-muted-foreground">
                    {distTotal ? formatPercent(d.value / distTotal, 0) : "—"}
                  </span>
                </div>
              ))}
              {modelInfo && (
                <p className="border-t border-border/60 pt-3 text-xs text-muted-foreground">
                  {modelInfo.riskScore.interpretation}
                </p>
              )}
            </CardContent>
          </Card>

          {/* Dataset */}
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="text-lg tracking-tight">Training dataset</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-center gap-3">
                <Database className="size-4 shrink-0 text-primary" />
                <span className="text-muted-foreground">{modelInfo?.dataset.name}</span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg border border-border/70 p-3">
                  <p className="text-xs text-muted-foreground">Approval rate</p>
                  <p className="mt-1 text-lg font-bold tabular-nums tracking-tight">
                    {modelInfo ? formatPercent(modelInfo.dataset.approvalRate, 0) : "—"}
                  </p>
                </div>
                <div className="rounded-lg border border-border/70 p-3">
                  <p className="text-xs text-muted-foreground">Model features</p>
                  <p className="mt-1 text-lg font-bold tabular-nums tracking-tight">
                    {modelInfo?.pipeline.nFeatures ?? "—"}
                  </p>
                </div>
              </div>
              <p className="text-xs leading-5 text-muted-foreground">
                {modelInfo?.dataset.description}
              </p>
            </CardContent>
          </Card>

          {/* Quick actions */}
          <Card className="border-border/80">
            <CardContent className="flex flex-col gap-3 p-5">
              <Button asChild className="w-full justify-start gap-2">
                <Link to="/assess">
                  <ClipboardList className="size-4" />
                  New risk assessment
                </Link>
              </Button>
              <Button asChild variant="outline" className="w-full justify-start gap-2">
                <Link to="/results">
                  <Scale className="size-4" />
                  View latest result
                </Link>
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Retrain report dialog */}
      <Dialog open={reportOpen} onOpenChange={setReportOpen}>
        <DialogContent className="max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <RefreshCw className="size-4 text-primary" />
              Training report
            </DialogTitle>
            <DialogDescription>
              Logistic Regression vs Random Forest — evaluated on the held-out
              test set, winner selected by ROC-AUC.
            </DialogDescription>
          </DialogHeader>
          {report && (
            <pre className="whitespace-pre-wrap rounded-xl border border-border/70 bg-muted/40 p-4 text-xs leading-5 text-foreground">
              {report}
            </pre>
          )}
          <div className="flex flex-wrap gap-2">
            {distItems.map((d) => (
              <Badge key={d.name} className={RISK_BADGE[riskTone(d.name)]}>
                {d.name}
              </Badge>
            ))}
          </div>
          <p className="text-xs text-muted-foreground">
            <span className="font-medium text-foreground">Note:</span>{" "}
            {modelInfo?.disclaimer}
          </p>
          <div className="flex justify-end gap-2">
            <Button asChild variant="outline" className="gap-2">
              <Link to="/analytics">
                <LineChart className="size-4" />
                See the comparison
              </Link>
            </Button>
            <Button onClick={() => setReportOpen(false)}>Close</Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
