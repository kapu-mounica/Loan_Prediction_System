import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Eye,
  HandHeart,
  Lock,
  Scale,
  ShieldCheck,
  UserCheck,
  Users,
} from "lucide-react";
import { Link } from "react-router";

const PRINCIPLES = [
  {
    icon: Users,
    title: "No protected characteristics as features",
    body: "The model never sees gender, caste, religion, or any other protected attribute. Only financial and application features are used — income, loan size, credit history, savings, debt, employment, education, and similar.",
  },
  {
    icon: Lock,
    title: "No applicant data is stored",
    body: "Assessment inputs are sent to the prediction API for a single evaluation and are never persisted. Results live only in the browser tab's session storage and disappear when the tab closes.",
  },
  {
    icon: Eye,
    title: "Transparency by design",
    body: "Every prediction is accompanied by its probability, an interpretable 0–100 risk score, and per-feature attributions showing what moved the decision. The full training pipeline, metrics, and model selection are exposed in the analytics pages.",
  },
  {
    icon: HandHeart,
    title: "Human review always required",
    body: "This is a decision-support tool, not a decision maker. Any real-world lending decision must be reviewed by a qualified human underwriter using complete, verified information.",
  },
  {
    icon: Scale,
    title: "Honest about limitations",
    body: "Predictions are ML estimates, not guarantees. The models were trained on a simulated, education-oriented dataset; real-world performance on a different population will differ until the models are retrained on real, representative data.",
  },
  {
    icon: UserCheck,
    title: "Educational purpose only",
    body: "The system exists for learning, portfolios, and placement demonstrations. It does not constitute financial, legal, or lending advice, and must never be used to deny credit to a real person.",
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: "easeOut" as const } },
};

export default function ResponsibleAI() {
  return (
    <div className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6">
      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <Badge variant="secondary" className="gap-1.5">
          <ShieldCheck className="size-3.5 text-primary" />
          Responsible AI
        </Badge>
        <h1 className="mt-4 text-balance text-3xl font-bold tracking-tight sm:text-4xl">
          Built responsibly, by design
        </h1>
        <p className="mt-3 max-w-2xl text-pretty leading-7 text-muted-foreground">
          Machine learning on financial data has real consequences. This
          project applies the same care to fairness, privacy, and honesty that
          a production lending system should — while staying clearly framed as
          an educational tool.
        </p>
      </motion.div>

      <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {PRINCIPLES.map((p, i) => (
          <motion.div
            key={p.title}
            initial={{ opacity: 0, y: 16 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-60px" }}
            transition={{ duration: 0.4, delay: i * 0.05 }}
          >
            <Card className="h-full border-border/80">
              <CardContent className="flex h-full flex-col gap-4 p-6">
                <span className="flex size-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <p.icon className="size-5" />
                </span>
                <h3 className="text-lg font-semibold tracking-tight">{p.title}</h3>
                <p className="text-sm leading-6 text-muted-foreground">{p.body}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      <motion.div initial="hidden" animate="show" variants={fadeUp}>
        <Card className="mt-8 border-primary/25 bg-gradient-to-br from-primary/8 via-card to-card">
          <CardHeader>
            <CardTitle className="text-lg tracking-tight">The disclaimer that ships with every prediction</CardTitle>
            <CardDescription>
              Returned by the API and shown on every result.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <blockquote className="rounded-xl border border-border/70 bg-background/60 p-5 text-sm leading-6 text-muted-foreground">
              “This tool is an educational decision-support system. Predictions
              are machine learning estimates, not guarantees, and do not
              constitute financial, legal, or lending advice. A human review is
              always required before any real lending decision.”
            </blockquote>
            <div className="mt-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <p className="max-w-xl text-sm text-muted-foreground">
                Want to dig into the methodology, dataset, and API? The About
                page documents the whole system.
              </p>
              <Button asChild className="shrink-0 gap-2">
                <Link to="/about">
                  Read the documentation
                  <ArrowRight className="size-4" />
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
