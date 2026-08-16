import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { api } from "@/convex/_generated/api";
import { saveAssessment } from "@/lib/loan-api";
import type { ApplicantInput } from "@/ml/types";
import { zodResolver } from "@hookform/resolvers/zod";
import { useAction } from "convex/react";
import { motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Loader2,
  Lock,
  ShieldCheck,
  User,
  Wallet,
  Home,
} from "lucide-react";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { useNavigate } from "react-router";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Validation schemas (mirror the API's rules in src/convex/loan/validate.ts)
// ---------------------------------------------------------------------------

const requiredNumber = (label: string, min: number, max: number, minMsg?: string, integer = false) =>
  z.preprocess(
    (v) => (typeof v === "string" && v.trim() === "" ? undefined : v),
    z.coerce
      .number()
      .refine((v) => Number.isFinite(v), `${label} is required.`)
      .refine((v) => v >= min, minMsg ?? `${label} must be at least ${min}.`)
      .refine((v) => v <= max, `${label} must be at most ${max}.`)
      .refine((v) => !integer || Number.isInteger(v), `${label} must be a whole number.`)
  );

const optionalNumber = (label: string, min: number, max: number) =>
  z.preprocess(
    (v) => (v === "" || v === null || v === undefined ? undefined : v),
    z.coerce
      .number()
      .refine((v) => !Number.isFinite(v) || v >= min, `${label} cannot be negative.`)
      .refine((v) => !Number.isFinite(v) || v <= max, `${label} must be at most ${max}.`)
      .optional()
  );

const s1Schema = z.object({
  applicant_age: requiredNumber("Age", 18, 90, "Age must be at least 18.", true),
  marital_status: z.enum(["Married", "Single"], { error: "Select a marital status." }),
  dependents: z.enum(["0", "1", "2", "3", "4"], { error: "Select the number of dependents." }),
  education: z.enum(["Graduate", "Not Graduate"], { error: "Select the education level." }),
  employment_status: z.enum(["Employed", "Unemployed"], { error: "Select the employment status." }),
  self_employed: z.enum(["Yes", "No"], { error: "Select an option." }),
});

const s2Schema = z.object({
  applicant_income: requiredNumber("Applicant income", 0, 1e8, "Income cannot be negative."),
  coapplicant_income: requiredNumber("Co-applicant income", 0, 1e8, "Income cannot be negative."),
  monthly_expenses: requiredNumber("Monthly expenses", 0, 1e8, "Expenses cannot be negative."),
  savings: requiredNumber("Savings", 0, 1e8, "Savings cannot be negative."),
  existing_loans: requiredNumber("Existing loans", 0, 20, "Existing loans cannot be negative.", true),
  debt_to_income_ratio: optionalNumber("Debt-to-income ratio", 0, 5),
});

const s3Schema = z.object({
  loan_amount: requiredNumber("Loan amount", 1, 1e8, "Loan amount must be greater than zero."),
  loan_term: z.enum(["12", "24", "36", "48", "60", "84", "120", "180", "240", "360"], {
    error: "Select a loan term.",
  }),
  credit_history: z.enum(["1", "0"], { error: "Select your credit history." }),
  property_area: z.enum(["Urban", "Semiurban", "Rural"], { error: "Select the property area." }),
});

type FormValues = z.infer<typeof s1Schema> &
  z.infer<typeof s2Schema> &
  z.infer<typeof s3Schema>;

const fullSchema = s1Schema.merge(s2Schema).merge(s3Schema);

const TERM_OPTIONS = ["12", "24", "36", "48", "60", "84", "120", "180", "240", "360"];
const DEFAULT_TERM = "360";

const SECTIONS = [
  {
    id: 0,
    label: "Applicant",
    title: "Applicant Information",
    description: "Who is applying for the loan?",
    icon: User,
    fields: ["applicant_age", "marital_status", "dependents", "education", "employment_status", "self_employed"],
  },
  {
    id: 1,
    label: "Financial",
    title: "Financial Information",
    description: "Income, expenses, savings, and existing debt.",
    icon: Wallet,
    fields: ["applicant_income", "coapplicant_income", "monthly_expenses", "savings", "existing_loans", "debt_to_income_ratio"],
  },
  {
    id: 2,
    label: "Loan",
    title: "Loan Information",
    description: "The amount you want to borrow and your repayment terms.",
    icon: Home,
    fields: ["loan_amount", "loan_term", "credit_history", "property_area"],
  },
] as const;

function SelectField({
  name,
  label,
  control,
  placeholder,
  options,
  description,
}: {
  name: keyof FormValues;
  label: string;
  control: ReturnType<typeof useForm<FormValues>>["control"];
  placeholder: string;
  options: { value: string; label: string }[];
  description?: string;
}) {
  return (
    <FormField
      control={control}
      name={name as never}
      render={({ field }) => (
        <FormItem>
          <FormLabel>{label}</FormLabel>
          <FormControl>
            <Select value={field.value} onValueChange={field.onChange}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder={placeholder} />
              </SelectTrigger>
              <SelectContent>
                {options.map((o) => (
                  <SelectItem key={o.value} value={o.value}>
                    {o.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FormControl>
          {description && <FormDescription>{description}</FormDescription>}
          <FormMessage />
        </FormItem>
      )}
    />
  );
}

export default function Assess() {
  const navigate = useNavigate();
  const predict = useAction(api.loan.api.predict);
  const [step, setStep] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(fullSchema),
    mode: "onTouched",
    defaultValues: {
      applicant_age: undefined as never,
      marital_status: "" as never,
      dependents: "" as never,
      education: "" as never,
      employment_status: "" as never,
      self_employed: "" as never,
      applicant_income: undefined as never,
      coapplicant_income: undefined as never,
      monthly_expenses: undefined as never,
      savings: undefined as never,
      existing_loans: undefined as never,
      debt_to_income_ratio: "",
      loan_amount: undefined as never,
      loan_term: DEFAULT_TERM,
      credit_history: "" as never,
      property_area: "" as never,
    },
  });

  const section = SECTIONS[step];

  const goNext = async () => {
    setApiError(null);
    const valid = await form.trigger(section.fields as never);
    if (valid) setStep((s) => Math.min(s + 1, 2));
  };

  const goBack = () => {
    setApiError(null);
    setStep((s) => Math.max(s - 1, 0));
  };

  const onSubmit = async (values: FormValues) => {
    setSubmitting(true);
    setApiError(null);
    try {
      const input: ApplicantInput = {
        applicant_age: values.applicant_age,
        applicant_income: values.applicant_income,
        coapplicant_income: values.coapplicant_income,
        loan_amount: values.loan_amount,
        loan_term: Number(values.loan_term),
        credit_history: Number(values.credit_history),
        employment_status: values.employment_status,
        education: values.education,
        marital_status: values.marital_status,
        dependents: Number(values.dependents),
        property_area: values.property_area,
        self_employed: values.self_employed,
        existing_loans: values.existing_loans,
        monthly_expenses: values.monthly_expenses,
        savings: values.savings,
        debt_to_income_ratio:
          values.debt_to_income_ratio === ""
            ? values.applicant_income > 0
              ? Math.round((values.monthly_expenses / values.applicant_income) * 100) / 100
              : null
            : values.debt_to_income_ratio,
      };
      const result = await predict({ input });
      saveAssessment(input, result);
      navigate("/results");
    } catch (err) {
      setApiError(
        err instanceof Error
          ? err.message
          : "Prediction failed. Please check the values and try again."
      );
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-12 sm:px-6">
      <div className="flex flex-col gap-1">
        <p className="text-sm font-semibold uppercase tracking-wider text-primary">
          Risk Assessment
        </p>
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">
          Loan Application
        </h1>
        <p className="mt-2 max-w-2xl text-pretty leading-7 text-muted-foreground">
          Enter the applicant&apos;s details below. The model evaluates the
          application with the exact same preprocessing pipeline used in
          training.
        </p>
      </div>

      {/* Privacy note */}
      <div className="mt-6 flex items-start gap-2.5 rounded-xl border border-border/80 bg-card p-4 text-sm text-muted-foreground">
        <Lock className="mt-0.5 size-4 shrink-0 text-primary" />
        <p>
          <span className="font-medium text-foreground">Privacy:</span> your
          details are sent to the prediction API for a single evaluation and are
          never stored. The result stays in this browser tab only.
        </p>
      </div>

      {/* Stepper */}
      <div className="mt-8 flex items-center gap-2" aria-label="Form progress">
        {SECTIONS.map((s, i) => (
          <div key={s.id} className="flex flex-1 items-center gap-2">
            <div
              className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full border text-xs font-semibold transition-colors ${
                i < step
                  ? "border-primary bg-primary text-primary-foreground"
                  : i === step
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-border bg-card text-muted-foreground"
              }`}
            >
              {i < step ? <Check className="size-4" /> : s.id + 1}
            </div>
            <div className="hidden min-w-0 sm:block">
              <p
                className={`truncate text-sm font-medium ${
                  i === step ? "text-foreground" : "text-muted-foreground"
                }`}
              >
                {s.label}
              </p>
            </div>
            {i < SECTIONS.length - 1 && (
              <div
                className={`h-px flex-1 ${
                  i < step ? "bg-primary" : "bg-border"
                }`}
              />
            )}
          </div>
        ))}
      </div>

      <motion.div
        key={step}
        initial={{ opacity: 0, x: 16 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.25 }}
        className="mt-6"
      >
        <Card className="border-border/80 shadow-lg shadow-primary/5">
          <CardHeader>
            <div className="flex items-center gap-3">
              <span className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <section.icon className="size-5" />
              </span>
              <div>
                <CardTitle className="text-xl">{section.title}</CardTitle>
                <CardDescription className="mt-1">
                  {section.description}
                </CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form
                onSubmit={form.handleSubmit(onSubmit)}
                className="flex flex-col gap-8"
                noValidate
              >
                {step === 0 && (
                  <div className="grid gap-6 sm:grid-cols-2">
                    <FormField
                      control={form.control}
                      name="applicant_age"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Applicant age</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              inputMode="numeric"
                              min={18}
                              max={90}
                              placeholder="e.g. 32"
                              {...field}
                              onChange={(e) => field.onChange(e.target.valueAsNumber)}
                            />
                          </FormControl>
                          <FormDescription>18–90 years.</FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <SelectField
                      name="marital_status"
                      label="Marital status"
                      control={form.control}
                      placeholder="Select…"
                      options={[
                        { value: "Married", label: "Married" },
                        { value: "Single", label: "Single" },
                      ]}
                    />
                    <SelectField
                      name="dependents"
                      label="Dependents"
                      control={form.control}
                      placeholder="Select…"
                      options={[
                        { value: "0", label: "0" },
                        { value: "1", label: "1" },
                        { value: "2", label: "2" },
                        { value: "3", label: "3" },
                        { value: "4", label: "4" },
                      ]}
                    />
                    <SelectField
                      name="education"
                      label="Education"
                      control={form.control}
                      placeholder="Select…"
                      options={[
                        { value: "Graduate", label: "Graduate" },
                        { value: "Not Graduate", label: "Not Graduate" },
                      ]}
                    />
                    <SelectField
                      name="employment_status"
                      label="Employment status"
                      control={form.control}
                      placeholder="Select…"
                      options={[
                        { value: "Employed", label: "Employed" },
                        { value: "Unemployed", label: "Unemployed" },
                      ]}
                    />
                    <SelectField
                      name="self_employed"
                      label="Self employed"
                      control={form.control}
                      placeholder="Select…"
                      options={[
                        { value: "Yes", label: "Yes" },
                        { value: "No", label: "No" },
                      ]}
                    />
                  </div>
                )}

                {step === 1 && (
                  <div className="grid gap-6 sm:grid-cols-2">
                    <FormField
                      control={form.control}
                      name="applicant_income"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Applicant income (₹/year)</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              inputMode="numeric"
                              min={0}
                              step={1000}
                              placeholder="e.g. 50000"
                              {...field}
                              onChange={(e) => field.onChange(e.target.valueAsNumber)}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="coapplicant_income"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Co-applicant income (₹/year)</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              inputMode="numeric"
                              min={0}
                              step={1000}
                              placeholder="e.g. 20000"
                              {...field}
                              onChange={(e) => field.onChange(e.target.valueAsNumber)}
                            />
                          </FormControl>
                          <FormDescription>0 if none.</FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="monthly_expenses"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Monthly expenses (₹/month)</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              inputMode="numeric"
                              min={0}
                              step={500}
                              placeholder="e.g. 20000"
                              {...field}
                              onChange={(e) => field.onChange(e.target.valueAsNumber)}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="savings"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Savings (₹)</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              inputMode="numeric"
                              min={0}
                              step={1000}
                              placeholder="e.g. 100000"
                              {...field}
                              onChange={(e) => field.onChange(e.target.valueAsNumber)}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="existing_loans"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Existing loans</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              inputMode="numeric"
                              min={0}
                              max={20}
                              step={1}
                              placeholder="e.g. 1"
                              {...field}
                              onChange={(e) => field.onChange(e.target.valueAsNumber)}
                            />
                          </FormControl>
                          <FormDescription>Active loans you currently owe.</FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <FormField
                      control={form.control}
                      name="debt_to_income_ratio"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Debt-to-income ratio</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              inputMode="decimal"
                              min={0}
                              max={5}
                              step={0.01}
                              placeholder="Auto-calculated"
                              {...field}
                              onChange={(e) =>
                                field.onChange(
                                  e.target.value === "" ? "" : e.target.valueAsNumber
                                )
                              }
                            />
                          </FormControl>
                          <FormDescription>
                            Leave blank to calculate from expenses ÷ income.
                          </FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                )}

                {step === 2 && (
                  <div className="grid gap-6 sm:grid-cols-2">
                    <FormField
                      control={form.control}
                      name="loan_amount"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Loan amount (₹)</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              inputMode="numeric"
                              min={1000}
                              step={1000}
                              placeholder="e.g. 150000"
                              {...field}
                              onChange={(e) => field.onChange(e.target.valueAsNumber)}
                            />
                          </FormControl>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                    <SelectField
                      name="loan_term"
                      label="Loan term (months)"
                      control={form.control}
                      placeholder="Select…"
                      options={TERM_OPTIONS.map((t) => ({
                        value: t,
                        label: `${t} months${t === "360" ? " (30 yrs)" : ""}`,
                      }))}
                    />
                    <SelectField
                      name="credit_history"
                      label="Credit history"
                      control={form.control}
                      placeholder="Select…"
                      description="Have you repaid past debts on time?"
                      options={[
                        { value: "1", label: "Yes — good credit history" },
                        { value: "0", label: "No — no or troubled credit history" },
                      ]}
                    />
                    <SelectField
                      name="property_area"
                      label="Property area"
                      control={form.control}
                      placeholder="Select…"
                      options={[
                        { value: "Urban", label: "Urban" },
                        { value: "Semiurban", label: "Semiurban" },
                        { value: "Rural", label: "Rural" },
                      ]}
                    />
                  </div>
                )}

                {apiError && (
                  <div
                    role="alert"
                    className="rounded-xl border border-destructive/40 bg-destructive/8 px-4 py-3 text-sm text-destructive"
                  >
                    {apiError}
                  </div>
                )}

                <div className="flex items-center justify-between border-t border-border/70 pt-6">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={goBack}
                    disabled={step === 0 || submitting}
                    className="gap-2"
                  >
                    <ArrowLeft className="size-4" />
                    Back
                  </Button>
                  {step < 2 ? (
                    <Button type="button" onClick={goNext} className="gap-2">
                      Continue
                      <ArrowRight className="size-4" />
                    </Button>
                  ) : (
                    <Button
                      type="submit"
                      disabled={submitting}
                      className="gap-2"
                    >
                      {submitting ? (
                        <>
                          <Loader2 className="size-4 animate-spin" />
                          Running model…
                        </>
                      ) : (
                        <>
                          <ShieldCheck className="size-4" />
                          Get prediction
                        </>
                      )}
                    </Button>
                  )}
                </div>
              </form>
            </Form>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
