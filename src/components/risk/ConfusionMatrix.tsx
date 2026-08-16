import { cn } from "@/lib/utils";

interface ConfusionMatrixProps {
  /** [TN, FP, FN, TP] for classes [Rejected, Approved]. */
  matrix: [number, number, number, number];
  labels?: [string, string];
  className?: string;
}

/**
 * 2×2 confusion matrix heatmap. Matrix order is [TN, FP, FN, TP]
 * matching scikit-learn's convention with classes [Rejected, Approved].
 */
export function ConfusionMatrix({ matrix, labels = ["Rejected", "Approved"], className }: ConfusionMatrixProps) {
  const [tn, fp, fn, tp] = matrix;
  const max = Math.max(tn, fp, fn, tp, 1);

  const cells = [
    { value: tn, label: "True Rejected", correct: true, row: 0, col: 0 },
    { value: fp, label: "False Approved", correct: false, row: 0, col: 1 },
    { value: fn, label: "False Rejected", correct: false, row: 1, col: 0 },
    { value: tp, label: "True Approved", correct: true, row: 1, col: 1 },
  ];

  return (
    <div className={cn("w-full", className)}>
      <div className="mb-2 grid grid-cols-[auto_1fr_1fr] items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
        <span />
        <span className="text-center font-medium">{labels[0]}</span>
        <span className="text-center font-medium">{labels[1]}</span>
        <span className="mr-1 text-right font-medium">Predicted</span>
        <span className="col-span-2 text-[11px]">Predicted class (columns)</span>
        <span className="mr-1 text-right font-medium">Actual</span>
        <span className="col-span-2 text-[11px]">Actual class (rows)</span>
      </div>

      <div className="grid grid-cols-[auto_1fr_1fr] gap-1.5">
        {[
          { rowLabel: labels[0], cells: [cells[0], cells[1]] },
          { rowLabel: labels[1], cells: [cells[2], cells[3]] },
        ].map((row) => (
          <div key={row.rowLabel} className="contents">
            <div className="flex items-center justify-end pr-2 text-xs font-medium text-muted-foreground">
              {row.rowLabel}
            </div>
            {row.cells.map((cell) => (
              <div
                key={cell.label}
                className={cn(
                  "relative flex flex-col items-center justify-center rounded-lg px-3 py-4",
                  cell.correct
                    ? "bg-emerald-500/12 text-emerald-900 dark:bg-emerald-500/15 dark:text-emerald-200"
                    : "bg-rose-500/12 text-rose-900 dark:bg-rose-500/15 dark:text-rose-200"
                )}
                style={{
                  boxShadow: `inset 0 0 0 1px color-mix(in oklab, ${cell.correct ? "var(--chart-1)" : "var(--destructive)"} 25%, transparent)`,
                }}
              >
                <span className="tabular-nums text-xl font-semibold">{cell.value}</span>
                <span className="mt-0.5 text-[10px] uppercase tracking-wide opacity-70">
                  {cell.label}
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>

      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span>Accuracy = (TN + TP) / total</span>
        <span>TN {tn} · FP {fp} · FN {fn} · TP {tp}</span>
      </div>
      <div className="sr-only">{`Confusion matrix: true negatives ${tn}, false positives ${fp}, false negatives ${fn}, true positives ${tp}.`}</div>
    </div>
  );
}
