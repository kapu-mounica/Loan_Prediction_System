import { RISK_COLORS, riskTone } from "@/lib/loan-api";
import { cn } from "@/lib/utils";

interface RiskGaugeProps {
  score: number; // 0–100
  category: string;
  size?: "sm" | "md" | "lg";
  className?: string;
}

/**
 * Interpretable 0–100 risk score gauge.
 * 0–30 low · 31–60 medium · 61–100 high (same boundaries as the model).
 */
export function RiskGauge({ score, category, size = "md", className }: RiskGaugeProps) {
  const tone = riskTone(category);
  const color = RISK_COLORS[tone];
  const clamped = Math.min(100, Math.max(0, score));
  const sizePx = size === "lg" ? 220 : size === "sm" ? 140 : 180;

  // Semicircle gauge: start at 180°, sweep 180° clockwise.
  const sweep = 180;
  const angle = (clamped / 100) * sweep;
  const rad = (180 - angle) * (Math.PI / 180);
  const cx = 110;
  const cy = 110;
  const r = 86;
  const endX = cx + r * Math.cos(rad);
  const endY = cy - r * Math.sin(rad);

  const path = describeArc(cx, cy, r, 180, 180 - angle);

  return (
    <div
      className={cn("relative inline-flex items-center justify-center", className)}
      style={{ width: sizePx, height: sizePx * 0.62 }}
      role="img"
      aria-label={`Risk score ${clamped} out of 100 — ${category}`}
    >
      <svg viewBox="0 0 220 140" className="h-full w-full">
        {/* track */}
        <path
          d={describeArc(cx, cy, r, 180, 0)}
          fill="none"
          stroke="var(--border)"
          strokeWidth={14}
          strokeLinecap="round"
        />
        {/* value arc */}
        <path
          d={path}
          fill="none"
          stroke={color}
          strokeWidth={14}
          strokeLinecap="round"
          style={{ transition: "stroke-dasharray 0.6s ease" }}
        />
        {/* needle */}
        <line
          x1={cx}
          y1={cy}
          x2={endX}
          y2={endY}
          stroke={color}
          strokeWidth={3}
          strokeLinecap="round"
          style={{ transition: "all 0.6s ease" }}
        />
        <circle cx={cx} cy={cy} r={5} fill={color} />
      </svg>
      <div className="absolute inset-x-0 bottom-0 flex flex-col items-center leading-none">
        <span
          className="tabular-nums text-2xl font-semibold tracking-tight"
          style={{ color }}
        >
          {clamped}
        </span>
        <span className="mt-0.5 text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
          / 100
        </span>
      </div>
    </div>
  );
}

function describeArc(cx: number, cy: number, r: number, startDeg: number, endDeg: number) {
  const polar = (deg: number) => {
    const rad = deg * (Math.PI / 180);
    return { x: cx + r * Math.cos(rad), y: cy - r * Math.sin(rad) };
  };
  const s = polar(startDeg);
  const e = polar(endDeg);
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${largeArc} 0 ${e.x} ${e.y}`;
}
