"use client";

import React, { useState } from "react";

export function cx(...xs: Array<string | false | null | undefined>) {
  return xs.filter(Boolean).join(" ");
}

export function Tooltip({
  children,
  content,
  position = "top",
}: {
  children: React.ReactNode;
  content: string;
  position?: "top" | "bottom" | "right";
}) {
  const [show, setShow] = useState(false);
  return (
    <div className="relative inline-flex">
      <div onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}>
        {children}
      </div>
      {show && (
        <div
          className={cx(
            "absolute z-50 px-3 py-2 text-xs bg-white border border-slate-200 rounded-lg text-slate-700 whitespace-nowrap shadow-lg",
            position === "right" ? "left-full ml-2 top-0" : position === "bottom" ? "top-full mt-2 left-1/2 -translate-x-1/2" : "bottom-full mb-2 left-1/2 -translate-x-1/2"
          )}
        >
          {content}
          <div className={cx("absolute w-2 h-2 bg-white border border-slate-200", position === "right" ? "-left-1 top-2" : position === "bottom" ? "bottom-full -translate-x-1/2 left-1/2 border-t-0 border-l-0" : "top-full -translate-x-1/2 left-1/2 border-b-0 border-l-0")} />
        </div>
      )}
    </div>
  );
}

export function Tabs({
  tabs,
  activeTab,
  onTabChange,
}: {
  tabs: { label: string; value: string; icon?: string }[];
  activeTab: string;
  onTabChange: (value: string) => void;
}) {
  return (
    <div className="flex gap-2 rounded-xl border border-slate-700/50 bg-slate-900/60 px-2 py-1.5">
      {tabs.map((tab) => (
        <button
          key={tab.value}
          onClick={() => onTabChange(tab.value)}
          className={cx(
            "flex items-center gap-1 rounded-lg px-4 py-2.5 text-sm font-medium transition whitespace-nowrap",
            activeTab === tab.value
              ? "bg-slate-100/15 text-slate-100"
              : "text-slate-300 hover:bg-slate-800/60 hover:text-slate-100"
          )}
        >
          {tab.icon && <span className="text-base">{tab.icon}</span>}
          {tab.label}
        </button>
      ))}
    </div>
  );
}

export function ProgressIndicator({
  steps,
  currentStep,
  estimatedTime,
}: {
  steps: { label: string; status: "pending" | "active" | "complete" | "error" }[];
  currentStep: number;
  estimatedTime?: string;
}) {
  return (
    <div className="space-y-3">
      <div className="sr-only">Current step {currentStep}</div>
      {steps.map((step, idx) => (
        <div key={idx} className="flex items-center gap-3">
          <div
            className={cx(
              "w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold",
              step.status === "complete"
                ? "bg-emerald-400/20 text-emerald-300 border border-emerald-400/40"
                : step.status === "error"
                ? "bg-rose-400/20 text-rose-300 border border-rose-400/40"
                : step.status === "active"
                ? "bg-cyan-400/20 text-cyan-300 border border-cyan-400/40 animate-pulse"
                : "bg-slate-700/40 text-slate-400 border border-slate-600/40"
            )}
          >
            {step.status === "complete" ? "✓" : step.status === "error" ? "✕" : idx + 1}
          </div>
          <div className="flex-1">
            <div className="text-sm text-slate-200">{step.label}</div>
          </div>
        </div>
      ))}
      {estimatedTime && <div className="text-xs text-slate-400 mt-4">Estimated time: {estimatedTime}</div>}
    </div>
  );
}

export function ScoreBadge({
  score,
  maxScore = 100,
  label,
  variant = "neutral",
}: {
  score: number | string;
  maxScore?: number;
  label?: string;
  variant?: "neutral" | "good" | "warn" | "bad";
}) {
  const numScore = typeof score === "number" ? score : parseFloat(score as string) || 0;
  const pct = Math.round((numScore / maxScore) * 100);
  const colors =
    variant === "good"
      ? "bg-emerald-400/15 border-emerald-300/40 text-emerald-100"
      : variant === "bad"
      ? "bg-rose-100 border-rose-300 text-rose-800"
      : variant === "warn"
      ? "bg-amber-100 border-amber-300 text-amber-900"
      : "bg-slate-100 border-slate-300 text-slate-800";

  return (
    <span className={cx("inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium", colors)}>
      <span>{label || "Score"}</span>
      <span className="font-bold">{pct}%</span>
    </span>
  );
}

export function VerdictBadge({
  verdict,
  confidence,
  compact = false,
}: {
  verdict: "SUPPORTED" | "REFUTED" | "NEI" | string;
  confidence?: number;
  compact?: boolean;
}) {
  const v = (verdict || "NEI").toUpperCase();
  const label = v === "SUPPORTED" ? "REAL" : v === "REFUTED" ? "FAKE" : "UNCLEAR";
  const color =
    v === "SUPPORTED"
      ? "verdict-supported"
      : v === "REFUTED"
      ? "verdict-refuted"
      : "verdict-nei";

  if (compact) {
    return (
      <span className={cx("inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold", color)}>
        {label}
      </span>
    );
  }

  return (
    <span className={cx("inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold", color)}>
      <span className="text-sm">{v === "SUPPORTED" ? "✓" : v === "REFUTED" ? "✕" : "?"}</span>
      <span>{v}</span>
      {typeof confidence === "number" && <span className="opacity-75">{Math.round(confidence * 100)}%</span>}
    </span>
  );
}

export function SentimentBadge({
  label,
  score,
  emotionalIntensity,
  biasRisk,
}: {
  label: "positive" | "negative" | "neutral";
  score?: number;
  emotionalIntensity?: number;
  biasRisk?: "low" | "medium" | "high";
}) {
  const icon = label === "positive" ? "😊" : label === "negative" ? "😠" : "😐";
  const color =
    label === "positive"
      ? "bg-blue-400/15 border-blue-300/40 text-blue-100 hover:bg-blue-400/20"
      : label === "negative"
      ? "bg-orange-100 border-orange-300 text-orange-900 hover:bg-orange-200"
      : "bg-slate-100 border-slate-300 text-slate-800 hover:bg-slate-200";

  const riskColor =
    biasRisk === "high" ? "text-rose-300" : biasRisk === "medium" ? "text-amber-300" : "text-emerald-300";

  return (
    <Tooltip content={`Emotional intensity: ${Math.round((emotionalIntensity || 0) * 100)}% | Bias risk: ${biasRisk || "unknown"}`}>
      <span
        className={cx(
          "inline-flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-semibold cursor-help transition",
          color
        )}
      >
        <span className="text-sm">{icon}</span>
        <span className="capitalize">{label}</span>
        {typeof score === "number" && <span className={cx("text-xs", riskColor)}>●</span>}
      </span>
    </Tooltip>
  );
}

export function Badge({
  children,
  variant = "neutral",
}: {
  children: React.ReactNode;
  variant?: "neutral" | "good" | "warn" | "bad";
}) {
  const cls =
    variant === "good"
      ? "bg-emerald-100 text-emerald-800 border-emerald-200"
      : variant === "bad"
      ? "bg-rose-100 text-rose-800 border-rose-200"
      : variant === "warn"
      ? "bg-amber-100 text-amber-900 border-amber-200"
      : "bg-zinc-100 text-zinc-800 border-zinc-200";

  return (
    <span className={cx("inline-flex items-center border rounded-full px-3 py-1 text-xs font-semibold", cls)}>
      {children}
    </span>
  );
}

export function Card({
  title,
  subtitle,
  right,
  children,
}: {
  title?: string;
  subtitle?: string;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="bg-white border rounded-2xl shadow-sm">
      {(title || subtitle || right) && (
        <div className="p-5 border-b flex items-start justify-between gap-3 flex-wrap">
          <div>
            {title && <div className="text-sm font-semibold text-zinc-900">{title}</div>}
            {subtitle && <div className="text-xs text-zinc-500 mt-1">{subtitle}</div>}
          </div>
          {right}
        </div>
      )}
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  title,
  type = "button",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "outline" | "ghost";
  title?: string;
  type?: "button" | "submit";
}) {
  const base = "px-4 py-2 rounded-lg text-sm font-medium transition";
  const cls =
    variant === "primary"
      ? disabled
        ? "bg-zinc-200 text-zinc-600 cursor-not-allowed"
        : "bg-zinc-900 text-white hover:bg-zinc-800"
      : variant === "outline"
      ? disabled
        ? "border text-zinc-400 cursor-not-allowed"
        : "border hover:bg-zinc-50"
      : disabled
      ? "text-zinc-400 cursor-not-allowed"
      : "hover:bg-zinc-100";

  return (
    <button type={type} className={cx(base, cls)} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  );
}

export function SmallLink({
  href,
  children,
  newTab = false,
}: {
  href: string;
  children: React.ReactNode;
  newTab?: boolean;
}) {
  return (
    <a
      className="text-xs px-2 py-1 rounded-lg border hover:bg-zinc-50 inline-flex items-center gap-1"
      href={href}
      target={newTab ? "_blank" : undefined}
      rel={newTab ? "noreferrer" : undefined}
    >
      {children}
    </a>
  );
}

export function Divider() {
  return <div className="h-px bg-zinc-100 my-4" />;
}

export function Alert({
  type = "info",
  title,
  message,
  action,
}: {
  type?: "info" | "success" | "warn" | "error";
  title?: string;
  message: string;
  action?: { label: string; onClick: () => void };
}) {
  const colors =
    type === "success"
      ? "bg-emerald-400/10 border-emerald-300/30 text-emerald-100"
      : type === "error"
      ? "bg-rose-100 border-rose-300 text-rose-800"
      : type === "warn"
      ? "bg-amber-100 border-amber-300 text-amber-900"
      : "bg-slate-100 border-slate-300 text-slate-800";

  const icon =
    type === "success" ? "✓" : type === "error" ? "✕" : type === "warn" ? "!" : "ℹ";

  return (
    <div className={cx("rounded-lg border p-4 flex items-start justify-between gap-3", colors)}>
      <div className="flex gap-3">
        <span className="text-lg">{icon}</span>
        <div>
          {title && <div className="font-semibold">{title}</div>}
          <div className="text-sm">{message}</div>
        </div>
      </div>
      {action && (
        <button
          onClick={action.onClick}
          className="px-2 py-1 text-xs font-medium rounded hover:opacity-75 transition whitespace-nowrap"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}

export function StatCard({
  label,
  value,
  unit,
  icon,
  trend,
}: {
  label: string;
  value: string | number;
  unit?: string;
  icon?: string;
  trend?: "up" | "down" | "neutral";
}) {
  return (
    <div className="glass-panel rounded-xl border p-4 hover-lift transition">
      <div className="flex items-start justify-between gap-2">
        <div className="text-xs uppercase tracking-wide text-slate-300">{label}</div>
        {icon && <span className="text-lg">{icon}</span>}
      </div>
      <div className="mt-3 flex items-end gap-2">
        <div className="text-2xl font-bold text-white">{value}</div>
        {unit && <div className="text-sm text-slate-400">{unit}</div>}
      </div>
      {trend && (
        <div className={cx("mt-2 text-xs", trend === "up" ? "text-emerald-300" : trend === "down" ? "text-rose-300" : "text-slate-400")}>
          {trend === "up" ? "↑ Positive" : trend === "down" ? "↓ Negative" : "→ Neutral"}
        </div>
      )}
    </div>
  );
}
