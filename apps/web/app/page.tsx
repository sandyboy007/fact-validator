"use client";

import { useEffect, useMemo, useState } from "react";
import { cx, Tooltip, Tabs, ProgressIndicator, ScoreBadge, VerdictBadge, SentimentBadge, Alert, StatCard } from "../components/ui";

type EvidenceItem = {
  url: string;
  snippet?: string;
  domain?: string;
  domain_score?: number;
  score?: number;
  title?: string;
  quality_score?: number;
  stance?: "support" | "refute" | "neutral" | string;
  source_type?: string;
  primary_source?: boolean;
  primary_source_reason?: string;
  published_year?: number;
  recency_score?: number;
  directness_score?: number;
  quote_grounded?: boolean;
  expertise_match?: number;
  numeric_match?: boolean;
  entity_match?: boolean;
  manipulation_flags?: string[];
};

type ClaimItem = {
  claim_text: string;
  verdict: "SUPPORTED" | "REFUTED" | "NEI" | string;
  confidence: number;
  debate_summary?: string;
  evidence: EvidenceItem[];
  structured_verdict?: string;
  uncertainty_reasons?: string[];
  needs_human_review?: boolean;
  human_review_reason?: string;
  claim_profile?: {
    atomic_claims?: string[];
    entities?: string[];
    numbers?: string[];
    years?: number[];
    expertise_profile?: string;
    loaded_language_terms?: string[];
  };
  evidence_summary?: {
    evidence_count?: number;
    high_credibility_sources?: number;
    primary_source_count?: number;
    primary_source_present?: boolean;
    supporting_items?: number;
    refuting_items?: number;
    conflict_level?: string;
    distinct_domains?: number;
    oldest_citation_year?: number;
    newest_citation_year?: number;
    average_quality_score?: number;
  };
  sentiment?: {
    score: number;
    label: "positive" | "negative" | "neutral";
    emotional_intensity: number;
    bias_risk: "low" | "medium" | "high";
    manipulation_flags?: string[];
  };
};

type AnalyzeResponse = {
  input_type: "url" | "text" | string;
  domain?: string;
  extracted_text_chars?: number;
  extracted_text_preview?: string;
  domain_score?: number;
  domain_label?: string;
  final_misinformation_likelihood?: number;
  claims?: ClaimItem[];
  timestamp_utc?: string;
  metadata?: Record<string, unknown>;
};

type RunRow = {
  id: number | string;
  time_utc?: string;
  timestamp_utc?: string;
  input_type?: string;
  type?: string;
  domain?: string;
  url?: string;
};

type ComparativeRankingRow = {
  system: string;
  n_claims: number;
  accuracy: number;
  avg_confidence: number;
  calibration_error: number;
  ece: number;
};

type ComparativeComparisonRow = {
  baseline_name: string;
  improvement_pct_points: number;
  significance_test?: {
    p_value?: number | null;
    is_significant_alpha_0_05?: boolean;
  };
};

type ComparativeReport = {
  metadata?: {
    generated_utc?: string;
    claims_compared?: number;
    full_variant?: string;
  };
  ranking?: ComparativeRankingRow[];
  comparisons?: ComparativeComparisonRow[];
  debate_lift?: {
    accuracy_delta_pct_points?: number;
    prediction_change_rate?: number;
  };
};

type ProductionMetricsReport = {
  metadata?: {
    generated_utc?: string;
    claims_in_split?: number;
  };
  latency?: {
    baseline_avg_sec?: number;
    debate_avg_sec?: number;
    debate_over_baseline_ratio?: number;
  };
  throughput?: {
    baseline_claims_per_hour?: number;
    debate_claims_per_hour?: number;
  };
  cost?: {
    monthly_usd_no_cache?: number;
    monthly_usd_with_cache?: number;
    monthly_savings_usd?: number;
    monthly_savings_pct?: number;
  };
  quality?: {
    accuracy?: number;
    error_rate?: number;
    macro_f1?: number;
    calibration_error?: number;
    ece?: number;
  };
};

type ExplainabilityCase = {
  claim_id: string;
  claim_text: string;
  ground_truth_label: string;
  predictions?: {
    full?: { label: string; confidence: number };
    baseline?: { label: string; confidence: number };
    no_debate?: { label: string; confidence: number };
  };
  scoring_logic?: string[];
  debate_trace?: {
    prover?: string;
    skeptic?: string;
    judge?: string;
  };
};

type ExplainabilityReport = {
  metadata?: {
    generated_utc?: string;
    case_count?: number;
    best_baseline?: string;
  };
  case_studies?: ExplainabilityCase[];
};

type LimitationsReport = {
  metadata?: {
    limitation_count?: number;
    high_severity_count?: number;
    generated_utc?: string;
  };
  limitations?: Array<{
    id: string;
    title: string;
    severity: string;
    impact: string;
    evidence: string;
    mitigation: string;
  }>;
};

type ReproducibilityReport = {
  summary?: {
    passed_checks?: number;
    total_checks?: number;
  };
  score?: {
    score_percent?: number;
  };
  metadata?: {
    git_commit?: string;
    git_branch?: string;
    generated_utc?: string;
  };
};

type EthicsReport = {
  metadata?: {
    risk_count?: number;
    high_severity_count?: number;
    generated_utc?: string;
  };
  ethical_risks?: Array<{
    id: string;
    title: string;
    severity: string;
    owner: string;
    mitigation: string;
  }>;
};

type DefenseReport = {
  qa?: Array<{
    category: string;
    question: string;
    answer: string;
    evidence: string;
  }>;
  metrics_cheatsheet?: Array<{
    metric: string;
    value: string;
    source: string;
  }>;
  metadata?: {
    generated_utc?: string;
  };
};

function fmtPct(x?: number) {
  if (typeof x !== "number") return "-";
  const v = Math.max(0, Math.min(1, x));
  return `${Math.round(v * 100)}%`;
}

function fmtCount(x?: number) {
  if (typeof x !== "number") return "-";
  return Math.round(x).toLocaleString();
}

function fmtSeconds(x?: number, digits = 2) {
  if (typeof x !== "number") return "-";
  return `${x.toFixed(digits)} s`;
}

function fmtRatePerHour(x?: number) {
  if (typeof x !== "number") return "-";
  return `${x.toFixed(1)} claims/hour`;
}

function fmtMoney(x?: number) {
  if (typeof x !== "number") return "-";
  return `$${x.toFixed(2)}`;
}

function fmtDateTime(utc?: string) {
  if (!utc) return "-";
  const normalized = utc.replace("T", " ").replace("Z", "");
  return normalized.slice(0, 19);
}

function safeHostFromUrl(u?: string) {
  try {
    if (!u) return "-";
    return new URL(u).hostname;
  } catch {
    return "-";
  }
}

function evScore(ev: EvidenceItem): number {
  if (typeof ev.domain_score === "number") return ev.domain_score;
  if (typeof ev.score === "number") return ev.score;
  return 0;
}

function evQuality(ev: EvidenceItem): number {
  if (typeof ev.quality_score === "number") return ev.quality_score;
  return evScore(ev);
}

function formatSystemName(raw?: string) {
  if (!raw) return "-";
  const normalized = raw.toLowerCase();
  if (normalized === "full_proxy") return "Full Verifier";
  if (normalized === "baseline") return "Baseline Verifier";
  if (normalized === "no_debate") return "Verifier (Debate Off)";
  return raw.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
}

function formatDebateDelta(delta?: number) {
  if (typeof delta !== "number") return "-";
  if (Math.abs(delta) < 0.005) {
    return `No measurable gain (0.00 percentage points)`;
  }
  const magnitude = Math.abs(delta).toFixed(2);
  const signed = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}`;
  if (delta > 0) {
    return `Improved by ${magnitude} percentage points (${signed})`;
  }
  return `Declined by ${magnitude} percentage points (${signed})`;
}

function sortEvidence(evs: EvidenceItem[]) {
  return [...(evs || [])].sort((a, b) => {
    const qa = evQuality(a);
    const qb = evQuality(b);
    if (qb !== qa) return qb - qa;
    const sa = evScore(a);
    const sb = evScore(b);
    if (sb !== sa) return sb - sa;
    return (a.domain ?? "").localeCompare(b.domain ?? "");
  });
}

function staggerStyle(index: number, stepMs = 70) {
  return { animationDelay: `${index * stepMs}ms` };
}

function ModuleStatusChip({
  loading,
  error,
  ready,
}: {
  loading: boolean;
  error: string | null;
  ready: boolean;
}) {
  const label = loading ? "Refreshing" : error ? "Unavailable" : ready ? "Live" : "Pending";
  const tone = loading
    ? "border-cyan-300/40 bg-cyan-400/10 text-cyan-100"
    : error
      ? "border-rose-300/40 bg-rose-400/10 text-rose-100"
      : ready
        ? "border-emerald-300/40 bg-emerald-400/10 text-emerald-100"
        : "border-slate-400/30 bg-slate-500/10 text-slate-200";

  return (
    <span className={cx("rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider", tone)}>
      {label}
    </span>
  );
}

function filterClaimsBySentiment(
  claims: ClaimItem[],
  sentimentFilter: string,
  biasRiskFilter: string
): ClaimItem[] {
  return claims.filter((claim) => {
    if (sentimentFilter !== "all" && claim.sentiment?.label !== sentimentFilter) {
      return false;
    }
    if (biasRiskFilter !== "all" && claim.sentiment?.bias_risk !== biasRiskFilter) {
      return false;
    }
    return true;
  });
}

export default function Page() {
  const API_BASE = "http://127.0.0.1:8000";

  const [tab, setTab] = useState<"url" | "text">("url");
  const [url, setUrl] = useState("");
  const [text, setText] = useState("");

  const [mode, setMode] = useState<"live" | "snapshot">("live");
  const [verifier, setVerifier] = useState<"baseline" | "debate">("baseline");
  const [debateModeActive, setDebateModeActive] = useState(false);

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxClaims, setMaxClaims] = useState(6);
  const [maxEvidence, setMaxEvidence] = useState(5);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [processingTime, setProcessingTime] = useState(0);

  const [runs, setRuns] = useState<RunRow[]>([]);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [runsLoading, setRunsLoading] = useState(false);

  const [comparative, setComparative] = useState<ComparativeReport | null>(null);
  const [comparativeError, setComparativeError] = useState<string | null>(null);
  const [comparativeLoading, setComparativeLoading] = useState(false);

  const [productionMetrics, setProductionMetrics] = useState<ProductionMetricsReport | null>(null);
  const [productionMetricsError, setProductionMetricsError] = useState<string | null>(null);
  const [productionMetricsLoading, setProductionMetricsLoading] = useState(false);

  const [explainability, setExplainability] = useState<ExplainabilityReport | null>(null);
  const [explainabilityError, setExplainabilityError] = useState<string | null>(null);
  const [explainabilityLoading, setExplainabilityLoading] = useState(false);

  const [limitations, setLimitations] = useState<LimitationsReport | null>(null);
  const [limitationsError, setLimitationsError] = useState<string | null>(null);
  const [limitationsLoading, setLimitationsLoading] = useState(false);

  const [reproAudit, setReproAudit] = useState<ReproducibilityReport | null>(null);
  const [reproAuditError, setReproAuditError] = useState<string | null>(null);
  const [reproAuditLoading, setReproAuditLoading] = useState(false);

  const [ethics, setEthics] = useState<EthicsReport | null>(null);
  const [ethicsError, setEthicsError] = useState<string | null>(null);
  const [ethicsLoading, setEthicsLoading] = useState(false);

  const [defense, setDefense] = useState<DefenseReport | null>(null);
  const [defenseError, setDefenseError] = useState<string | null>(null);
  const [defenseLoading, setDefenseLoading] = useState(false);

  const [openClaimIdx, setOpenClaimIdx] = useState<number | null>(0);
  const [resultTab, setResultTab] = useState<"overview" | "claims" | "evidence">("overview");
  const [audienceMode, setAudienceMode] = useState<"user" | "analyst">("user");
  const [workspaceView, setWorkspaceView] = useState<"evaluation" | "operations" | "governance" | "defense">("evaluation");
  const [showDetailedWorkspace, setShowDetailedWorkspace] = useState(false);
  const [sentimentFilter, setSentimentFilter] = useState<"all" | "positive" | "negative" | "neutral">("all");
  const [biasRiskFilter, setBiasRiskFilter] = useState<"all" | "low" | "medium" | "high">("all");

  const [processingSteps, setProcessingSteps] = useState<
    { label: string; status: "pending" | "active" | "complete" | "error" }[]
  >([]);

  // Calculate estimated time based on verifier mode
  const estimatedTime =
    verifier === "debate" ? "~60-120s" : verifier === "baseline" ? "~10-30s" : "~5-10s";

  // Validation helpers
  const isValidUrl = useMemo(() => {
    if (!url.trim()) return false;
    try {
      new URL(url.trim());
      return true;
    } catch {
      return false;
    }
  }, [url]);

  const isValidText = useMemo(() => {
    return text.trim().length > 10;
  }, [text]);

  const canAnalyze = useMemo(() => {
    if (tab === "url") return isValidUrl;
    return isValidText;
  }, [tab, isValidUrl, isValidText]);

  const debateEnabled = debateModeActive && verifier === "debate";
  const primaryClaim = result?.claims?.[0] ?? null;
  const primaryEvidence = sortEvidence(primaryClaim?.evidence || []);
  const supportEvidence = primaryEvidence.filter((ev) => ev.stance === "support").slice(0, 3);
  const refuteEvidence = primaryEvidence.filter((ev) => ev.stance === "refute").slice(0, 3);
  const neutralEvidence = primaryEvidence
    .filter((ev) => ev.stance !== "support" && ev.stance !== "refute")
    .slice(0, 3);

  const workspaceSummary = useMemo(() => {
    switch (workspaceView) {
      case "evaluation":
        return {
          title: "Model Quality Workspace",
          description: "Track comparative performance, accuracy lift, and confidence calibration across evaluator variants.",
          moduleCount: 1,
        };
      case "operations":
        return {
          title: "Runtime Operations Workspace",
          description: "Monitor latency, throughput, cost efficiency, and inspect explainability traces for live readiness.",
          moduleCount: 2,
        };
      case "governance":
        return {
          title: "Risk & Governance Workspace",
          description: "Review ethics risk posture, reproducibility guarantees, and active limitations with mitigation context.",
          moduleCount: 3,
        };
      default:
        return {
          title: "Presentation Workspace",
          description: "Prepare defense-ready narratives with evidence-backed Q&A and metric talking points.",
          moduleCount: 1,
        };
    }
  }, [workspaceView]);

  async function analyze() {
    setLoading(true);
    setError(null);
    setResult(null);
    setProcessingTime(0);
    const startTime = Date.now();

    // Update processing steps
    setProcessingSteps([
      { label: "Validating input", status: "active" },
      { label: "Extracting content", status: "pending" },
      { label: "Analyzing claims", status: "pending" },
      { label: debateEnabled ? "Running debate mode" : "Scoring credibility", status: "pending" },
    ]);

    try {
      const payload: {
        mode: "live" | "snapshot";
        verifier: "baseline" | "debate";
        max_claims: number;
        max_evidence_per_claim: number;
        url?: string;
        text?: string;
      } = {
        mode,
        verifier: debateEnabled ? "debate" : "baseline",
        max_claims: maxClaims,
        max_evidence_per_claim: maxEvidence,
      };
      if (tab === "url") payload.url = url.trim();
      else payload.text = text.trim();

      // Step 1: Validate
      setProcessingSteps((p) => [{ ...p[0], status: "complete" }, { ...p[1], status: "active" }, ...p.slice(2)]);

      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const t = await res.text();
        setProcessingSteps((p) => [...p.slice(0, -1), { ...p[p.length - 1], status: "error" }]);
        throw new Error(`API error ${res.status}: ${t}`);
      }

      // Step 2: Done extracting
      setProcessingSteps((p) => [{ ...p[0], status: "complete" }, { ...p[1], status: "complete" }, { ...p[2], status: "active" }, ...p.slice(3)]);

      const json = (await res.json()) as AnalyzeResponse;
      if (json.claims) {
        json.claims = json.claims.map((c) => ({ ...c, evidence: sortEvidence(c.evidence || []) }));
      }

      // Step 3: Claims analyzed
      setProcessingSteps((p) => [...p.slice(0, 2), { ...p[2], status: "complete" }, { ...p[3], status: debateEnabled ? "active" : "complete" }]);

      setResult(json);
      setOpenClaimIdx(0);
      setProcessingTime(Date.now() - startTime);
      setProcessingSteps((p) => [...p.slice(0, -1), { ...p[p.length - 1], status: "complete" }]);
      fetchRuns();
    } catch (e: unknown) {
      const errorMsg = e instanceof Error ? e.message : "Failed to analyze";
      setError(errorMsg);
      setProcessingSteps((p) => [...p.slice(0, -1), { ...p[p.length - 1], status: "error" }]);
    } finally {
      setLoading(false);
    }
  }

  async function fetchRuns() {
    setRunsLoading(true);
    setRunsError(null);

    try {
      const res = await fetch(`${API_BASE}/runs?limit=10`);
      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Runs API error ${res.status}: ${t}`);
      }
      const json = await res.json();
      const items: RunRow[] = Array.isArray(json) ? json : (json?.items ?? []);
      setRuns(items);
    } catch (e: unknown) {
      setRuns([]);
      setRunsError(e instanceof Error ? e.message : "Could not load run history. Check API connection.");
    } finally {
      setRunsLoading(false);
    }
  }

  async function fetchComparative() {
    setComparativeLoading(true);
    setComparativeError(null);

    try {
      const res = await fetch(`${API_BASE}/evaluation/comparative`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Comparative API error ${res.status}: ${text}`);
      }
      const json = (await res.json()) as ComparativeReport;
      setComparative(json);
    } catch (e: unknown) {
      setComparative(null);
      setComparativeError(e instanceof Error ? e.message : "Could not load comparative summary.");
    } finally {
      setComparativeLoading(false);
    }
  }

  async function fetchProductionMetrics() {
    setProductionMetricsLoading(true);
    setProductionMetricsError(null);

    try {
      const res = await fetch(`${API_BASE}/evaluation/production-metrics`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Production metrics API error ${res.status}: ${text}`);
      }
      const json = (await res.json()) as ProductionMetricsReport;
      setProductionMetrics(json);
    } catch (e: unknown) {
      setProductionMetrics(null);
      setProductionMetricsError(e instanceof Error ? e.message : "Could not load production metrics.");
    } finally {
      setProductionMetricsLoading(false);
    }
  }

  async function fetchExplainability() {
    setExplainabilityLoading(true);
    setExplainabilityError(null);

    try {
      const res = await fetch(`${API_BASE}/evaluation/explainability`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Explainability API error ${res.status}: ${text}`);
      }
      const json = (await res.json()) as ExplainabilityReport;
      setExplainability(json);
    } catch (e: unknown) {
      setExplainability(null);
      setExplainabilityError(e instanceof Error ? e.message : "Could not load explainability demo.");
    } finally {
      setExplainabilityLoading(false);
    }
  }

  async function fetchLimitations() {
    setLimitationsLoading(true);
    setLimitationsError(null);

    try {
      const res = await fetch(`${API_BASE}/evaluation/limitations`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Limitations API error ${res.status}: ${text}`);
      }
      const json = (await res.json()) as LimitationsReport;
      setLimitations(json);
    } catch (e: unknown) {
      setLimitations(null);
      setLimitationsError(e instanceof Error ? e.message : "Could not load limitations report.");
    } finally {
      setLimitationsLoading(false);
    }
  }

  async function fetchReproAudit() {
    setReproAuditLoading(true);
    setReproAuditError(null);

    try {
      const res = await fetch(`${API_BASE}/evaluation/reproducibility`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Reproducibility API error ${res.status}: ${text}`);
      }
      const json = (await res.json()) as ReproducibilityReport;
      setReproAudit(json);
    } catch (e: unknown) {
      setReproAudit(null);
      setReproAuditError(e instanceof Error ? e.message : "Could not load reproducibility audit.");
    } finally {
      setReproAuditLoading(false);
    }
  }

  async function fetchEthics() {
    setEthicsLoading(true);
    setEthicsError(null);

    try {
      const res = await fetch(`${API_BASE}/evaluation/ethics`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Ethics API error ${res.status}: ${text}`);
      }
      const json = (await res.json()) as EthicsReport;
      setEthics(json);
    } catch (e: unknown) {
      setEthics(null);
      setEthicsError(e instanceof Error ? e.message : "Could not load ethics assessment.");
    } finally {
      setEthicsLoading(false);
    }
  }

  async function fetchDefense() {
    setDefenseLoading(true);
    setDefenseError(null);

    try {
      const res = await fetch(`${API_BASE}/evaluation/defense`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Defense API error ${res.status}: ${text}`);
      }
      const json = (await res.json()) as DefenseReport;
      setDefense(json);
    } catch (e: unknown) {
      setDefense(null);
      setDefenseError(e instanceof Error ? e.message : "Could not load defense talking points.");
    } finally {
      setDefenseLoading(false);
    }
  }

  useEffect(() => {
    fetchRuns();
    fetchComparative();
    fetchProductionMetrics();
    fetchExplainability();
    fetchLimitations();
    fetchReproAudit();
    fetchEthics();
    fetchDefense();
  }, []);

  const examples = [
    "https://ourworldindata.org/co2-and-greenhouse-gas-emissions",
    "https://www.bbc.com/news",
    "https://en.wikipedia.org/wiki/Naples",
    "https://www.who.int",
  ];

  const showResultSkeleton = loading && !result;

  return (
    <main className="min-h-screen text-slate-100 relative overflow-hidden">
      <div className="pointer-events-none absolute inset-0 grid-overlay opacity-30" />
      <div className="pointer-events-none absolute -top-24 left-[-5rem] h-80 w-80 rounded-full bg-cyan-400/20 blur-3xl" />
      <div className="pointer-events-none absolute top-10 right-0 h-96 w-96 rounded-full bg-blue-500/20 blur-3xl" />
      <div className="pointer-events-none absolute bottom-0 left-1/3 h-80 w-80 rounded-full bg-emerald-400/10 blur-3xl" />

      <div className="relative mx-auto max-w-7xl px-4 py-8 md:py-10 section-fade-in flex flex-col gap-6">
        {/* Header */}
        <header className="glass-panel rounded-3xl p-5 md:p-7 border order-1">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div>
              <div className="inline-flex items-center gap-2 rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-[11px] uppercase tracking-wider text-cyan-200">
                <span className="h-2 w-2 rounded-full bg-cyan-300 shadow-[0_0_12px_#42d9ff]" />
                AI Verification Console
              </div>
              <h1 className="mt-4 text-3xl md:text-4xl font-semibold leading-tight text-white">
                Fact Validator AI
              </h1>
              <p className="mt-2 max-w-2xl text-sm md:text-base text-slate-300">
                Professional-grade claim intelligence with credibility scoring, counter-evidence analysis, uncertainty disclosure, and explainable verdicts.
              </p>
              <div className="mt-3 inline-flex rounded-xl border border-slate-300/20 overflow-hidden">
                <button
                  className={cx("px-3 py-1.5 text-xs font-semibold transition", audienceMode === "user" ? "bg-cyan-400/15 text-cyan-100" : "text-slate-300 hover:bg-slate-800/50")}
                  onClick={() => setAudienceMode("user")}
                >
                  User View
                </button>
                <button
                  className={cx("px-3 py-1.5 text-xs font-semibold transition", audienceMode === "analyst" ? "bg-cyan-400/15 text-cyan-100" : "text-slate-300 hover:bg-slate-800/50")}
                  onClick={() => setAudienceMode("analyst")}
                >
                  Analyst View
                </button>
              </div>
            </div>
            <div className="w-full lg:w-auto flex flex-col gap-2 text-xs md:text-sm">
              <div className="flex flex-wrap gap-2">
                <a href="/source" className="glass-panel rounded-xl px-3 md:px-4 py-2 md:py-2.5 text-slate-100 hover:border-cyan-300/40 transition border text-center">Source Checker</a>
                {audienceMode === "analyst" && (
                  <>
                    <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer" className="glass-panel rounded-xl px-3 md:px-4 py-2 md:py-2.5 text-slate-100 hover:border-cyan-300/40 transition border text-center">API Docs</a>
                    <a href={`${API_BASE}/evaluation/comparative`} target="_blank" rel="noreferrer" className="glass-panel rounded-xl px-3 md:px-4 py-2 md:py-2.5 text-slate-100 hover:border-cyan-300/40 transition border text-center">Comparative</a>
                    <a href={`${API_BASE}/evaluation/production-metrics`} target="_blank" rel="noreferrer" className="glass-panel rounded-xl px-3 md:px-4 py-2 md:py-2.5 text-slate-100 hover:border-cyan-300/40 transition border text-center">Operations</a>
                  </>
                )}
              </div>
              <details className={cx("rounded-xl border border-slate-300/20 bg-slate-900/35 px-3 py-2 text-slate-200", audienceMode === "user" ? "hidden" : "block")}>
                <summary className="cursor-pointer select-none text-sm">More reports</summary>
                <div className="mt-2 flex flex-wrap gap-2">
                  <a href={`${API_BASE}/evaluation/benchmark`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-2.5 py-1.5 hover:border-cyan-300/40 transition">Benchmark</a>
                  <a href={`${API_BASE}/evaluation/baselines`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-2.5 py-1.5 hover:border-cyan-300/40 transition">Baselines</a>
                  <a href={`${API_BASE}/evaluation/ablations`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-2.5 py-1.5 hover:border-cyan-300/40 transition">Ablation</a>
                  <a href={`${API_BASE}/evaluation/explainability`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-2.5 py-1.5 hover:border-cyan-300/40 transition">Explainability</a>
                  <a href={`${API_BASE}/evaluation/limitations`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-2.5 py-1.5 hover:border-cyan-300/40 transition">Limitations</a>
                  <a href={`${API_BASE}/evaluation/reproducibility`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-2.5 py-1.5 hover:border-cyan-300/40 transition">Reproducibility</a>
                  <a href={`${API_BASE}/evaluation/ethics`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-2.5 py-1.5 hover:border-cyan-300/40 transition">Ethics</a>
                  <a href={`${API_BASE}/evaluation/defense`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-2.5 py-1.5 hover:border-cyan-300/40 transition">Defense</a>
                </div>
              </details>
            </div>
          </div>
        </header>

        {audienceMode === "analyst" && (
        <div className="order-2">

        <section className="glass-panel rounded-2xl border p-3 md:p-4 mb-6 section-fade-in">
          <Tabs
            tabs={[
              { label: "Evaluation", value: "evaluation", icon: "📈" },
              { label: "Operations", value: "operations", icon: "⚙️" },
              { label: "Governance", value: "governance", icon: "🛡️" },
              { label: "Defense", value: "defense", icon: "🎤" },
            ]}
            activeTab={workspaceView}
            onTabChange={(v) => setWorkspaceView(v as "evaluation" | "operations" | "governance" | "defense")}
          />
          <div className="mt-3 rounded-xl border border-slate-700/40 bg-slate-900/35 px-3 py-2.5 md:px-4 md:py-3">
            <div className="flex items-center justify-between gap-2 flex-wrap">
              <div className="text-sm font-semibold text-slate-100">{workspaceSummary.title}</div>
              <div className="flex items-center gap-2">
                <div className="text-[11px] uppercase tracking-wider text-slate-400">{workspaceSummary.moduleCount} active module{workspaceSummary.moduleCount > 1 ? "s" : ""}</div>
                <button
                  className="rounded-lg border border-slate-400/30 px-2.5 py-1 text-[11px] font-semibold text-slate-200 hover:border-cyan-300/40 transition"
                  onClick={() => setShowDetailedWorkspace((v) => !v)}
                >
                  {showDetailedWorkspace ? "Simple View" : "Detailed View"}
                </button>
              </div>
            </div>
            <p className="mt-1 text-xs md:text-sm text-slate-300">{workspaceSummary.description}</p>
          </div>
        </section>

        {workspaceView === "defense" && (
        <section className="glass-panel rounded-2xl border p-5 mb-6 section-fade-in">
          <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white">🎤 Defense Briefing</h3>
              <ModuleStatusChip loading={defenseLoading} error={defenseError} ready={Boolean(defense)} />
            </div>
            <div className="flex items-center gap-2">
              <a href={`${API_BASE}/evaluation/defense`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs text-slate-200 hover:border-cyan-300/40 transition">
                Open JSON
              </a>
              <button
                className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs hover:border-cyan-300/40 transition disabled:opacity-50"
                onClick={fetchDefense}
                disabled={defenseLoading}
              >
                {defenseLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>

          {defenseLoading && !defense ? (
            <div className="h-20 rounded-xl bg-slate-700/30 animate-pulse" />
          ) : defenseError ? (
            <Alert type="warn" title="Defense data unavailable" message={defenseError} />
          ) : defense ? (
            <div className="space-y-3">
              <div className="rounded-lg border border-cyan-400/20 bg-cyan-500/5 p-3">
                <div className="text-xs uppercase tracking-wider text-cyan-200/90 mb-1">Quick interpretation</div>
                <div className="text-sm text-slate-100">
                  Defense pack is ready with <span className="font-semibold">{defense.qa?.length ?? 0}</span> prepared Q&A responses and <span className="font-semibold">{defense.metrics_cheatsheet?.length ?? 0}</span> supporting metrics.
                </div>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Prepared answers</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtCount(defense.qa?.length)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Ready metrics</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtCount(defense.metrics_cheatsheet?.length)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Last updated</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtDateTime(defense.metadata?.generated_utc)}</div>
                </div>
              </div>

              {showDetailedWorkspace && defense.qa && defense.qa.length > 0 && (
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">Featured Q&A</div>
                  <div className="text-sm text-slate-100 mb-1">Q: {defense.qa[0].question}</div>
                  <div className="text-xs text-slate-300 mb-1">A: {defense.qa[0].answer}</div>
                  <div className="text-xs text-slate-400">Evidence: {defense.qa[0].evidence}</div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-slate-400">No defense report loaded yet.</div>
          )}
        </section>
        )}

        {workspaceView === "governance" && (
        <section className="glass-panel rounded-2xl border p-5 mb-6 section-fade-in">
          <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white">⚖️ Ethics & Societal Impact</h3>
              <ModuleStatusChip loading={ethicsLoading} error={ethicsError} ready={Boolean(ethics)} />
            </div>
            <div className="flex items-center gap-2">
              <a href={`${API_BASE}/evaluation/ethics`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs text-slate-200 hover:border-cyan-300/40 transition">
                Open JSON
              </a>
              <button
                className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs hover:border-cyan-300/40 transition disabled:opacity-50"
                onClick={fetchEthics}
                disabled={ethicsLoading}
              >
                {ethicsLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>

          {ethicsLoading && !ethics ? (
            <div className="h-20 rounded-xl bg-slate-700/30 animate-pulse" />
          ) : ethicsError ? (
            <Alert type="warn" title="Ethics data unavailable" message={ethicsError} />
          ) : ethics ? (
            <div className="space-y-4">
              <div className="rounded-lg border border-cyan-400/20 bg-cyan-500/5 p-3">
                <div className="text-xs uppercase tracking-wider text-cyan-200/90 mb-1">Quick interpretation</div>
                <div className="text-sm text-slate-100">
                  Governance scan found <span className="font-semibold">{fmtCount(ethics.metadata?.risk_count)}</span> total ethics risks, including <span className="font-semibold">{fmtCount(ethics.metadata?.high_severity_count)}</span> high-priority items to monitor closely.
                </div>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Risks identified</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtCount(ethics.metadata?.risk_count)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">High-priority risks</div>
                  <div className="text-sm text-rose-200 mt-1">{fmtCount(ethics.metadata?.high_severity_count)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Last updated</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtDateTime(ethics.metadata?.generated_utc)}</div>
                </div>
              </div>

              {showDetailedWorkspace && ethics.ethical_risks && ethics.ethical_risks.length > 0 && (
                <div className="sm:col-span-2 lg:col-span-3 rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">Top ethical risk</div>
                  <div className="text-sm text-slate-100 mb-1">
                    {ethics.ethical_risks[0].id}: {ethics.ethical_risks[0].title}
                  </div>
                  <div className="text-xs text-slate-300 mb-1">Owner: {ethics.ethical_risks[0].owner}</div>
                  <div className="text-xs text-slate-300">Mitigation: {ethics.ethical_risks[0].mitigation}</div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-slate-400">No ethics report loaded yet.</div>
          )}
        </section>
        )}

        {workspaceView === "governance" && (
        <section className="glass-panel rounded-2xl border p-5 mb-6 section-fade-in">
          <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white">🧪 Reproducibility Audit</h3>
              <ModuleStatusChip loading={reproAuditLoading} error={reproAuditError} ready={Boolean(reproAudit)} />
            </div>
            <div className="flex items-center gap-2">
              <a href={`${API_BASE}/evaluation/reproducibility`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs text-slate-200 hover:border-cyan-300/40 transition">
                Open JSON
              </a>
              <button
                className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs hover:border-cyan-300/40 transition disabled:opacity-50"
                onClick={fetchReproAudit}
                disabled={reproAuditLoading}
              >
                {reproAuditLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>

          {reproAuditLoading && !reproAudit ? (
            <div className="h-20 rounded-xl bg-slate-700/30 animate-pulse" />
          ) : reproAuditError ? (
            <Alert type="warn" title="Reproducibility data unavailable" message={reproAuditError} />
          ) : reproAudit ? (
            <div className="space-y-4">
              <div className="rounded-lg border border-cyan-400/20 bg-cyan-500/5 p-3">
                <div className="text-xs uppercase tracking-wider text-cyan-200/90 mb-1">Quick interpretation</div>
                <div className="text-sm text-slate-100">
                  Reproducibility confidence is <span className="font-semibold">{typeof reproAudit.score?.score_percent === "number" ? `${reproAudit.score.score_percent.toFixed(1)}%` : "-"}</span> with <span className="font-semibold">{fmtCount(reproAudit.summary?.passed_checks)}</span> of <span className="font-semibold">{fmtCount(reproAudit.summary?.total_checks)}</span> checklist items passing.
                </div>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Reproducibility confidence</div>
                  <div className="text-sm text-cyan-100 mt-1">
                    {typeof reproAudit.score?.score_percent === "number"
                      ? `${reproAudit.score.score_percent.toFixed(1)}%`
                      : "-"}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Checklist pass count</div>
                  <div className="text-sm text-slate-100 mt-1">
                    {fmtCount(reproAudit.summary?.passed_checks)}/{fmtCount(reproAudit.summary?.total_checks)}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Source branch</div>
                  <div className="text-sm text-slate-100 mt-1">{reproAudit.metadata?.git_branch || "-"}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Source revision</div>
                  <div className="text-sm text-slate-100 mt-1">{reproAudit.metadata?.git_commit || "-"}</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-400">No reproducibility report loaded yet.</div>
          )}
        </section>
        )}

        {workspaceView === "evaluation" && (
        <section className="glass-panel rounded-2xl border p-5 mb-6 section-fade-in">
          <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white">📈 Comparative Evaluation</h3>
              <ModuleStatusChip loading={comparativeLoading} error={comparativeError} ready={Boolean(comparative)} />
            </div>
            <div className="flex items-center gap-2">
              <a href={`${API_BASE}/evaluation/comparative`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs text-slate-200 hover:border-cyan-300/40 transition">
                Open JSON
              </a>
              <button
                className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs hover:border-cyan-300/40 transition disabled:opacity-50"
                onClick={fetchComparative}
                disabled={comparativeLoading}
              >
                {comparativeLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>

          {comparativeLoading && !comparative ? (
            <div className="h-20 rounded-xl bg-slate-700/30 animate-pulse" />
          ) : comparativeError ? (
            <Alert type="warn" title="Comparative data unavailable" message={comparativeError} />
          ) : comparative ? (
            <div className="space-y-4">
              {(() => {
                const top = comparative.ranking?.[0];
                const topAcc = typeof top?.accuracy === "number" ? top.accuracy : undefined;
                const claims = comparative.metadata?.claims_compared;
                const correct = typeof topAcc === "number" && typeof claims === "number"
                  ? Math.round(topAcc * claims)
                  : null;
                const delta = comparative.debate_lift?.accuracy_delta_pct_points;

                return (
                  <div className="rounded-lg border border-cyan-400/20 bg-cyan-500/5 p-3">
                    <div className="text-xs uppercase tracking-wider text-cyan-200/90 mb-1">Quick interpretation</div>
                    <div className="text-sm text-slate-100">
                      Best current setup is <span className="font-semibold">{formatSystemName(top?.system)}</span>
                      {typeof topAcc === "number" ? ` with ${Math.round(topAcc * 100)}% correct verdicts` : ""}
                      {typeof correct === "number" && typeof claims === "number" ? ` (${fmtCount(correct)}/${fmtCount(claims)})` : ""}
                      {typeof delta === "number" ? ` and debate impact is ${formatDebateDelta(delta).toLowerCase()}.` : "."}
                    </div>
                    {typeof claims === "number" && claims < 30 && (
                      <div className="mt-2 text-xs text-amber-200/90">
                        Small sample notice: only {claims} claims were evaluated, so this is directional and may change with larger runs.
                      </div>
                    )}
                  </div>
                );
              })()}

              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Best model setup</div>
                  <div className="text-sm text-cyan-100 font-semibold mt-1">{formatSystemName(comparative.ranking?.[0]?.system)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Correct verdict rate</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtPct(comparative.ranking?.[0]?.accuracy)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Claims evaluated</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtCount(comparative.metadata?.claims_compared)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Gain from debate mode</div>
                  <div className="text-sm text-slate-100 mt-1">{formatDebateDelta(comparative.debate_lift?.accuracy_delta_pct_points)}</div>
                </div>
              </div>

              {showDetailedWorkspace && (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="bg-slate-900/70 text-slate-300 border-b border-slate-700">
                      <tr>
                        <th className="px-3 py-2 text-left text-xs font-semibold">Rank</th>
                        <th className="px-3 py-2 text-left text-xs font-semibold">System</th>
                        <th className="px-3 py-2 text-left text-xs font-semibold">Accuracy</th>
                        <th className="px-3 py-2 text-left text-xs font-semibold">Avg Conf.</th>
                        <th className="px-3 py-2 text-left text-xs font-semibold">ECE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(comparative.ranking || []).slice(0, 5).map((row, idx) => (
                        <tr key={`${row.system}_${idx}`} className="border-t border-slate-700/50 hover:bg-slate-900/40 transition">
                          <td className="px-3 py-2 text-slate-300">#{idx + 1}</td>
                          <td className="px-3 py-2 text-slate-100 font-medium">{row.system}</td>
                          <td className="px-3 py-2 text-slate-200">{fmtPct(row.accuracy)}</td>
                          <td className="px-3 py-2 text-slate-300">{typeof row.avg_confidence === "number" ? `${row.avg_confidence.toFixed(1)}%` : "-"}</td>
                          <td className="px-3 py-2 text-slate-300">{typeof row.ece === "number" ? row.ece.toFixed(3) : "-"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              {showDetailedWorkspace && comparative.comparisons && comparative.comparisons.length > 0 && (
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3 text-xs text-slate-300">
                  <span className="text-slate-400 uppercase tracking-wider">Best delta vs comparator:</span>{" "}
                  {(() => {
                    const sorted = [...comparative.comparisons!].sort(
                      (a, b) => (b.improvement_pct_points || 0) - (a.improvement_pct_points || 0)
                    );
                    const best = sorted[0];
                    if (!best) return "-";
                    const pVal = best.significance_test?.p_value;
                    const pTxt = typeof pVal === "number" ? pVal.toFixed(4) : "NA";
                    return `${formatSystemName(best.baseline_name)}: ${formatDebateDelta(best.improvement_pct_points)} (p=${pTxt})`;
                  })()}
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-slate-400">No comparative report loaded yet.</div>
          )}
        </section>
        )}

        {workspaceView === "operations" && (
        <section className="glass-panel rounded-2xl border p-5 mb-6 section-fade-in">
          <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white">⚙️ Operational Metrics</h3>
              <ModuleStatusChip loading={productionMetricsLoading} error={productionMetricsError} ready={Boolean(productionMetrics)} />
            </div>
            <div className="flex items-center gap-2">
              <a href={`${API_BASE}/evaluation/production-metrics`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs text-slate-200 hover:border-cyan-300/40 transition">
                Open JSON
              </a>
              <button
                className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs hover:border-cyan-300/40 transition disabled:opacity-50"
                onClick={fetchProductionMetrics}
                disabled={productionMetricsLoading}
              >
                {productionMetricsLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>

          {productionMetricsLoading && !productionMetrics ? (
            <div className="h-20 rounded-xl bg-slate-700/30 animate-pulse" />
          ) : productionMetricsError ? (
            <Alert type="warn" title="Production metrics unavailable" message={productionMetricsError} />
          ) : productionMetrics ? (
            <div className="space-y-4">
              <div className="rounded-lg border border-cyan-400/20 bg-cyan-500/5 p-3">
                <div className="text-xs uppercase tracking-wider text-cyan-200/90 mb-1">Quick interpretation</div>
                <div className="text-sm text-slate-100">
                  Current pipeline processes about <span className="font-semibold">{fmtRatePerHour(productionMetrics.throughput?.debate_claims_per_hour)}</span> with an estimated <span className="font-semibold">{fmtPct(productionMetrics.quality?.error_rate)}</span> wrong-verdict rate.
                </div>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Average response time</div>
                  <div className="text-sm text-slate-100 mt-1">
                    {fmtSeconds(productionMetrics.latency?.baseline_avg_sec)}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Processing capacity</div>
                  <div className="text-sm text-slate-100 mt-1">
                    {fmtRatePerHour(productionMetrics.throughput?.debate_claims_per_hour)}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Estimated monthly savings</div>
                  <div className="text-sm text-emerald-200 mt-1">
                    {fmtMoney(productionMetrics.cost?.monthly_savings_usd)}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Wrong verdict rate</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtPct(productionMetrics.quality?.error_rate)}</div>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-400">No production metrics loaded yet.</div>
          )}
        </section>
        )}

        {workspaceView === "operations" && (
        <section className="glass-panel rounded-2xl border p-5 mb-6 section-fade-in">
          <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white">🧠 Explainability Cases</h3>
              <ModuleStatusChip loading={explainabilityLoading} error={explainabilityError} ready={Boolean(explainability)} />
            </div>
            <div className="flex items-center gap-2">
              <a href={`${API_BASE}/evaluation/explainability`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs text-slate-200 hover:border-cyan-300/40 transition">
                Open JSON
              </a>
              <button
                className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs hover:border-cyan-300/40 transition disabled:opacity-50"
                onClick={fetchExplainability}
                disabled={explainabilityLoading}
              >
                {explainabilityLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>

          {explainabilityLoading && !explainability ? (
            <div className="h-20 rounded-xl bg-slate-700/30 animate-pulse" />
          ) : explainabilityError ? (
            <Alert type="warn" title="Explainability data unavailable" message={explainabilityError} />
          ) : explainability ? (
            <div className="space-y-4">
              <div className="rounded-lg border border-cyan-400/20 bg-cyan-500/5 p-3">
                <div className="text-xs uppercase tracking-wider text-cyan-200/90 mb-1">Quick interpretation</div>
                <div className="text-sm text-slate-100">
                  Explainability includes <span className="font-semibold">{fmtCount(explainability.metadata?.case_count)}</span> annotated examples so you can see why verdicts were made, compared against <span className="font-semibold">{formatSystemName(explainability.metadata?.best_baseline)}</span>.
                </div>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Explained examples</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtCount(explainability.metadata?.case_count)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Compared against</div>
                  <div className="text-sm text-slate-100 mt-1">{formatSystemName(explainability.metadata?.best_baseline)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Last updated</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtDateTime(explainability.metadata?.generated_utc)}</div>
                </div>
              </div>

              {showDetailedWorkspace && explainability.case_studies && explainability.case_studies.length > 0 && (
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">Sample case</div>
                  <div className="text-sm text-slate-100 mb-2">{explainability.case_studies[0].claim_text}</div>
                  <div className="text-xs text-slate-300 mb-2">
                    Full: {explainability.case_studies[0].predictions?.full?.label || "-"} | Baseline: {explainability.case_studies[0].predictions?.baseline?.label || "-"}
                  </div>
                  <div className="text-xs text-slate-300">
                    Judge: {explainability.case_studies[0].debate_trace?.judge || "-"}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-slate-400">No explainability report loaded yet.</div>
          )}
        </section>
        )}

        {workspaceView === "governance" && (
        <section className="glass-panel rounded-2xl border p-5 mb-6 section-fade-in">
          <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
            <div className="flex items-center gap-2">
              <h3 className="text-base font-semibold text-white">📉 Limitations Register</h3>
              <ModuleStatusChip loading={limitationsLoading} error={limitationsError} ready={Boolean(limitations)} />
            </div>
            <div className="flex items-center gap-2">
              <a href={`${API_BASE}/evaluation/limitations`} target="_blank" rel="noreferrer" className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs text-slate-200 hover:border-cyan-300/40 transition">
                Open JSON
              </a>
              <button
                className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs hover:border-cyan-300/40 transition disabled:opacity-50"
                onClick={fetchLimitations}
                disabled={limitationsLoading}
              >
                {limitationsLoading ? "Refreshing..." : "Refresh"}
              </button>
            </div>
          </div>

          {limitationsLoading && !limitations ? (
            <div className="h-20 rounded-xl bg-slate-700/30 animate-pulse" />
          ) : limitationsError ? (
            <Alert type="warn" title="Limitations data unavailable" message={limitationsError} />
          ) : limitations ? (
            <div className="space-y-4">
              <div className="rounded-lg border border-cyan-400/20 bg-cyan-500/5 p-3">
                <div className="text-xs uppercase tracking-wider text-cyan-200/90 mb-1">Quick interpretation</div>
                <div className="text-sm text-slate-100">
                  The limitations register tracks <span className="font-semibold">{fmtCount(limitations.metadata?.limitation_count)}</span> known constraints, including <span className="font-semibold">{fmtCount(limitations.metadata?.high_severity_count)}</span> high-impact items requiring mitigation.
                </div>
              </div>

              <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Known limitations</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtCount(limitations.metadata?.limitation_count)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">High-impact items</div>
                  <div className="text-sm text-rose-200 mt-1">{fmtCount(limitations.metadata?.high_severity_count)}</div>
                </div>
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400">Last updated</div>
                  <div className="text-sm text-slate-100 mt-1">{fmtDateTime(limitations.metadata?.generated_utc)}</div>
                </div>
              </div>

              {showDetailedWorkspace && limitations.limitations && limitations.limitations.length > 0 && (
                <div className="rounded-lg border border-slate-700/40 bg-slate-900/35 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-400 mb-2">Top limitation</div>
                  <div className="text-sm text-slate-100 mb-1">
                    {limitations.limitations[0].id}: {limitations.limitations[0].title}
                  </div>
                  <div className="text-xs text-slate-300 mb-1">Impact: {limitations.limitations[0].impact}</div>
                  <div className="text-xs text-slate-300">Mitigation: {limitations.limitations[0].mitigation}</div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-slate-400">No limitations report loaded yet.</div>
          )}
        </section>
        )}

        </div>
        )}

        {/* Main Layout */}
        <section className="grid xl:grid-cols-[1.1fr_1.5fr] gap-6 order-1">
          {/* Left: Input Panel */}
          <div id="full-verifier" className="glass-panel rounded-3xl border p-5 md:p-6 section-fade-in">
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm uppercase tracking-wider text-slate-300 font-semibold">Input Setup</div>
              <button className="text-xs px-3 py-1.5 rounded-lg border border-slate-300/20 hover:border-cyan-300/40 transition" onClick={() => setShowAdvanced((v) => !v)}>
                {showAdvanced ? "Hide" : "Show"} advanced
              </button>
            </div>

            {/* Tab Selection */}
            <div className="grid grid-cols-2 gap-2 mb-4">
              <button className={cx("rounded-xl py-2.5 text-sm border transition font-medium", tab === "url" ? "bg-cyan-400/15 border-cyan-300/40 text-cyan-100" : "border-slate-300/20 hover:border-slate-300/40")} onClick={() => setTab("url")}>
                🔗 URL
              </button>
              <button className={cx("rounded-xl py-2.5 text-sm border transition font-medium", tab === "text" ? "bg-cyan-400/15 border-cyan-300/40 text-cyan-100" : "border-slate-300/20 hover:border-slate-300/40")} onClick={() => setTab("text")}>
                📝 Text
              </button>
            </div>

            {/* Verifier Mode with Debate Toggle */}
            <div className="grid sm:grid-cols-2 gap-3 mb-3">
              <label className="text-xs text-slate-300">
                Mode
                <select className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-2 text-sm" value={mode} onChange={(e) => setMode(e.target.value as "live" | "snapshot")}>
                  <option value="live">Live</option>
                  <option value="snapshot">Snapshot</option>
                </select>
              </label>
              <label className="text-xs text-slate-300">
                Verifier
                <select className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-2 text-sm" value={verifier} onChange={(e) => setVerifier(e.target.value as "baseline" | "debate")}>
                  <option value="baseline">Baseline</option>
                  <option value="debate">Debate Mode</option>
                </select>
              </label>
            </div>

            {/* Debate Mode Banner */}
            {verifier === "debate" && (
              <div className="mb-3 p-3 rounded-lg bg-purple-400/10 border border-purple-300/30">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="flex items-center gap-2 text-sm font-semibold text-purple-100">
                      <span>⚖️ Debate Mode Active</span>
                      <Tooltip content="LLMs debate each claim as Prover vs Skeptic with Judge arbiter">
                        <span className="cursor-help text-purple-300/60 text-xs">[?]</span>
                      </Tooltip>
                    </div>
                    <div className="text-xs text-purple-200/70 mt-1">Estimated time: ~60-120s</div>
                  </div>
                  <button
                    onClick={() => setDebateModeActive(!debateModeActive)}
                    className={cx(
                      "px-3 py-1 rounded-lg text-xs font-medium transition",
                      debateModeActive ? "bg-purple-400/30 text-purple-100 border border-purple-300/40" : "border border-purple-300/20 text-purple-200 hover:border-purple-300/40"
                    )}
                  >
                    {debateModeActive ? "✓ Enabled" : "Enable"}
                  </button>
                </div>
              </div>
            )}

            {/* Advanced Options */}
            {showAdvanced && (
              <div className="grid sm:grid-cols-2 gap-3 mb-4 p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                <label className="text-xs text-slate-300">
                  Max claims
                  <input type="number" min={1} max={20} className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-2 text-sm" value={maxClaims} onChange={(e) => setMaxClaims(Math.min(20, Math.max(1, parseInt(e.target.value || "6", 10))))} />
                </label>
                <label className="text-xs text-slate-300">
                  Evidence per claim
                  <input type="number" min={1} max={10} className="mt-1 w-full rounded-xl border border-slate-300/20 bg-slate-900/60 px-3 py-2 text-sm" value={maxEvidence} onChange={(e) => setMaxEvidence(Math.min(10, Math.max(1, parseInt(e.target.value || "5", 10))))} />
                </label>
              </div>
            )}

            {/* Input Field */}
            {tab === "url" ? (
              <div className="mt-4">
                <label className="text-xs text-slate-300 block">Target URL</label>
                <div className="mt-1 relative">
                  <input
                    className={cx("w-full rounded-xl border bg-slate-900/60 px-3 py-3 text-sm transition", isValidUrl ? "border-emerald-300/40 bg-emerald-400/5" : url.trim() ? "border-rose-300/40 bg-rose-400/5" : "border-slate-300/20")}
                    placeholder="https://example.com/article"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                  />
                  {isValidUrl && <span className="absolute right-3 top-3 text-emerald-400">✓</span>}
                  {url.trim() && !isValidUrl && <span className="absolute right-3 top-3 text-rose-400">✕</span>}
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {examples.map((x) => (
                    <button key={x} className="rounded-full border border-slate-300/20 px-2 py-1 text-xs text-slate-300 hover:border-cyan-300/40 hover:text-cyan-100 transition truncate max-w-[120px]" onClick={() => setUrl(x)} title={x}>
                      Sample
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="mt-4">
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-slate-300">Claim or text</label>
                  <span className="text-xs text-slate-400">{text.length}/10000</span>
                </div>
                <textarea
                  className={cx("w-full min-h-[140px] rounded-xl border bg-slate-900/60 px-3 py-3 text-sm transition", isValidText ? "border-emerald-300/40 bg-emerald-400/5" : text.trim() ? "border-rose-300/40 bg-rose-400/5" : "border-slate-300/20")}
                  placeholder="Paste a claim to verify..."
                  value={text}
                  onChange={(e) => setText(e.target.value.slice(0, 10000))}
                  maxLength={10000}
                />
              </div>
            )}

            {/* Submit Button */}
            <button
              className={cx(
                "mt-4 w-full rounded-xl px-4 py-3 text-sm font-semibold transition",
                !canAnalyze || loading
                  ? "bg-slate-700/60 text-slate-400 cursor-not-allowed"
                  : "bg-gradient-to-r from-cyan-300 to-blue-400 text-slate-950 hover:brightness-110 shadow-lg hover:shadow-cyan-400/50"
              )}
              onClick={analyze}
              disabled={!canAnalyze || loading}
            >
              {loading ? "⏳ Analyzing..." : "🔍 Run verification"}
            </button>

            {/* Error Alert */}
            {error && (
              <Alert
                type="error"
                title="Verification failed"
                message={error}
                action={{ label: "Retry", onClick: analyze }}
              />
            )}

            {/* Loading Progress */}
            {loading && processingSteps.length > 0 && (
              <div className="mt-4 p-4 rounded-lg bg-slate-800/40 border border-slate-700/40">
                <ProgressIndicator
                  steps={processingSteps}
                  currentStep={processingSteps.findIndex((s) => s.status === "active")}
                  estimatedTime={estimatedTime}
                />
              </div>
            )}
          </div>

          {/* Right: Results Panel */}
          <div className="grid gap-6 section-fade-in">
            {/* Summary Stats */}
            {result && (
              <div className="grid md:grid-cols-3 gap-4">
                <StatCard
                  label="Domain credibility"
                  value={typeof result.domain_score === "number" ? result.domain_score : "-"}
                  unit={result.domain_score ? "/ 100" : ""}
                  icon="🏢"
                  trend={
                    typeof result.domain_score === "number"
                      ? result.domain_score >= 70
                        ? "up"
                        : result.domain_score >= 40
                        ? "neutral"
                        : "down"
                      : undefined
                  }
                />
                <StatCard
                  label="Misinformation risk"
                  value={fmtPct(result.final_misinformation_likelihood)}
                  icon="⚠️"
                  trend={
                    typeof result.final_misinformation_likelihood === "number"
                      ? result.final_misinformation_likelihood > 0.6
                        ? "down"
                        : result.final_misinformation_likelihood > 0.3
                        ? "neutral"
                        : "up"
                      : undefined
                  }
                />
                <StatCard
                  label="Claims analyzed"
                  value={result.claims?.length || 0}
                  unit="total"
                  icon="📊"
                />
              </div>
            )}

            {audienceMode === "user" && result && (
              <div className="glass-panel rounded-2xl border p-4 md:p-5 space-y-4">
                <div className="rounded-xl border border-cyan-400/20 bg-cyan-500/5 p-4">
                  <div className="text-xs uppercase tracking-wider text-cyan-200/90 mb-1">Verdict</div>
                  <div className="text-lg font-semibold text-white">
                    {primaryClaim ? `${primaryClaim.verdict} (${fmtPct(primaryClaim.confidence)})` : "No claim extracted yet"}
                  </div>
                  <div className="text-sm text-slate-300 mt-2">
                    {primaryClaim?.debate_summary || "Run a verification to get a plain-language explanation."}
                  </div>
                </div>

                <div className="grid md:grid-cols-3 gap-3">
                  <div className="rounded-lg border border-emerald-300/20 bg-emerald-500/5 p-3">
                    <div className="text-xs uppercase tracking-wider text-emerald-200/90 mb-1">Supports Claim</div>
                    <div className="text-sm text-slate-100">{fmtCount(supportEvidence.length)} sources</div>
                  </div>
                  <div className="rounded-lg border border-rose-300/20 bg-rose-500/5 p-3">
                    <div className="text-xs uppercase tracking-wider text-rose-200/90 mb-1">Refutes Claim</div>
                    <div className="text-sm text-slate-100">{fmtCount(refuteEvidence.length)} sources</div>
                  </div>
                  <div className="rounded-lg border border-slate-300/20 bg-slate-700/20 p-3">
                    <div className="text-xs uppercase tracking-wider text-slate-300 mb-1">Neutral Context</div>
                    <div className="text-sm text-slate-100">{fmtCount(neutralEvidence.length)} sources</div>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-300/20 bg-slate-900/45 p-4">
                  <div className="flex items-center justify-between gap-2 flex-wrap">
                    <div className="text-sm font-semibold text-white">Claim & Evidence Details</div>
                    <div className="text-xs text-slate-400">Show claim verdicts, sentiment, and source links</div>
                  </div>

                  {result.claims && result.claims.length > 0 ? (
                    <div className="mt-3 grid gap-3">
                      {result.claims.slice(0, 3).map((claim, idx) => (
                        <div key={`user-claim-${idx}`} className="rounded-lg border border-slate-300/20 bg-slate-800/40 p-3">
                          <div className="flex items-start justify-between gap-2 flex-wrap">
                            <div className="text-xs uppercase tracking-wider text-slate-400">Claim #{idx + 1}</div>
                            <div className="flex items-center gap-2 flex-wrap">
                              <VerdictBadge verdict={claim.verdict} confidence={claim.confidence} />
                              {claim.sentiment && (
                                <SentimentBadge
                                  label={claim.sentiment.label}
                                  score={claim.sentiment.score}
                                  emotionalIntensity={claim.sentiment.emotional_intensity}
                                  biasRisk={claim.sentiment.bias_risk}
                                />
                              )}
                            </div>
                          </div>

                          <p className="mt-2 text-sm text-slate-100 leading-relaxed">{claim.claim_text}</p>

                          {claim.sentiment && (
                            <div className="mt-2 text-xs text-slate-300">
                              Sentiment score: {claim.sentiment.score.toFixed(2)} | Emotional intensity: {Math.round(claim.sentiment.emotional_intensity * 100)}% | Bias risk: {claim.sentiment.bias_risk.toUpperCase()}
                            </div>
                          )}

                          <div className="mt-3 grid gap-2">
                            {sortEvidence(claim.evidence || []).slice(0, 3).map((ev, evIdx) => (
                              <a
                                key={`user-claim-${idx}-ev-${evIdx}`}
                                href={ev.url}
                                target="_blank"
                                rel="noreferrer"
                                className="rounded-lg border border-slate-300/20 bg-slate-900/55 px-3 py-2 hover:border-cyan-300/40 transition"
                              >
                                <div className="flex items-center justify-between gap-2">
                                  <div className="truncate text-sm font-medium text-cyan-100">{ev.domain || safeHostFromUrl(ev.url)}</div>
                                  <div
                                    className={cx(
                                      "px-2 py-0.5 text-[11px] rounded-full border uppercase tracking-wider",
                                      ev.stance === "support"
                                        ? "border-emerald-300/40 bg-emerald-400/15 text-emerald-100"
                                        : ev.stance === "refute"
                                        ? "border-rose-300/40 bg-rose-400/15 text-rose-100"
                                        : "border-slate-300/30 bg-slate-500/20 text-slate-200"
                                    )}
                                  >
                                    {ev.stance || "context"}
                                  </div>
                                </div>
                                <div className="mt-1 text-xs text-slate-300 line-clamp-2">{ev.snippet || "Open source link"}</div>
                              </a>
                            ))}
                            {(!claim.evidence || claim.evidence.length === 0) && (
                              <div className="text-xs text-slate-400">No evidence sources were returned for this claim.</div>
                            )}
                          </div>
                        </div>
                      ))}

                      {result.claims.length > 3 && (
                        <div className="text-xs text-slate-400">Showing top 3 claims in User View. Switch to Analyst View for all claim details and filters.</div>
                      )}
                    </div>
                  ) : (
                    <div className="mt-3 text-sm text-slate-400">No claim-level details available for this run.</div>
                  )}
                </div>
              </div>
            )}

            {/* Extracted Content Preview */}
            {result && audienceMode === "analyst" && (
              <div className="glass-panel rounded-2xl border p-4 md:p-5 stagger-card hover-lift" style={staggerStyle(1)}>
                <div className="flex items-start justify-between gap-2 mb-3">
                  <div className="text-sm font-semibold text-white">📄 Extracted Content</div>
                  <div className="text-xs text-slate-400 bg-slate-800/40 px-2 py-1 rounded">
                    {result.extracted_text_chars || 0} chars
                  </div>
                </div>
                <p className="text-sm leading-relaxed text-slate-200">{result.extracted_text_preview || "Content extraction pending..."}</p>
              </div>
            )}

            {/* Results Tabs */}
            {result && audienceMode === "analyst" && (
              <div className="glass-panel rounded-2xl border overflow-hidden stagger-card" style={staggerStyle(2)}>
                <Tabs
                  tabs={[
                    { label: "Claims", value: "claims", icon: "📋" },
                    { label: "Evidence", value: "evidence", icon: "📚" },
                    { label: "Details", value: "overview", icon: "🔍" },
                  ]}
                  activeTab={resultTab}
                  onTabChange={(v) => setResultTab(v as "overview" | "claims" | "evidence")}
                />

                <div className="p-5">
                  {resultTab === "claims" && (
                    <div>
                      {/* Sentiment Filter Controls */}
                      {result?.claims && result.claims.some((c) => c.sentiment) && (
                        <div className="mb-4 p-3 rounded-lg bg-slate-800/40 border border-slate-700/40">
                          <div className="text-xs uppercase tracking-wider text-slate-400 mb-3 font-semibold">Filter by sentiment</div>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                            <button
                              onClick={() => setSentimentFilter("all")}
                              className={cx(
                                "px-3 py-2 rounded-lg text-xs font-medium transition border",
                                sentimentFilter === "all"
                                  ? "bg-cyan-400/15 border-cyan-300/40 text-cyan-100"
                                  : "border-slate-300/20 text-slate-300 hover:border-slate-300/40"
                              )}
                            >
                              All claims
                            </button>
                            <button
                              onClick={() => setSentimentFilter("positive")}
                              className={cx(
                                "px-3 py-2 rounded-lg text-xs font-medium transition border",
                                sentimentFilter === "positive"
                                  ? "bg-blue-400/15 border-blue-300/40 text-blue-100"
                                  : "border-slate-300/20 text-slate-300 hover:border-slate-300/40"
                              )}
                            >
                              😊 Positive
                            </button>
                            <button
                              onClick={() => setSentimentFilter("negative")}
                              className={cx(
                                "px-3 py-2 rounded-lg text-xs font-medium transition border",
                                sentimentFilter === "negative"
                                  ? "bg-orange-400/15 border-orange-300/40 text-orange-100"
                                  : "border-slate-300/20 text-slate-300 hover:border-slate-300/40"
                              )}
                            >
                              😠 Negative
                            </button>
                            <button
                              onClick={() => setSentimentFilter("neutral")}
                              className={cx(
                                "px-3 py-2 rounded-lg text-xs font-medium transition border",
                                sentimentFilter === "neutral"
                                  ? "bg-slate-400/15 border-slate-300/40 text-slate-100"
                                  : "border-slate-300/20 text-slate-300 hover:border-slate-300/40"
                              )}
                            >
                              😐 Neutral
                            </button>
                          </div>
                          <div className="mt-3 text-xs uppercase tracking-wider text-slate-400 font-semibold">Bias risk level</div>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
                            <button
                              onClick={() => setBiasRiskFilter("all")}
                              className={cx(
                                "px-3 py-2 rounded-lg text-xs font-medium transition border",
                                biasRiskFilter === "all"
                                  ? "bg-cyan-400/15 border-cyan-300/40 text-cyan-100"
                                  : "border-slate-300/20 text-slate-300 hover:border-slate-300/40"
                              )}
                            >
                              All risk levels
                            </button>
                            <button
                              onClick={() => setBiasRiskFilter("low")}
                              className={cx(
                                "px-3 py-2 rounded-lg text-xs font-medium transition border",
                                biasRiskFilter === "low"
                                  ? "bg-emerald-400/15 border-emerald-300/40 text-emerald-100"
                                  : "border-slate-300/20 text-slate-300 hover:border-slate-300/40"
                              )}
                            >
                              🟢 Low risk
                            </button>
                            <button
                              onClick={() => setBiasRiskFilter("medium")}
                              className={cx(
                                "px-3 py-2 rounded-lg text-xs font-medium transition border",
                                biasRiskFilter === "medium"
                                  ? "bg-amber-400/15 border-amber-300/40 text-amber-100"
                                  : "border-slate-300/20 text-slate-300 hover:border-slate-300/40"
                              )}
                            >
                              🟡 Medium risk
                            </button>
                            <button
                              onClick={() => setBiasRiskFilter("high")}
                              className={cx(
                                "px-3 py-2 rounded-lg text-xs font-medium transition border",
                                biasRiskFilter === "high"
                                  ? "bg-rose-400/15 border-rose-300/40 text-rose-100"
                                  : "border-slate-300/20 text-slate-300 hover:border-slate-300/40"
                              )}
                            >
                              🔴 High risk
                            </button>
                          </div>
                        </div>
                      )}

                      <div className="grid gap-3">
                        {showResultSkeleton ? (
                          Array.from({ length: 3 }).map((_, i) => (
                            <div key={`claim-skeleton-${i}`} className="rounded-xl border border-slate-300/20 bg-slate-900/45 p-4 animate-pulse">
                              <div className="h-3 w-24 rounded bg-slate-700" />
                              <div className="h-4 w-full rounded bg-slate-700 mt-3" />
                              <div className="h-4 w-11/12 rounded bg-slate-700 mt-2" />
                          </div>
                        ))
                      ) : result.claims && result.claims.length > 0 ? (
                        (() => {
                          const filteredClaims = filterClaimsBySentiment(result.claims, sentimentFilter, biasRiskFilter);
                          return filteredClaims.length > 0 ? (
                            filteredClaims.map((c, idx) => {
                              const open = openClaimIdx === idx;
                              return (
                                <div key={idx} className="rounded-xl border border-slate-300/20 bg-slate-900/45 overflow-hidden stagger-card hover-lift" style={staggerStyle(idx, 50)}>
                                  <button
                                    className="w-full p-4 text-left hover:bg-slate-900/60 transition"
                                onClick={() => setOpenClaimIdx(open ? null : idx)}
                              >
                                <div className="flex items-start justify-between gap-3">
                                  <div className="flex-1 min-w-0">
                                    <div className="text-xs uppercase tracking-wider text-slate-400">Claim #{idx + 1}</div>
                                    <div className="mt-1 text-sm text-white leading-relaxed">{c.claim_text}</div>
                                    <div className="mt-2 flex flex-wrap items-center gap-2">
                                      <VerdictBadge verdict={c.verdict} confidence={c.confidence} />
                                      {c.sentiment && (
                                        <SentimentBadge
                                          label={c.sentiment.label}
                                          score={c.sentiment.score}
                                          emotionalIntensity={c.sentiment.emotional_intensity}
                                          biasRisk={c.sentiment.bias_risk}
                                        />
                                      )}
                                      {c.needs_human_review && (
                                        <span className="px-2 py-1 text-xs rounded-full border border-amber-300/30 bg-amber-300/10 text-amber-100">
                                          👤 Human review
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                  <div className="text-slate-400 text-xs whitespace-nowrap">{open ? "↑" : "↓"}</div>
                                </div>
                              </button>

                              {open && (
                                <div className="border-t border-slate-300/20 p-4 bg-slate-900/30">
                                  {c.debate_summary && (
                                    <div className="mb-4 p-3 rounded-lg bg-purple-400/10 border border-purple-300/20">
                                      <div className="text-xs font-semibold text-purple-200 mb-1">⚖️ Debate Summary</div>
                                      <p className="text-sm text-purple-100/80">{c.debate_summary}</p>
                                    </div>
                                  )}

                                  <div className="grid md:grid-cols-2 gap-3 mb-4 text-xs">
                                    <div className="rounded-lg border border-slate-300/20 bg-slate-800/40 p-3">
                                      <div className="font-semibold text-slate-100 mb-2">📊 Claim Profile</div>
                                      <div className="space-y-1 text-slate-300">
                                        <div>Expertise: {c.claim_profile?.expertise_profile || "general"}</div>
                                        <div>Entities: {c.claim_profile?.entities?.join(", ") || "-"}</div>
                                        <div>Numbers: {c.claim_profile?.numbers?.join(", ") || "-"}</div>
                                      </div>
                                    </div>
                                    <div className="rounded-lg border border-slate-300/20 bg-slate-800/40 p-3">
                                      <div className="font-semibold text-slate-100 mb-2">🔒 Trust Signals</div>
                                      <div className="space-y-1 text-slate-300">
                                        <div>High credibility: {c.evidence_summary?.high_credibility_sources || 0}</div>
                                        <div>Primary sources: {c.evidence_summary?.primary_source_count || 0}</div>
                                        <div>Conflict level: {c.evidence_summary?.conflict_level || "low"}</div>
                                      </div>
                                    </div>
                                  </div>

                                  {c.sentiment && (
                                    <div className="rounded-lg border border-slate-300/20 bg-slate-800/40 p-3 mb-4 text-xs">
                                      <div className="font-semibold text-slate-100 mb-2">😊 Sentiment Analysis</div>
                                      <div className="space-y-2 text-slate-300">
                                        <div className="flex items-center justify-between">
                                          <span>Sentiment:</span>
                                          <SentimentBadge
                                            label={c.sentiment.label}
                                            score={c.sentiment.score}
                                            emotionalIntensity={c.sentiment.emotional_intensity}
                                            biasRisk={c.sentiment.bias_risk}
                                          />
                                        </div>
                                        <div>Emotional intensity: {Math.round(c.sentiment.emotional_intensity * 100)}%</div>
                                        <div>
                                          Bias risk:{" "}
                                          <span
                                            className={
                                              c.sentiment.bias_risk === "high"
                                                ? "text-rose-400"
                                                : c.sentiment.bias_risk === "medium"
                                                ? "text-amber-400"
                                                : "text-emerald-400"
                                            }
                                          >
                                            {c.sentiment.bias_risk.toUpperCase()}
                                          </span>
                                        </div>
                                        {c.sentiment.manipulation_flags && c.sentiment.manipulation_flags.length > 0 && (
                                          <div className="pt-2 border-t border-slate-600">
                                            <div className="font-semibold text-amber-300 mb-1">⚠️ Detected manipulation:</div>
                                            <div className="space-y-1">
                                              {c.sentiment.manipulation_flags.map((flag, i) => (
                                                <div key={i} className="text-amber-200">
                                                  • {flag}
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  )}

                                  {c.uncertainty_reasons && c.uncertainty_reasons.length > 0 && (
                                    <div className="rounded-lg border border-amber-300/30 bg-amber-300/10 p-3 text-xs text-amber-100">
                                      <div className="font-semibold mb-2">⚡ Uncertainty Signals</div>
                                      <ul className="list-disc pl-5 space-y-1">
                                        {c.uncertainty_reasons.map((r, i) => (
                                          <li key={i}>{r}</li>
                                        ))}
                                      </ul>
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                          );
                            })
                          ) : (
                            <div className="text-center py-8 text-slate-400">
                              No claims match the selected filters.
                            </div>
                          );
                        })()
                      ) : (
                        <div className="text-center py-8 text-slate-400">
                          No claims found. Try analyzing longer text or a different source.
                        </div>
                      )}
                      </div>
                    </div>
                  )}

                  {resultTab === "evidence" && (
                    <div className="grid gap-3">
                      {result.claims && result.claims.length > 0 ? (
                        result.claims.flatMap((c, cIdx) =>
                          sortEvidence(c.evidence || []).slice(0, 3).map((ev, eIdx) => {
                            const score = evScore(ev);
                            const quality = evQuality(ev);
                            return (
                              <a
                                key={`${cIdx}-${eIdx}`}
                                href={ev.url}
                                target="_blank"
                                rel="noreferrer"
                                className="rounded-lg border border-slate-300/20 bg-slate-900/60 p-3 stagger-card hover-lift hover:border-cyan-300/40 transition"
                                style={staggerStyle(eIdx, 40)}
                              >
                                <div className="flex items-start justify-between gap-2">
                                  <div className="flex-1 min-w-0">
                                    <div className="font-semibold text-slate-100 text-sm truncate">{ev.domain || safeHostFromUrl(ev.url)}</div>
                                    <div className="text-xs text-slate-300 mt-1">{ev.snippet?.slice(0, 100)}...</div>
                                    <div className="flex flex-wrap gap-1 mt-2">
                                      <ScoreBadge score={score} maxScore={100} label="Domain" variant={score >= 70 ? "good" : score >= 40 ? "warn" : "bad"} />
                                      {typeof quality === "number" && (
                                        <ScoreBadge score={quality} maxScore={100} label="Quality" variant={quality >= 70 ? "good" : "warn"} />
                                      )}
                                      {ev.primary_source && <span className="px-2 py-0.5 text-xs rounded-full bg-emerald-300/20 text-emerald-200 border border-emerald-300/30">Primary</span>}
                                    </div>
                                  </div>
                                  <div className="text-cyan-400 text-lg">→</div>
                                </div>
                              </a>
                            );
                          })
                        )
                      ) : (
                        <div className="text-center py-8 text-slate-400">No evidence found.</div>
                      )}
                    </div>
                  )}

                  {resultTab === "overview" && result && (
                    <div className="space-y-4">
                      <div className="grid gap-3">
                        <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                          <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">Analysis Type</div>
                          <div className="text-sm text-slate-100">{result.input_type === "url" ? "🔗 URL Analysis" : "📝 Text Analysis"}</div>
                        </div>
                        <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                          <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">Processing Time</div>
                          <div className="text-sm text-slate-100">{processingTime > 0 ? `${(processingTime / 1000).toFixed(1)}s` : "Pending..."}</div>
                        </div>
                        <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                          <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">Domain</div>
                          <div className="text-sm text-slate-100 font-mono">{result.domain || "N/A"}</div>
                        </div>
                        <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                          <div className="text-xs uppercase tracking-wide text-slate-400 mb-1">Timestamp</div>
                          <div className="text-xs text-slate-300 font-mono">{result.timestamp_utc || "N/A"}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Empty State */}
            {!result && !loading && (
              <div className="glass-panel rounded-2xl border p-8 md:p-12 text-center stagger-card">
                <div className="text-4xl mb-4">🔍</div>
                <h3 className="text-lg font-semibold text-slate-200 mb-2">No analysis yet</h3>
                <p className="text-sm text-slate-400 max-w-sm mx-auto">
                  Enter a URL or paste text in the left panel to get started with professional-grade fact verification.
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Run History */}
        <section className="glass-panel rounded-2xl border p-4 section-fade-in order-3">
          <details>
            <summary className="cursor-pointer select-none flex items-center justify-between gap-3 text-sm font-semibold text-white">
              <span>📊 Recent verification</span>
              <span className="text-xs text-slate-400">{fmtCount(runs.length)} records</span>
            </summary>

            <div className="mt-4">
              <div className="flex items-center justify-end flex-wrap gap-3 mb-3">
                <button
                  className="rounded-lg border border-slate-300/20 px-3 py-1.5 text-xs hover:border-cyan-300/40 transition disabled:opacity-50"
                  onClick={fetchRuns}
                  disabled={runsLoading}
                >
                  {runsLoading ? "Refreshing..." : "Refresh"}
                </button>
              </div>

              {runsLoading && runs.length === 0 ? (
                <div className="grid gap-2">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <div key={`run-skeleton-${i}`} className="h-10 rounded-lg bg-slate-700/40 animate-pulse" />
                  ))}
                </div>
              ) : runsError ? (
                <Alert type="warn" message={runsError} title="Could not load history" />
              ) : runs.length === 0 ? (
                <div className="text-center py-6 text-slate-400 text-sm">No verification runs yet</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="bg-slate-900/70 text-slate-300 border-b border-slate-700">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-semibold">#</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold">Time</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold">Type</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold">Domain</th>
                        <th className="px-4 py-3 text-left text-xs font-semibold">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {runs.map((r, i) => (
                        <tr key={`${r.id}_${i}`} className="border-t border-slate-700/50 hover:bg-slate-900/50 transition">
                          <td className="px-4 py-3 text-slate-200">{r.id}</td>
                          <td className="px-4 py-3 text-xs text-slate-400 font-mono">{fmtDateTime(r.time_utc || r.timestamp_utc)}</td>
                          <td className="px-4 py-3 text-slate-300">{(r.input_type || r.type || "-").slice(0, 10)}</td>
                          <td className="px-4 py-3 text-slate-300 max-w-[150px] truncate">{r.domain || safeHostFromUrl(r.url) || "-"}</td>
                          <td className="px-4 py-3">
                            {r.url ? (
                              <a
                                className="text-xs px-2 py-1.5 rounded-lg border border-slate-300/20 text-slate-200 hover:border-cyan-300/40 transition"
                                href={r.url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                Open
                              </a>
                            ) : (
                              "-"
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </details>
        </section>
      </div>
    </main>
  );
}
